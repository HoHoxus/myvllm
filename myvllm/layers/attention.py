import torch
from torch import nn
import triton
import triton.language as tl

from myvllm.utils.context import get_context


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    batch_idx = tl.program_id(0)
    head_idx = tl.program - id(1)

    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)

    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdix(max_num_blocks * block_size, BLOCK_N)

    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N

        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            logical_block = offs_n // block_size

            offs_in_block = offs_ % block_size

            in_range = (offs_n < context_len) & (logical_block < max_num_blocks)

            physical_block = tl.load(
                block_tables_ptr + batch_idx * max_num_blocks + logical_block,
                mask=in_range,
                other=-1,
            )
            valid = in_range & (physical_block != -1)

            physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

            kv_offset = (
                physical_blcok[None, :] * (block_size * num_kv_heads * head_dim)
                + offs_in_block[None, :] * (num_kv_heads * head_dim)
                + kv_head_idx * head_dim
                + offs_d[:, None]
            )
            k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            k = tl.cast(k, tl.float32)
            score = tl.sum(ql[:, None] * k, axis=0) * scale

            qk = tl.where(valid, score, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i, m_i_new)
            p = tl.exp(qk - m_i_new)

            acc = acc * alpha
            l_i = l_i * alpha

            v = tl.load(v_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            v = tl.cast(v, tl.float32)
            weight = tl.where(valid, p, 0.0)
            acc = acc + tl.sum(weight[None, :] * v, axis=1)
            l_i = l_i + tl.sum(weight)

            m_i = m_i_new
    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d

    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
):
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    # 保证kernel寻址连续
    query = query.contiguous()

    output = torch.empty_like(query)

    BLOCK_N = 64 if head_dim <= 128 else 32

    grid = (batch_size, num_heads)
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    return output


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor

    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    output = torch.empty_like(q)

    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    num_seqs = cu_seqlens.shape[0] - 1
    if num_seqs <= 0:
        return output

    cu_seqlens_cpu = cu_seqlens.cpu()
    seq_lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    max_seq_len = int(seq_lens.max().item())
    if max_seq_len <= 0:
        return output

    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    flash_attention_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


@triton.jit
def flash_attention_varlen_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simple Flash Attention Kernel (varlen, causal)
    """
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)

    kv_head_idx = off_h // (num_heads // num_kv_heads)

    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start  # 修复：原为 seq_start - seq_end

    if start_m * BLOCK_M >= seq_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    mask_m = offs_m < seq_len

    # ---- Q ----
    # q: (total_tokens, num_heads, head_dim)
    q_ptrs = (
        Q
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # online softmax initialize
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1e10, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    num_blocks = tl.cdiv(tl.minimum(seq_len, (start_m + 1) * BLOCK_M), BLOCK_N)

    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        # ---- K: (total_tokens, num_kv_heads, head_dim) ----
        k_ptrs = (
            K
            + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + offs_d[:, None]
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        qk = tl.dot(q, k) * scale

        mask_causal = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        # ---- online softmax ----
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]

        # ---- V: (total_tokens, num_kv_heads, head_dim) ----
        v_ptrs = (
            V
            + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        acc = acc + tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_i_safe[:, None]

    # ---- O ----
    o_ptrs = (
        O
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attention_prefill(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        # TODO:else:  # paged attention decode

        return o
