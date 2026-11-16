"""CPU checks loaded directly so CUDA engine dependencies are not required."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


def load_module(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SamplingParams = load_module("sampling_params_under_test", "myvllm/sampling_params.py").SamplingParams
Sampler = load_module("sampler_under_test", "myvllm/layers/sampler.py").Sampler


class SamplingTests(unittest.TestCase):
    def sample(self, logits, temperatures):
        # Exercise the tensor implementation without invoking a compiler on CPU.
        return Sampler.forward.__wrapped__(Sampler(), logits, temperatures)

    def test_greedy_and_mixed_batch(self):
        logits = torch.tensor([[1., 9., 3.], [8., 2., 1.], [1., 2., 7.]])
        original = logits.clone()
        for seed in range(5):
            torch.manual_seed(seed)
            tokens = self.sample(logits, torch.tensor([0., 0.7, 0.]))
            self.assertEqual(tokens[[0, 2]].tolist(), [1, 2])
            self.assertIn(tokens[1].item(), range(3))
        torch.testing.assert_close(logits, original)

    def test_positive_temperature_preserves_sampling_algorithm(self):
        logits = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
        temperatures = torch.tensor([0.5, 1.5])
        torch.manual_seed(42)
        probs = torch.softmax(logits / temperatures[:, None], dim=-1)
        expected = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)
        torch.manual_seed(42)
        torch.testing.assert_close(self.sample(logits, temperatures), expected)

    def test_parameter_validation(self):
        self.assertEqual(SamplingParams(temperature=0).temperature, 0)
        for temperature in [-1, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                SamplingParams(temperature=temperature)
        for max_tokens in [0, -1, 1.5, True]:
            with self.subTest(max_tokens=max_tokens), self.assertRaises(ValueError):
                SamplingParams(max_tokens=max_tokens)


if __name__ == "__main__":
    unittest.main()
