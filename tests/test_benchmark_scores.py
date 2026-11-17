import unittest
import torch
from torch import nn
from src.benchmark_scores import odin, benchmark_scores


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.BatchNorm1d(8),
                                 nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 3)).eval()
        self.x = torch.rand(2, 3, 2, 2)

    def test_odin_zero_step_equals_temperature_scaled_msp(self):
        expected = 1 - (self.net(self.x) / 10).softmax(1).max(1).values
        torch.testing.assert_close(odin(self.net, self.x, 10, 0), expected)

    def test_odin_step_increases_confidence_locally(self):
        before = odin(self.net, self.x, 1, 0)
        after = odin(self.net, self.x, 1, 1e-5)
        self.assertTrue(torch.all(after <= before + 1e-6))
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))

    def test_msp_is_deterministic_prediction_confidence(self):
        expected = 1 - self.net(self.x).softmax(1).max(1).values
        _, scores = benchmark_scores(self.net, self.x)
        torch.testing.assert_close(scores['msp_warning'], expected)


if __name__ == '__main__':
    unittest.main()
