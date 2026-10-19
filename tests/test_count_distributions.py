import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sclamp_sr.distributions import negative_binomial_from_mean_dispersion


class NegativeBinomialParameterizationTest(unittest.TestCase):
    def test_distribution_mean_matches_decoder_mean(self):
        mean = torch.tensor([[0.1, 1.0, 10.0], [3.0, 7.5, 25.0]])
        dispersion = torch.tensor([[0.5, 2.0, 4.0], [1.5, 8.0, 12.0]])

        distribution = negative_binomial_from_mean_dispersion(mean, dispersion)

        torch.testing.assert_close(distribution.mean, mean, rtol=1e-5, atol=1e-6)

    def test_distribution_variance_matches_mean_dispersion_form(self):
        mean = torch.tensor([0.25, 2.0, 12.0])
        dispersion = torch.tensor([0.75, 3.0, 5.0])

        distribution = negative_binomial_from_mean_dispersion(mean, dispersion)
        expected = mean + mean.square() / dispersion

        torch.testing.assert_close(distribution.variance, expected, rtol=1e-5, atol=1e-6)

    def test_log_probability_has_finite_gradients(self):
        mean = torch.tensor([0.5, 4.0, 15.0], requires_grad=True)
        dispersion = torch.tensor([1.0, 2.5, 6.0], requires_grad=True)
        target = torch.tensor([0.0, 3.0, 17.0])

        loss = -negative_binomial_from_mean_dispersion(mean, dispersion).log_prob(target).mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(dispersion.grad).all())

    def test_model_modules_use_only_the_shared_constructor(self):
        package = ROOT / "sclamp_sr"
        offenders = []
        for path in package.glob("*.py"):
            if path.name == "distributions.py":
                continue
            if "NegativeBinomial(" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
