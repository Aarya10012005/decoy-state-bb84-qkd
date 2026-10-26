import unittest
from dataclasses import replace

import numpy as np

from src.channel import simulate_channel_and_detection
from src.config import QKDConfig
from src.decoy_estimation import count_by_intensity, estimate_decoy_parameters
from src.finite_key import chernoff_lower, chernoff_upper, estimate_decoy_parameters_finite
from src.key_rate import binary_entropy, estimate_key_rate
from src.state_preparation import prepare_states
from src.theory import expected_estimates, true_single_photon


def run(config, seed=0):
    prepared = prepare_states(config, seed=seed)
    measured = simulate_channel_and_detection(prepared, config, seed=seed + 1)
    return prepared, measured


class ConfigTest(unittest.TestCase):
    def test_default_is_valid(self):
        QKDConfig().validate()

    def test_rejects_bad_intensity_ordering(self):
        with self.assertRaises(ValueError):
            QKDConfig(mu_decoy=0.6).validate()

    def test_rejects_bad_probabilities(self):
        with self.assertRaises(ValueError):
            QKDConfig(intensity_prob_signal=0.9).validate()


class ChannelTest(unittest.TestCase):
    def test_dark_counts_give_random_bits(self):
        # No photons ever arrive, so every click is a dark count with e0 = 1/2.
        config = QKDConfig(num_rounds=200_000, detector_efficiency=0.0, dark_count_prob=0.05)
        prepared, measured = run(config)
        sifted = measured.clicks & (measured.bob_bases == prepared.bases)
        qber = (measured.bob_bits[sifted] != prepared.bits[sifted]).mean()
        self.assertAlmostEqual(qber, 0.5, delta=0.02)

    def test_gains_match_theory(self):
        config = QKDConfig(num_rounds=500_000)
        prepared, measured = run(config)
        theory = expected_estimates(config)
        counts = count_by_intensity(prepared, measured)
        for label in ("signal", "decoy"):
            self.assertAlmostEqual(counts[label].gain / theory.gain[label], 1.0, delta=0.05)


class DecoyEstimationTest(unittest.TestCase):
    def test_theory_bounds_are_valid(self):
        config = QKDConfig()
        theory = expected_estimates(config)
        y1, e1 = true_single_photon(config)
        self.assertLessEqual(theory.y1_lower, y1)
        self.assertGreaterEqual(theory.e1_upper, e1)
        self.assertGreater(theory.y1_lower, 0.9 * y1)

    def test_finite_is_more_conservative(self):
        config = QKDConfig()
        prepared, measured = run(config)
        asym = estimate_decoy_parameters(prepared, measured, config.mu_signal, config.mu_decoy)
        finite = estimate_decoy_parameters_finite(
            prepared, measured, config.mu_signal, config.mu_decoy, config.eps_parameter_estimation
        )
        self.assertLessEqual(finite.y1_lower, asym.y1_lower)
        self.assertGreaterEqual(finite.e1_upper, asym.e1_upper)
        self.assertLessEqual(finite.y0_lower, asym.y0_lower)


class ChernoffTest(unittest.TestCase):
    def test_bounds_bracket_count(self):
        for k in (0, 1, 10, 1000):
            self.assertLessEqual(chernoff_lower(k, 1e-6), k)
            self.assertGreaterEqual(chernoff_upper(k, 1e-6), k)

    def test_coverage(self):
        rng = np.random.default_rng(3)
        mean, eps = 50.0, 0.05
        counts = rng.poisson(mean, size=20_000)
        misses_low = np.mean([chernoff_lower(k, eps) > mean for k in counts])
        misses_high = np.mean([chernoff_upper(k, eps) < mean for k in counts])
        self.assertLess(misses_low, eps)
        self.assertLess(misses_high, eps)


class KeyRateTest(unittest.TestCase):
    def test_binary_entropy(self):
        self.assertEqual(binary_entropy(0.0), 0.0)
        self.assertAlmostEqual(binary_entropy(0.5), 1.0)

    def test_rate_decreases_with_loss(self):
        config = QKDConfig()
        q = config.basis_match_prob()
        rates = [
            estimate_key_rate(
                expected_estimates(replace(config, channel_loss_db=loss)),
                mu_signal=config.mu_signal,
                f_ec=config.error_correction_efficiency,
                sifting_factor=q,
            ).key_rate
            for loss in (0.0, 10.0, 20.0, 30.0)
        ]
        self.assertTrue(all(a > b > 0 for a, b in zip(rates, rates[1:])))


if __name__ == "__main__":
    unittest.main()
