import unittest
from dataclasses import replace

import numpy as np

from src.attacks import InterceptResend, PhotonNumberSplitting, analyze
from src.channel import simulate_channel_and_detection
from src.config import QKDConfig
from src.optimization import expected_key_rate, optimize_parameters
from src.postprocessing import cascade, run_postprocessing, toeplitz_hash
from src.state_preparation import prepare_states
from src.theory import expected_gain

try:
    from src.qiskit_bb84 import run_qiskit_bb84

    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False


class AttackTest(unittest.TestCase):
    def test_intercept_resend_raises_qber(self):
        config = QKDConfig(num_rounds=300_000)
        report = analyze(config, InterceptResend(1.0))
        self.assertAlmostEqual(report.qber_signal, 0.25 + config.misalignment_error / 2, delta=0.02)
        self.assertEqual(report.decoy_secret_fraction, 0.0)

    def test_pns_hides_from_signal_gain(self):
        config = QKDConfig(channel_loss_db=15.0)
        attack = PhotonNumberSplitting.calibrated(config)
        report = analyze(config, attack)
        self.assertAlmostEqual(report.gain_signal / expected_gain(config.mu_signal, config), 1.0, delta=0.08)

    def test_pns_breaks_naive_but_not_decoy(self):
        config = QKDConfig(channel_loss_db=15.0)
        report = analyze(config, PhotonNumberSplitting.calibrated(config))
        secret = 1.0 - report.eve_known_fraction
        self.assertGreater(report.naive_secret_fraction, secret)
        self.assertLessEqual(report.decoy_secret_fraction, secret)

    def test_honest_channel_has_no_eve_knowledge(self):
        config = QKDConfig(num_rounds=100_000)
        prepared = prepare_states(config, seed=1)
        measured = simulate_channel_and_detection(prepared, config, seed=2)
        self.assertFalse(measured.eve_knows.any())


class OptimizationTest(unittest.TestCase):
    def test_optimum_beats_default(self):
        for loss in (3.0, 6.0):
            config = QKDConfig(channel_loss_db=loss)
            result = optimize_parameters(config)
            self.assertGreaterEqual(result.key_rate, expected_key_rate(config) * 0.999)
            result.config.validate()

    def test_optimum_unlocks_key_at_6db(self):
        config = QKDConfig(channel_loss_db=6.0)
        self.assertEqual(expected_key_rate(config), 0.0)
        self.assertGreater(optimize_parameters(config).key_rate, 0.0)


class PostProcessingTest(unittest.TestCase):
    def test_cascade_corrects_errors(self):
        rng = np.random.default_rng(0)
        alice = rng.integers(0, 2, size=5000, dtype=np.int8)
        bob = alice ^ (rng.random(5000) < 0.03).astype(np.int8)
        result = cascade(alice, bob, qber=0.03)
        self.assertTrue(np.array_equal(result.corrected, alice))
        self.assertGreater(result.leaked_bits, 0)

    def test_toeplitz_matches_matrix_product(self):
        rng = np.random.default_rng(1)
        n, m = 40, 12
        bits = rng.integers(0, 2, size=n)
        seed = rng.integers(0, 2, size=n + m - 1)
        matrix = np.array([[seed[i - j + n - 1] for j in range(n)] for i in range(m)])
        self.assertTrue(np.array_equal(toeplitz_hash(bits, m, seed), matrix @ bits % 2))

    def test_pipeline_gives_identical_keys(self):
        config = QKDConfig()
        prepared = prepare_states(config, seed=7)
        measured = simulate_channel_and_detection(prepared, config, seed=11)
        result = run_postprocessing(config, prepared, measured)
        self.assertTrue(result.verified)
        self.assertGreater(result.key_length, 0)
        self.assertTrue(np.array_equal(result.alice_key, result.bob_key))


@unittest.skipUnless(HAVE_QISKIT, "qiskit / qiskit-aer not installed")
class QiskitTest(unittest.TestCase):
    def test_honest_qber_is_misalignment(self):
        config = QKDConfig()
        result = run_qiskit_bb84(config, num_rounds=8000)
        self.assertAlmostEqual(result.qber, config.misalignment_error, delta=0.01)

    def test_eavesdropper_qber(self):
        config = QKDConfig()
        result = run_qiskit_bb84(config, num_rounds=8000, eavesdropper=True)
        self.assertAlmostEqual(result.qber, 0.255, delta=0.03)


if __name__ == "__main__":
    unittest.main()
