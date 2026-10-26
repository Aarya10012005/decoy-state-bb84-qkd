"""Closed-form expectations for the simulated channel model, used to validate
the Monte Carlo simulation and to draw smooth reference curves."""

from __future__ import annotations

import numpy as np

from .config import QKDConfig
from .decoy_estimation import E0, DecoyEstimates, IntensityCounts, decoy_bounds
from .finite_key import finite_estimates_from_counts


def expected_gain(mu: float, config: QKDConfig) -> float:
    eta = config.transmittance()
    return 1.0 - (1.0 - config.dark_count_prob) * np.exp(-eta * mu)


def expected_error_gain(mu: float, config: QKDConfig) -> float:
    """E*Q: probability of an erroneous click per pulse in the matching basis."""
    eta = config.transmittance()
    photon_click = 1.0 - np.exp(-eta * mu)
    dark_only = config.dark_count_prob * np.exp(-eta * mu)
    return config.misalignment_error * photon_click + E0 * dark_only


def true_single_photon(config: QKDConfig) -> tuple[float, float]:
    """Actual (Y1, e1) of the channel model, which decoy bounds approximate."""
    eta = config.transmittance()
    p_d = config.dark_count_prob
    y1 = eta + p_d * (1.0 - eta)
    e1 = (config.misalignment_error * eta + E0 * p_d * (1.0 - eta)) / y1
    return y1, e1


def expected_estimates(config: QKDConfig) -> DecoyEstimates:
    """Decoy estimates in the infinite-statistics limit."""
    gain = {}
    qber = {}
    for label, mu in config.mu_map().items():
        gain[label] = expected_gain(mu, config)
        qber[label] = expected_error_gain(mu, config) / gain[label] if gain[label] else 0.0

    y0 = gain["vacuum"]
    y1_lower, e1_upper = decoy_bounds(
        config.mu_signal,
        config.mu_decoy,
        q_mu_high=gain["signal"],
        q_nu_low=gain["decoy"],
        y0_high=y0,
        y0_low=y0,
        eq_nu_high=expected_error_gain(config.mu_decoy, config),
    )
    return DecoyEstimates(
        gain=gain, qber=qber, y0_lower=y0, y1_lower=y1_lower, e1_upper=e1_upper
    )


def expected_counts(config: QKDConfig) -> dict[str, IntensityCounts]:
    """Expected per-intensity counts for `config.num_rounds` pulses."""
    q = config.basis_match_prob()
    probs = {
        "signal": config.intensity_prob_signal,
        "decoy": config.intensity_prob_decoy,
        "vacuum": config.intensity_prob_vacuum,
    }
    counts = {}
    for label, mu in config.mu_map().items():
        pulses = config.num_rounds * probs[label]
        gain = expected_gain(mu, config)
        counts[label] = IntensityCounts(
            pulses=pulses,
            clicks=pulses * gain,
            matched_pulses=pulses * q,
            sifted=pulses * q * gain,
            errors=pulses * q * expected_error_gain(mu, config),
        )
    return counts


def expected_finite_estimates(config: QKDConfig) -> DecoyEstimates:
    """Finite-size decoy estimates if the expected counts were observed: a
    smooth, noise-free model of `estimate_decoy_parameters_finite`."""
    return finite_estimates_from_counts(
        expected_counts(config),
        config.mu_signal,
        config.mu_decoy,
        config.eps_parameter_estimation,
    )
