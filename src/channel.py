from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import QKDConfig
from .state_preparation import PreparedState


@dataclass
class MeasurementResult:
    bob_bases: np.ndarray
    clicks: np.ndarray
    bob_bits: np.ndarray


def _transmittance(config: QKDConfig) -> float:
    channel_eta = 10 ** (-config.channel_loss_db / 10.0)
    return channel_eta * config.detector_efficiency


def simulate_channel_and_detection(
    prepared: PreparedState, config: QKDConfig, seed: int | None = None
) -> MeasurementResult:
    config.validate()
    rng = np.random.default_rng(seed)

    bob_bases = rng.random(prepared.bits.size) < config.bob_basis_prob_z
    eta = _transmittance(config)

    no_click_prob = (1.0 - config.dark_count_prob) * (1.0 - eta) ** prepared.photon_numbers
    clicks = rng.random(prepared.bits.size) >= no_click_prob

    bob_bits = np.full(prepared.bits.size, -1, dtype=np.int8)
    same_basis = bob_bases == prepared.bases
    detected = clicks

    error_mask = rng.random(prepared.bits.size) < config.misalignment_error
    correct_bits = np.where(error_mask, 1 - prepared.bits, prepared.bits)
    random_bits = rng.integers(0, 2, size=prepared.bits.size, dtype=np.int8)

    bob_bits[detected & same_basis] = correct_bits[detected & same_basis]
    bob_bits[detected & ~same_basis] = random_bits[detected & ~same_basis]

    return MeasurementResult(bob_bases=bob_bases, clicks=clicks, bob_bits=bob_bits)
