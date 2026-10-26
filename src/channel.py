from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import QKDConfig
from .state_preparation import PreparedState


@dataclass
class ArrivingState:
    """Optical state reaching Bob's detector for each pulse."""

    photons: np.ndarray
    bits: np.ndarray
    bases: np.ndarray
    eve_knows: np.ndarray


@dataclass
class MeasurementResult:
    bob_bases: np.ndarray
    clicks: np.ndarray
    bob_bits: np.ndarray
    eve_knows: np.ndarray


class Channel(Protocol):
    def transmit(
        self, prepared: PreparedState, config: QKDConfig, rng: np.random.Generator
    ) -> ArrivingState: ...


def channel_transmittance(config: QKDConfig) -> float:
    """Fibre transmittance alone, excluding the detector efficiency."""
    return 10 ** (-config.channel_loss_db / 10.0)


class HonestChannel:
    """Lossy fibre: each photon survives independently."""

    def transmit(
        self, prepared: PreparedState, config: QKDConfig, rng: np.random.Generator
    ) -> ArrivingState:
        return ArrivingState(
            photons=rng.binomial(prepared.photon_numbers, channel_transmittance(config)),
            bits=prepared.bits,
            bases=prepared.bases,
            eve_knows=np.zeros(prepared.bits.size, dtype=bool),
        )


def _detect(
    arriving: ArrivingState, config: QKDConfig, rng: np.random.Generator
) -> MeasurementResult:
    n = arriving.bits.size
    bob_bases = rng.random(n) < config.bob_basis_prob_z

    # A click is caused either by at least one detected photon or by a dark count.
    signal_click = rng.random(n) >= (1.0 - config.detector_efficiency) ** arriving.photons
    dark_click = rng.random(n) < config.dark_count_prob
    clicks = signal_click | dark_click

    same_basis = bob_bases == arriving.bases
    error_mask = rng.random(n) < config.misalignment_error
    aligned_bits = np.where(error_mask, 1 - arriving.bits, arriving.bits)
    random_bits = rng.integers(0, 2, size=n, dtype=np.int8)

    # Photon detections in the basis of the arriving state carry its bit (up to
    # misalignment); dark-count-only clicks and mismatched detections are random.
    informative = signal_click & same_basis
    bob_bits = np.full(n, -1, dtype=np.int8)
    bob_bits[clicks] = random_bits[clicks]
    bob_bits[informative] = aligned_bits[informative]

    return MeasurementResult(
        bob_bases=bob_bases, clicks=clicks, bob_bits=bob_bits, eve_knows=arriving.eve_knows
    )


def simulate_channel_and_detection(
    prepared: PreparedState,
    config: QKDConfig,
    seed: int | None = None,
    channel: Channel | None = None,
) -> MeasurementResult:
    """Send the prepared pulses through `channel` (honest fibre by default,
    or an eavesdropping attack from `src.attacks`) and detect them at Bob."""
    config.validate()
    rng = np.random.default_rng(seed)
    arriving = (channel or HonestChannel()).transmit(prepared, config, rng)
    return _detect(arriving, config, rng)
