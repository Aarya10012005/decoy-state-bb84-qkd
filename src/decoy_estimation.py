from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import MeasurementResult
from .state_preparation import INTENSITY_LABELS, PreparedState

# Error rate of vacuum (dark-count) events: Bob's bit is uniformly random.
E0 = 0.5


@dataclass
class DecoyEstimates:
    gain: dict[str, float]
    qber: dict[str, float]
    y0_lower: float
    y1_lower: float
    e1_upper: float


@dataclass
class IntensityCounts:
    """Raw counts for one intensity class."""

    pulses: int
    clicks: int
    matched_pulses: int
    sifted: int
    errors: int

    @property
    def gain(self) -> float:
        return self.clicks / self.pulses if self.pulses else 0.0

    @property
    def qber(self) -> float:
        return self.errors / self.sifted if self.sifted else 0.0

    @property
    def error_gain(self) -> float:
        """E*Q: errors per pulse sent in the matching basis."""
        return self.errors / self.matched_pulses if self.matched_pulses else 0.0


def count_by_intensity(
    prepared: PreparedState, measured: MeasurementResult
) -> dict[str, IntensityCounts]:
    same_basis = measured.bob_bases == prepared.bases
    errors = measured.bob_bits != prepared.bits
    counts = {}
    for label in INTENSITY_LABELS:
        mask = prepared.intensities == label
        sifted = mask & measured.clicks & same_basis
        counts[label] = IntensityCounts(
            pulses=int(mask.sum()),
            clicks=int((mask & measured.clicks).sum()),
            matched_pulses=int((mask & same_basis).sum()),
            sifted=int(sifted.sum()),
            errors=int((sifted & errors).sum()),
        )
    return counts


def decoy_bounds(
    mu: float,
    nu: float,
    q_mu_high: float,
    q_nu_low: float,
    y0_high: float,
    y0_low: float,
    eq_nu_high: float,
) -> tuple[float, float]:
    """Vacuum + weak decoy bounds (Ma, Qi, Zhao, Lo, PRA 72, 012326, 2005).

    Returns (Y1 lower bound, e1 upper bound). Each input should be the
    worst-case value for the direction it enters the bound.
    """
    denom = mu * nu - nu**2
    if denom <= 0:
        return 0.0, 1.0

    y1_lower = (
        mu
        / denom
        * (
            q_nu_low * np.exp(nu)
            - q_mu_high * np.exp(mu) * (nu**2 / mu**2)
            - (mu**2 - nu**2) / mu**2 * y0_high
        )
    )
    y1_lower = max(0.0, y1_lower)

    if y1_lower <= 0:
        return 0.0, 1.0
    e1_upper = (eq_nu_high * np.exp(nu) - E0 * y0_low) / (nu * y1_lower)
    return y1_lower, min(max(e1_upper, 0.0), 0.5)


def estimate_decoy_parameters(
    prepared: PreparedState,
    measured: MeasurementResult,
    mu_signal: float,
    mu_decoy: float,
) -> DecoyEstimates:
    """Asymptotic estimates: observed frequencies are taken as expectations."""
    counts = count_by_intensity(prepared, measured)
    gain = {label: c.gain for label, c in counts.items()}
    qber = {label: c.qber for label, c in counts.items()}

    y0 = gain["vacuum"]
    y1_lower, e1_upper = decoy_bounds(
        mu_signal,
        mu_decoy,
        q_mu_high=gain["signal"],
        q_nu_low=gain["decoy"],
        y0_high=y0,
        y0_low=y0,
        eq_nu_high=counts["decoy"].error_gain,
    )

    return DecoyEstimates(
        gain=gain, qber=qber, y0_lower=y0, y1_lower=y1_lower, e1_upper=e1_upper
    )
