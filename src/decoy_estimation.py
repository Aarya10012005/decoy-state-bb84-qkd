from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .channel import MeasurementResult
from .state_preparation import PreparedState


@dataclass
class DecoyEstimates:
    gain: Dict[str, float]
    qber: Dict[str, float]
    y1_lower: float
    e1_upper: float


def _gain(clicks: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    return clicks[mask].mean()


def _qber(
    prepared: PreparedState,
    measured: MeasurementResult,
    mask: np.ndarray,
) -> float:
    same_basis = measured.bob_bases == prepared.bases
    sifted = mask & measured.clicks & same_basis
    if sifted.sum() == 0:
        return 0.0
    return (measured.bob_bits[sifted] != prepared.bits[sifted]).mean()


def estimate_decoy_parameters(
    prepared: PreparedState,
    measured: MeasurementResult,
    mu_signal: float,
    mu_decoy: float,
) -> DecoyEstimates:
    labels = ("signal", "decoy", "vacuum")
    gain = {}
    qber = {}

    for label in labels:
        mask = prepared.intensities == label
        gain[label] = _gain(measured.clicks, mask)
        qber[label] = _qber(prepared, measured, mask)

    q_mu = gain["signal"]
    q_nu = gain["decoy"]
    y0 = gain["vacuum"]

    mu = mu_signal
    nu = mu_decoy
    exp_mu = np.exp(mu)
    exp_nu = np.exp(nu)

    denom = mu * nu - nu**2
    if denom <= 0:
        y1_lower = 0.0
    else:
        y1_lower = (
            mu
            / denom
            * (
                q_nu * exp_nu
                - q_mu * exp_mu * (nu**2 / mu**2)
                - (mu**2 - nu**2) / mu**2 * y0
            )
        )
        y1_lower = max(0.0, y1_lower)

    e0 = 0.5
    if y1_lower <= 0:
        e1_upper = 1.0
    else:
        e1_upper = (qber["decoy"] * q_nu * exp_nu - e0 * y0) / (nu * y1_lower)
        e1_upper = min(max(e1_upper, 0.0), 1.0)

    return DecoyEstimates(gain=gain, qber=qber, y1_lower=y1_lower, e1_upper=e1_upper)
