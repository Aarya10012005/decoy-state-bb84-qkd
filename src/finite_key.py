from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoy_estimation import DecoyEstimates
from .state_preparation import PreparedState
from .channel import MeasurementResult


@dataclass
class FiniteKeyBounds:
    gain_low: dict[str, float]
    gain_high: dict[str, float]
    qber_low: dict[str, float]
    qber_high: dict[str, float]


def _hoeffding_interval(p_hat: float, n: int, eps: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    delta = np.sqrt(np.log(1.0 / eps) / (2.0 * n))
    return max(0.0, p_hat - delta), min(1.0, p_hat + delta)


def _estimate_bounds(
    prepared: PreparedState,
    measured: MeasurementResult,
    eps: float,
) -> FiniteKeyBounds:
    labels = ("signal", "decoy", "vacuum")
    gain_low: dict[str, float] = {}
    gain_high: dict[str, float] = {}
    qber_low: dict[str, float] = {}
    qber_high: dict[str, float] = {}

    same_basis = measured.bob_bases == prepared.bases

    for label in labels:
        mask = prepared.intensities == label
        n_total = int(mask.sum())
        clicks = int(measured.clicks[mask].sum())
        q_hat = clicks / n_total if n_total > 0 else 0.0
        gain_low[label], gain_high[label] = _hoeffding_interval(q_hat, n_total, eps)

        sifted = mask & measured.clicks & same_basis
        n_sifted = int(sifted.sum())
        if n_sifted > 0:
            errors = int((measured.bob_bits[sifted] != prepared.bits[sifted]).sum())
            e_hat = errors / n_sifted
        else:
            e_hat = 0.0
        qber_low[label], qber_high[label] = _hoeffding_interval(e_hat, n_sifted, eps)

    return FiniteKeyBounds(
        gain_low=gain_low,
        gain_high=gain_high,
        qber_low=qber_low,
        qber_high=qber_high,
    )


def estimate_decoy_parameters_finite(
    prepared: PreparedState,
    measured: MeasurementResult,
    mu_signal: float,
    mu_decoy: float,
    eps: float,
) -> DecoyEstimates:
    bounds = _estimate_bounds(prepared, measured, eps)

    mu = mu_signal
    nu = mu_decoy
    exp_mu = np.exp(mu)
    exp_nu = np.exp(nu)

    q_mu_high = bounds.gain_high["signal"]
    q_nu_low = bounds.gain_low["decoy"]
    q_nu_high = bounds.gain_high["decoy"]
    y0_high = bounds.gain_high["vacuum"]

    denom = mu * nu - nu**2
    if denom <= 0:
        y1_lower = 0.0
    else:
        y1_lower = (
            mu
            / denom
            * (
                q_nu_low * exp_nu
                - q_mu_high * exp_mu * (nu**2 / mu**2)
                - (mu**2 - nu**2) / mu**2 * y0_high
            )
        )
        y1_lower = max(0.0, y1_lower)

    e0 = 0.5
    e_nu_high = bounds.qber_high["decoy"]
    if y1_lower <= 0:
        e1_upper = 1.0
    else:
        e1_upper = (e_nu_high * q_nu_high * exp_nu - e0 * y0_high) / (nu * y1_lower)
        e1_upper = min(max(e1_upper, 0.0), 1.0)

    gain = {k: bounds.gain_low[k] for k in ("signal", "decoy", "vacuum")}
    qber = {k: bounds.qber_high[k] for k in ("signal", "decoy", "vacuum")}

    return DecoyEstimates(gain=gain, qber=qber, y1_lower=y1_lower, e1_upper=e1_upper)
