from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoy_estimation import DecoyEstimates


@dataclass
class KeyRateResult:
    key_rate: float
    raw_rate: float
    leak_ec: float


@dataclass
class FiniteKeyRateResult:
    key_rate: float
    key_length: float
    pa_penalty_bits: float


def _binary_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def estimate_key_rate(
    estimates: DecoyEstimates,
    mu_signal: float,
    f_ec: float,
    qber_signal: float | None = None,
) -> KeyRateResult:
    q_mu = estimates.gain["signal"]
    e_mu = estimates.qber["signal"] if qber_signal is None else qber_signal
    y1 = estimates.y1_lower
    e1 = estimates.e1_upper

    p1 = mu_signal * np.exp(-mu_signal)
    raw_rate = q_mu
    leak_ec = f_ec * _binary_entropy(e_mu)

    key_rate = p1 * y1 * (1.0 - _binary_entropy(e1)) - q_mu * leak_ec
    key_rate = max(0.0, key_rate)

    return KeyRateResult(key_rate=key_rate, raw_rate=raw_rate, leak_ec=leak_ec)


def estimate_finite_key_rate(
    estimates: DecoyEstimates,
    mu_signal: float,
    f_ec: float,
    eps_pa: float,
    num_pulses: int,
) -> FiniteKeyRateResult:
    key_rate = estimate_key_rate(
        estimates, mu_signal=mu_signal, f_ec=f_ec
    ).key_rate

    pa_penalty_bits = 2.0 * np.log2(1.0 / eps_pa)
    key_length = max(0.0, num_pulses * key_rate - pa_penalty_bits)
    finite_key_rate = key_length / max(1, num_pulses)

    return FiniteKeyRateResult(
        key_rate=finite_key_rate,
        key_length=key_length,
        pa_penalty_bits=pa_penalty_bits,
    )
