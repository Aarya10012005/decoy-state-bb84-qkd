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


def binary_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def estimate_key_rate(
    estimates: DecoyEstimates,
    mu_signal: float,
    f_ec: float,
    sifting_factor: float,
) -> KeyRateResult:
    """GLLP / Lo-Ma-Chen secret key rate per signal pulse:

    R = q * [Q0 + Q1 (1 - h(e1)) - Q_mu f h(E_mu)]
    """
    q_mu = estimates.gain["signal"]
    e_mu = estimates.qber["signal"]

    q0 = estimates.y0_lower * np.exp(-mu_signal)
    q1 = estimates.y1_lower * mu_signal * np.exp(-mu_signal)
    leak_ec = f_ec * binary_entropy(e_mu)

    key_rate = sifting_factor * (
        q0 + q1 * (1.0 - binary_entropy(estimates.e1_upper)) - q_mu * leak_ec
    )
    key_rate = max(0.0, key_rate)

    return KeyRateResult(
        key_rate=key_rate, raw_rate=sifting_factor * q_mu, leak_ec=leak_ec
    )


def estimate_finite_key_rate(
    estimates: DecoyEstimates,
    mu_signal: float,
    f_ec: float,
    sifting_factor: float,
    test_fraction: float,
    eps_pa: float,
    eps_cor: float,
    num_signal_pulses: int,
    num_pulses: int,
    leak_ec_bits: float | None = None,
) -> FiniteKeyRateResult:
    """Finite key length from signal pulses whose sifted bits are not used for
    testing; `key_rate` is normalised to all `num_pulses` sent.

    If `leak_ec_bits` is given (bits actually disclosed by error correction),
    it replaces the modelled leakage f_ec * h(E_mu) per sifted bit.
    """
    rate_per_signal = estimate_key_rate(
        estimates,
        mu_signal=mu_signal,
        f_ec=f_ec if leak_ec_bits is None else 0.0,
        sifting_factor=sifting_factor,
    ).key_rate

    pa_penalty_bits = 2.0 * np.log2(1.0 / eps_pa) + np.log2(2.0 / eps_cor)
    usable_pulses = (1.0 - test_fraction) * num_signal_pulses
    key_length = usable_pulses * rate_per_signal - pa_penalty_bits
    if leak_ec_bits is not None:
        key_length -= leak_ec_bits
    key_length = max(0.0, key_length)
    finite_key_rate = key_length / max(1, num_pulses)

    return FiniteKeyRateResult(
        key_rate=finite_key_rate,
        key_length=key_length,
        pa_penalty_bits=pa_penalty_bits,
    )
