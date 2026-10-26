from __future__ import annotations

import numpy as np

from .channel import MeasurementResult
from .decoy_estimation import (
    DecoyEstimates,
    IntensityCounts,
    count_by_intensity,
    decoy_bounds,
)
from .state_preparation import PreparedState

# Number of one-sided bounds used in parameter estimation; eps is split
# between them by a union bound.
NUM_ESTIMATED_BOUNDS = 5


def chernoff_upper(count: int, eps: float) -> float:
    """Upper bound on the expected value of a sum of independent Bernoulli
    trials, given the observed count, holding except with probability eps.

    Inverts P[X <= (1 - d) m] <= exp(-d^2 m / 2).
    """
    beta = np.log(1.0 / eps)
    return count + beta + np.sqrt(beta**2 + 2.0 * beta * count)


def chernoff_lower(count: int, eps: float) -> float:
    """Lower bound counterpart of `chernoff_upper`.

    Inverts P[X >= (1 + d) m] <= exp(-d^2 m / (2 + d)).
    """
    beta = np.log(1.0 / eps)
    return max(0.0, count - (np.sqrt(beta**2 + 8.0 * beta * count) - beta) / 2.0)


def _rate_interval(count: int, trials: int, eps: float) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    low = chernoff_lower(count, eps) / trials
    high = min(1.0, chernoff_upper(count, eps) / trials)
    return low, high


def estimate_decoy_parameters_finite(
    prepared: PreparedState,
    measured: MeasurementResult,
    mu_signal: float,
    mu_decoy: float,
    eps: float,
) -> DecoyEstimates:
    """Finite-size decoy estimates with multiplicative Chernoff bounds.

    `gain` and `qber` hold the observed values (they set the error-correction
    leakage, which is what is actually disclosed); Y0, Y1 and e1 are bounded.
    """
    counts = count_by_intensity(prepared, measured)
    return finite_estimates_from_counts(counts, mu_signal, mu_decoy, eps)


def finite_estimates_from_counts(
    counts: dict[str, IntensityCounts], mu_signal: float, mu_decoy: float, eps: float
) -> DecoyEstimates:
    """Finite-size decoy bounds from per-intensity counts (observed, or expected
    counts when used as a deterministic model in `src.theory`)."""
    eps_each = eps / NUM_ESTIMATED_BOUNDS

    signal, decoy, vacuum = counts["signal"], counts["decoy"], counts["vacuum"]
    _, q_mu_high = _rate_interval(signal.clicks, signal.pulses, eps_each)
    q_nu_low, _ = _rate_interval(decoy.clicks, decoy.pulses, eps_each)
    y0_low, y0_high = _rate_interval(vacuum.clicks, vacuum.pulses, eps_each)
    _, eq_nu_high = _rate_interval(decoy.errors, decoy.matched_pulses, eps_each)

    y1_lower, e1_upper = decoy_bounds(
        mu_signal,
        mu_decoy,
        q_mu_high=q_mu_high,
        q_nu_low=q_nu_low,
        y0_high=y0_high,
        y0_low=y0_low,
        eq_nu_high=eq_nu_high,
    )

    gain = {label: c.gain for label, c in counts.items()}
    qber = {label: c.qber for label, c in counts.items()}

    return DecoyEstimates(
        gain=gain, qber=qber, y0_lower=y0_low, y1_lower=y1_lower, e1_upper=e1_upper
    )
