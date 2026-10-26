"""Optimisation of the decoy-state source parameters.

The objective uses the deterministic model in `src.theory` (expected counts
instead of Monte Carlo samples), so it is smooth and noise-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from .channel import simulate_channel_and_detection
from .config import QKDConfig
from .finite_key import estimate_decoy_parameters_finite
from .key_rate import estimate_finite_key_rate, estimate_key_rate
from .state_preparation import prepare_states
from .theory import expected_estimates, expected_finite_estimates

FIBRE_LOSS_DB_PER_KM = 0.2

# Search box for (mu_signal, mu_decoy, p_signal, p_decoy); p_vacuum is the rest.
BOUNDS = [(0.05, 1.2), (0.005, 0.5), (0.05, 0.98), (0.01, 0.9)]
MIN_VACUUM_PROB = 0.005


@dataclass
class OptimizationResult:
    config: QKDConfig
    key_rate: float


def _apply(config: QKDConfig, x: np.ndarray) -> QKDConfig | None:
    mu, nu, p_signal, p_decoy = (float(v) for v in x)
    p_vacuum = 1.0 - p_signal - p_decoy
    if nu >= mu or p_vacuum < MIN_VACUUM_PROB:
        return None
    return replace(
        config,
        mu_signal=mu,
        mu_decoy=nu,
        intensity_prob_signal=p_signal,
        intensity_prob_decoy=p_decoy,
        intensity_prob_vacuum=p_vacuum,
    )


def expected_key_rate(config: QKDConfig, finite: bool = True) -> float:
    """Expected key rate: per pulse sent (finite) or per signal pulse (asymptotic)."""
    q = config.basis_match_prob()
    if not finite:
        return estimate_key_rate(
            expected_estimates(config),
            mu_signal=config.mu_signal,
            f_ec=config.error_correction_efficiency,
            sifting_factor=q,
        ).key_rate
    return estimate_finite_key_rate(
        expected_finite_estimates(config),
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
        sifting_factor=q,
        test_fraction=config.test_fraction,
        eps_pa=config.eps_privacy_amplification,
        eps_cor=config.eps_correctness,
        num_signal_pulses=config.num_rounds * config.intensity_prob_signal,
        num_pulses=config.num_rounds,
    ).key_rate


def optimize_parameters(
    config: QKDConfig, finite: bool = True, seed: int = 0
) -> OptimizationResult:
    """Maximise the expected key rate over intensities and their probabilities.

    In the asymptotic limit the probabilities do not matter, so only (mu, nu)
    are effectively optimised.
    """

    def objective(x: np.ndarray) -> float:
        candidate = _apply(config, x)
        if candidate is None:
            return 1.0  # infeasible; any feasible point scores <= 0
        return -expected_key_rate(candidate, finite)

    result = differential_evolution(
        objective, BOUNDS, seed=seed, tol=1e-8, maxiter=300, polish=False
    )
    best = _apply(config, result.x)
    if best is None or -result.fun <= 0.0:
        return OptimizationResult(config=config, key_rate=0.0)
    return OptimizationResult(config=best, key_rate=-result.fun)


def plot_optimization(config: QKDConfig, out_dir: str) -> None:
    losses = np.arange(0.0, 42.0, 3.0)
    cases = [
        ("asymptotic", None, "#31a354"),
        ("N = 1e8", 10**8, "#2c7fb8"),
        ("N = 1e6", 10**6, "#756bb1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for name, num_rounds, color in cases:
        finite = num_rounds is not None
        base = config if not finite else replace(config, num_rounds=num_rounds)
        default_rates, optimal_rates, optimal = [], [], []
        for loss_db in losses:
            sweep = replace(base, channel_loss_db=float(loss_db))
            default_rates.append(expected_key_rate(sweep, finite))
            result = optimize_parameters(sweep, finite)
            optimal_rates.append(result.key_rate)
            optimal.append(result.config if result.key_rate > 0 else None)

        def positive(ys):
            ys = np.asarray(ys, dtype=float)
            return losses[ys > 0], ys[ys > 0]

        axes[0].plot(*positive(optimal_rates), "-", color=color, label=f"{name}, optimised")
        axes[0].plot(*positive(default_rates), "--", color=color, alpha=0.7, label=f"{name}, default")

        if name == "N = 1e8":
            ok = [i for i, c in enumerate(optimal) if c is not None]
            x = losses[ok]
            axes[1].plot(x, [optimal[i].mu_signal for i in ok], "o-", label=r"signal $\mu$")
            axes[1].plot(x, [optimal[i].mu_decoy for i in ok], "s-", label=r"decoy $\nu$")
            axes[2].plot(x, [optimal[i].intensity_prob_signal for i in ok], "o-", label="signal")
            axes[2].plot(x, [optimal[i].intensity_prob_decoy for i in ok], "s-", label="decoy")
            axes[2].plot(x, [optimal[i].intensity_prob_vacuum for i in ok], "^-", label="vacuum")

    axes[0].set_yscale("log")
    axes[0].set_title("Key rate: optimised vs default parameters")
    axes[0].set_ylabel("Key rate (asymptotic: per signal pulse)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Optimal intensities (N = 1e8)")
    axes[1].set_ylabel("Mean photon number")
    axes[1].legend()
    axes[2].set_title("Optimal intensity probabilities (N = 1e8)")
    axes[2].set_ylabel("Probability")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Channel loss (dB)")
        ax.grid(True, which="both", alpha=0.3)
        km = ax.secondary_xaxis(
            "top",
            functions=(lambda db: db / FIBRE_LOSS_DB_PER_KM, lambda k: k * FIBRE_LOSS_DB_PER_KM),
        )
        km.set_xlabel("Fibre length (km, 0.2 dB/km)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "optimization.png"), dpi=160)
    plt.close(fig)


def _simulated_finite_rate(config: QKDConfig) -> float:
    prepared = prepare_states(config, seed=7)
    measured = simulate_channel_and_detection(prepared, config, seed=11)
    estimates = estimate_decoy_parameters_finite(
        prepared, measured, config.mu_signal, config.mu_decoy, config.eps_parameter_estimation
    )
    return estimate_finite_key_rate(
        estimates,
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
        sifting_factor=config.basis_match_prob(),
        test_fraction=config.test_fraction,
        eps_pa=config.eps_privacy_amplification,
        eps_cor=config.eps_correctness,
        num_signal_pulses=int((prepared.intensities == "signal").sum()),
        num_pulses=config.num_rounds,
    ).key_rate


def main() -> None:
    config = QKDConfig(channel_loss_db=6.0)
    print(f"Finite-key optimisation at {config.channel_loss_db} dB, N = {config.num_rounds:.0e}\n")

    result = optimize_parameters(config)
    best = result.config
    rows = [
        ("mu_signal", config.mu_signal, best.mu_signal),
        ("mu_decoy", config.mu_decoy, best.mu_decoy),
        ("p_signal", config.intensity_prob_signal, best.intensity_prob_signal),
        ("p_decoy", config.intensity_prob_decoy, best.intensity_prob_decoy),
        ("p_vacuum", config.intensity_prob_vacuum, best.intensity_prob_vacuum),
        ("expected key rate", expected_key_rate(config), result.key_rate),
        ("simulated key rate", _simulated_finite_rate(config), _simulated_finite_rate(best)),
    ]
    print(f"{'':20}{'default':>12}{'optimised':>12}")
    for name, before, after in rows:
        print(f"{name:20}{before:>12.4g}{after:>12.4g}")

    os.makedirs("plots", exist_ok=True)
    plot_optimization(QKDConfig(), "plots")
    print("\nSaved plots/optimization.png")


if __name__ == "__main__":
    main()
