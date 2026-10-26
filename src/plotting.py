from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from .attacks import plot_attacks
from .channel import simulate_channel_and_detection
from .config import QKDConfig
from .decoy_estimation import estimate_decoy_parameters
from .finite_key import estimate_decoy_parameters_finite
from .key_rate import estimate_finite_key_rate, estimate_key_rate
from .optimization import plot_optimization
from .state_preparation import INTENSITY_LABELS, prepare_states
from .theory import expected_estimates


def plot_gains_and_qber(config: QKDConfig, out_dir: str) -> None:
    prepared = prepare_states(config, seed=7)
    measured = simulate_channel_and_detection(prepared, config, seed=11)
    estimates = estimate_decoy_parameters(
        prepared, measured, config.mu_signal, config.mu_decoy
    )
    theory = expected_estimates(config)

    labels = list(INTENSITY_LABELS)
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x - width / 2, [estimates.gain[k] for k in labels], width, color="#2c7fb8", label="simulated")
    axes[0].bar(x + width / 2, [theory.gain[k] for k in labels], width, color="#a6bddb", label="theory")
    axes[0].set_yscale("log")
    axes[0].set_title("Gain by intensity")
    axes[0].set_ylabel("Gain")

    qber_labels = labels[:2]
    xq = np.arange(len(qber_labels))
    axes[1].bar(xq - width / 2, [estimates.qber[k] for k in qber_labels], width, color="#f03b20", label="simulated")
    axes[1].bar(xq + width / 2, [theory.qber[k] for k in qber_labels], width, color="#fdae6b", label="theory")
    axes[1].set_title("QBER by intensity")
    axes[1].set_ylabel("QBER")

    for ax, ticks in ((axes[0], labels), (axes[1], qber_labels)):
        ax.set_xticks(np.arange(len(ticks)), ticks)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gains_qber.png"), dpi=160)
    plt.close(fig)


def plot_key_rate_vs_loss(config: QKDConfig, out_dir: str) -> None:
    sim_losses = np.arange(0.0, 32.0, 2.0)
    theory_losses = np.linspace(0.0, 40.0, 201)
    q = config.basis_match_prob()

    theory_rates = [
        estimate_key_rate(
            expected_estimates(replace(config, channel_loss_db=float(loss_db))),
            mu_signal=config.mu_signal,
            f_ec=config.error_correction_efficiency,
            sifting_factor=q,
        ).key_rate
        for loss_db in theory_losses
    ]

    asymptotic_rates = []
    finite_rates = []
    for loss_db in sim_losses:
        sweep_config = replace(config, channel_loss_db=float(loss_db))
        prepared = prepare_states(sweep_config, seed=7)
        measured = simulate_channel_and_detection(prepared, sweep_config, seed=11)
        estimates = estimate_decoy_parameters(
            prepared, measured, sweep_config.mu_signal, sweep_config.mu_decoy
        )
        estimates_finite = estimate_decoy_parameters_finite(
            prepared,
            measured,
            sweep_config.mu_signal,
            sweep_config.mu_decoy,
            sweep_config.eps_parameter_estimation,
        )
        asymptotic_rates.append(
            estimate_key_rate(
                estimates,
                mu_signal=sweep_config.mu_signal,
                f_ec=sweep_config.error_correction_efficiency,
                sifting_factor=q,
            ).key_rate
        )
        finite_rates.append(
            estimate_finite_key_rate(
                estimates_finite,
                mu_signal=sweep_config.mu_signal,
                f_ec=sweep_config.error_correction_efficiency,
                sifting_factor=q,
                test_fraction=sweep_config.test_fraction,
                eps_pa=sweep_config.eps_privacy_amplification,
                eps_cor=sweep_config.eps_correctness,
                num_signal_pulses=int((prepared.intensities == "signal").sum()),
                num_pulses=sweep_config.num_rounds,
            ).key_rate
        )

    def positive(xs, ys):
        xs, ys = np.asarray(xs), np.asarray(ys)
        keep = ys > 0
        return xs[keep], ys[keep]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(*positive(theory_losses, theory_rates), color="#31a354", label="Asymptotic (theory)")
    ax.plot(*positive(sim_losses, asymptotic_rates), "o", color="#31a354", label="Asymptotic (simulated)")
    ax.plot(
        *positive(sim_losses, finite_rates),
        "s--",
        color="#756bb1",
        label=f"Finite key, N = {config.num_rounds:.0e}",
    )
    ax.set_yscale("log")
    ax.set_title("Secret key rate vs channel loss")
    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Key rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "key_rate_vs_loss.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    config = QKDConfig()
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    plot_gains_and_qber(config, out_dir)
    plot_key_rate_vs_loss(config, out_dir)
    plot_attacks(config, out_dir)
    plot_optimization(config, out_dir)


if __name__ == "__main__":
    main()
