from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .channel import simulate_channel_and_detection
from .config import QKDConfig
from .decoy_estimation import estimate_decoy_parameters
from .key_rate import estimate_key_rate
from .state_preparation import prepare_states


def plot_gains_and_qber(config: QKDConfig, out_dir: str) -> None:
    prepared = prepare_states(config, seed=7)
    measured = simulate_channel_and_detection(prepared, config, seed=11)
    estimates = estimate_decoy_parameters(
        prepared, measured, config.mu_signal, config.mu_decoy
    )

    labels = ["signal", "decoy", "vacuum"]
    gains = [estimates.gain[label] for label in labels]
    qbers = [estimates.qber[label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, gains, color="#2c7fb8")
    axes[0].set_title("Gain by intensity")
    axes[0].set_ylabel("Gain")

    axes[1].bar(labels, qbers, color="#f03b20")
    axes[1].set_title("QBER by intensity")
    axes[1].set_ylabel("QBER")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/gains_qber.png", dpi=160)
    plt.close(fig)


def plot_key_rate_vs_loss(config: QKDConfig, out_dir: str) -> None:
    losses = np.linspace(0.0, 20.0, 11)
    rates = []

    for loss_db in losses:
        sweep_config = QKDConfig(**{**config.__dict__, "channel_loss_db": float(loss_db)})
        prepared = prepare_states(sweep_config, seed=7)
        measured = simulate_channel_and_detection(prepared, sweep_config, seed=11)
        estimates = estimate_decoy_parameters(
            prepared, measured, sweep_config.mu_signal, sweep_config.mu_decoy
        )
        key_rate = estimate_key_rate(
            estimates,
            mu_signal=sweep_config.mu_signal,
            f_ec=sweep_config.error_correction_efficiency,
        )
        rates.append(key_rate.key_rate)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, rates, marker="o", color="#31a354")
    ax.set_title("Asymptotic key rate vs channel loss")
    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Key rate per pulse")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/key_rate_vs_loss.png", dpi=160)
    plt.close(fig)


def main() -> None:
    config = QKDConfig(num_rounds=20000)
    out_dir = "plots"
    plot_gains_and_qber(config, out_dir)
    plot_key_rate_vs_loss(config, out_dir)


if __name__ == "__main__":
    main()
