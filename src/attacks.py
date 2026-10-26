"""Eavesdropping attacks on the quantum channel.

Each attack implements the `Channel` protocol from `src.channel`, so it can be
passed to `simulate_channel_and_detection(..., channel=attack)`. Attacks also
mark the pulses whose bit Eve learns, so the secrecy claimed by an analysis can
be compared with what Eve actually knows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

from .channel import ArrivingState, channel_transmittance, simulate_channel_and_detection
from .config import QKDConfig
from .decoy_estimation import estimate_decoy_parameters
from .key_rate import binary_entropy, estimate_key_rate
from .state_preparation import PreparedState, prepare_states
from .theory import expected_gain


@dataclass
class InterceptResend:
    """Eve measures a `fraction` of non-empty pulses in a random basis and
    resends a pulse prepared in her result. Wrong-basis guesses cause errors."""

    fraction: float = 1.0

    def transmit(
        self, prepared: PreparedState, config: QKDConfig, rng: np.random.Generator
    ) -> ArrivingState:
        n = prepared.bits.size
        intercepted = (rng.random(n) < self.fraction) & (prepared.photon_numbers > 0)
        eve_bases = rng.random(n) < 0.5
        eve_right_basis = eve_bases == prepared.bases
        eve_bits = np.where(
            eve_right_basis, prepared.bits, rng.integers(0, 2, size=n, dtype=np.int8)
        )

        # Eve sits next to Alice, so the resent pulse still crosses the lossy fibre.
        return ArrivingState(
            photons=rng.binomial(prepared.photon_numbers, channel_transmittance(config)),
            bits=np.where(intercepted, eve_bits, prepared.bits).astype(np.int8),
            bases=np.where(intercepted, eve_bases, prepared.bases),
            eve_knows=intercepted & eve_right_basis,
        )


@dataclass
class PhotonNumberSplitting:
    """Eve replaces the fibre with a lossless line. She forwards single-photon
    pulses with probability `single_pass`; from multi-photon pulses she keeps one
    photon (learning the bit once bases are announced) and forwards the rest with
    probability `multi_pass`. No errors are introduced."""

    single_pass: float
    multi_pass: float = 1.0

    @classmethod
    def calibrated(cls, config: QKDConfig) -> "PhotonNumberSplitting":
        """Choose the pass probabilities so the signal gain matches the honest
        channel: plain BB84 cannot tell this attack apart from fibre loss."""
        mu, eta_d = config.mu_signal, config.detector_efficiency
        target = expected_gain(mu, config) - config.dark_count_prob

        n = np.arange(2, 60)
        multi_gain = float(np.sum(poisson.pmf(n, mu) * (1.0 - (1.0 - eta_d) ** (n - 1))))
        single_gain = float(poisson.pmf(1, mu) * eta_d)

        if multi_gain >= target:
            return cls(single_pass=0.0, multi_pass=target / multi_gain)
        return cls(single_pass=min(1.0, (target - multi_gain) / single_gain))

    def transmit(
        self, prepared: PreparedState, config: QKDConfig, rng: np.random.Generator
    ) -> ArrivingState:
        n = prepared.photon_numbers
        single = n == 1
        multi = n >= 2
        passes = np.where(single, rng.random(n.size) < self.single_pass, False)
        passes |= multi & (rng.random(n.size) < self.multi_pass)

        photons = np.where(single, 1, np.maximum(n - 1, 0)) * passes
        return ArrivingState(
            photons=photons,
            bits=prepared.bits,
            bases=prepared.bases,
            eve_knows=multi,
        )


def naive_key_rate(gain_signal: float, qber_signal: float, f_ec: float, q: float) -> float:
    """BB84 rate that assumes an ideal single-photon source (no decoy analysis):
    every detected signal pulse is treated as a secure single photon."""
    h = binary_entropy(qber_signal)
    return max(0.0, q * gain_signal * (1.0 - h - f_ec * h))


@dataclass
class AttackReport:
    qber_signal: float
    gain_signal: float
    gain_decoy: float
    y1_lower: float
    naive_secret_fraction: float
    decoy_secret_fraction: float
    eve_known_fraction: float


def analyze(config: QKDConfig, channel=None, seed: int = 7) -> AttackReport:
    """Run the protocol with `channel` and report, per sifted signal bit, the
    secret fraction each analysis claims versus the fraction Eve really knows."""
    prepared = prepare_states(config, seed=seed)
    measured = simulate_channel_and_detection(prepared, config, seed=seed + 4, channel=channel)
    estimates = estimate_decoy_parameters(prepared, measured, config.mu_signal, config.mu_decoy)
    q = config.basis_match_prob()

    gain, qber = estimates.gain["signal"], estimates.qber["signal"]
    decoy_rate = estimate_key_rate(
        estimates, config.mu_signal, config.error_correction_efficiency, q
    ).key_rate
    naive_rate = naive_key_rate(gain, qber, config.error_correction_efficiency, q)

    sifted_signal = (
        (prepared.intensities == "signal")
        & measured.clicks
        & (measured.bob_bases == prepared.bases)
    )
    sifted_rate = q * gain
    return AttackReport(
        qber_signal=qber,
        gain_signal=gain,
        gain_decoy=estimates.gain["decoy"],
        y1_lower=estimates.y1_lower,
        naive_secret_fraction=naive_rate / sifted_rate if sifted_rate else 0.0,
        decoy_secret_fraction=decoy_rate / sifted_rate if sifted_rate else 0.0,
        eve_known_fraction=float(measured.eve_knows[sifted_signal].mean())
        if sifted_signal.any()
        else 0.0,
    )


def plot_attacks(config: QKDConfig, out_dir: str) -> None:
    fractions = np.linspace(0.0, 1.0, 11)
    qbers = [analyze(config, InterceptResend(f)).qber_signal for f in fractions]

    losses = np.arange(0.0, 26.0, 2.0)
    naive, decoy, secret = [], [], []
    for loss_db in losses:
        sweep = replace(config, channel_loss_db=float(loss_db))
        report = analyze(sweep, PhotonNumberSplitting.calibrated(sweep))
        naive.append(report.naive_secret_fraction)
        decoy.append(report.decoy_secret_fraction)
        secret.append(1.0 - report.eve_known_fraction)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(fractions, qbers, "o-", color="#f03b20", label="simulated")
    axes[0].plot(
        fractions,
        config.misalignment_error + fractions * (0.25 - config.misalignment_error / 2),
        "--",
        color="grey",
        label=r"theory: $e_d + f\,(1/4 - e_d/2)$",
    )
    axes[0].axhline(0.11, color="black", lw=0.8, ls=":", label="BB84 abort threshold (11%)")
    axes[0].set_xlabel("Fraction of pulses intercepted")
    axes[0].set_ylabel("Signal QBER")
    axes[0].set_title("Intercept-resend attack")
    axes[0].legend()

    axes[1].fill_between(losses, secret, 1.0, color="#fcbba1", alpha=0.6, label="known to Eve")
    axes[1].plot(losses, secret, color="#cb181d", label="actually secret")
    axes[1].plot(losses, naive, "s--", color="#636363", label="claimed: no decoy")
    axes[1].plot(losses, decoy, "o-", color="#31a354", label="claimed: decoy state")
    axes[1].set_xlabel("Channel loss (dB)")
    axes[1].set_ylabel("Fraction of sifted signal bits")
    axes[1].set_title("Photon-number-splitting attack")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="center right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "attacks.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    config = QKDConfig()
    print(f"Channel loss: {config.channel_loss_db} dB, N = {config.num_rounds:.0e}\n")
    header = (
        f"{'scenario':28}{'QBER':>8}{'Q_mu':>10}{'Q_nu':>10}{'Y1 low':>9}"
        f"{'naive':>8}{'decoy':>8}{'Eve knows':>11}"
    )
    print(header)
    print("-" * len(header))

    scenarios = [
        ("no eavesdropper", config, None),
        ("intercept-resend 100%", config, InterceptResend(1.0)),
        ("intercept-resend 20%", config, InterceptResend(0.2)),
    ]
    for loss_db in (3.0, 15.0):
        sweep = replace(config, channel_loss_db=loss_db)
        if loss_db != config.channel_loss_db:
            scenarios.append((f"no eavesdropper @ {loss_db:g} dB", sweep, None))
        scenarios.append((f"PNS @ {loss_db:g} dB", sweep, PhotonNumberSplitting.calibrated(sweep)))

    for name, cfg, channel in scenarios:
        r = analyze(cfg, channel)
        print(
            f"{name:28}{r.qber_signal:>8.3f}{r.gain_signal:>10.2e}{r.gain_decoy:>10.2e}"
            f"{r.y1_lower:>9.3f}{r.naive_secret_fraction:>8.3f}{r.decoy_secret_fraction:>8.3f}"
            f"{r.eve_known_fraction:>11.3f}"
        )

    print(
        "\n'naive' / 'decoy' = secret fraction of sifted signal bits claimed by each analysis."
        "\nThe analysis is broken whenever it claims more than 1 - 'Eve knows'."
    )
    os.makedirs("plots", exist_ok=True)
    plot_attacks(config, "plots")
    print("Saved plots/attacks.png")


if __name__ == "__main__":
    main()
