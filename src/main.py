from __future__ import annotations

from .channel import simulate_channel_and_detection
from .config import QKDConfig
from .decoy_estimation import estimate_decoy_parameters
from .finite_key import estimate_decoy_parameters_finite
from .key_rate import estimate_finite_key_rate, estimate_key_rate
from .sifting import sift_and_split
from .state_preparation import INTENSITY_LABELS, prepare_states
from .theory import expected_estimates, true_single_photon


def main() -> None:
    config = QKDConfig()
    prepared = prepare_states(config, seed=7)
    measured = simulate_channel_and_detection(prepared, config, seed=11)
    estimates = estimate_decoy_parameters(
        prepared, measured, config.mu_signal, config.mu_decoy
    )
    estimates_finite = estimate_decoy_parameters_finite(
        prepared,
        measured,
        config.mu_signal,
        config.mu_decoy,
        config.eps_parameter_estimation,
    )
    theory = expected_estimates(config)
    y1_true, e1_true = true_single_photon(config)

    q = config.basis_match_prob()
    key_rate = estimate_key_rate(
        estimates,
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
        sifting_factor=q,
    )
    theory_key_rate = estimate_key_rate(
        theory,
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
        sifting_factor=q,
    )
    finite_key_rate = estimate_finite_key_rate(
        estimates_finite,
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
        sifting_factor=q,
        test_fraction=config.test_fraction,
        eps_pa=config.eps_privacy_amplification,
        eps_cor=config.eps_correctness,
        num_signal_pulses=int((prepared.intensities == "signal").sum()),
        num_pulses=config.num_rounds,
    )
    sifting = sift_and_split(prepared, measured, config.test_fraction, seed=13)

    print("Prepared rounds:", prepared.bits.size)
    print("Overall transmittance:", config.transmittance())
    print("Basis Z fraction:", prepared.bases.mean())
    for label in INTENSITY_LABELS:
        frac = (prepared.intensities == label).mean()
        print(f"{label} fraction:", frac)
    print("Mean photon number:", prepared.photon_numbers.mean())
    print("Click rate:", measured.clicks.mean())
    same_basis = measured.bob_bases == prepared.bases
    sifted = measured.clicks & same_basis
    if sifted.any():
        qber = (measured.bob_bits[sifted] != prepared.bits[sifted]).mean()
        print("Sifted fraction:", sifted.mean())
        print("QBER (sifted):", qber)
    else:
        print("Sifted fraction:", 0.0)
        print("QBER (sifted): N/A")

    print()
    print(f"{'':24}{'simulated':>14}{'finite-key':>14}{'theory':>14}")
    for label in INTENSITY_LABELS:
        print(
            f"{'Gain (' + label + ')':24}{estimates.gain[label]:>14.4e}"
            f"{'':>14}{theory.gain[label]:>14.4e}"
        )
    for label in ("signal", "decoy"):
        print(
            f"{'QBER (' + label + ')':24}{estimates.qber[label]:>14.4e}"
            f"{'':>14}{theory.qber[label]:>14.4e}"
        )
    print(
        f"{'Y1 lower bound':24}{estimates.y1_lower:>14.4e}"
        f"{estimates_finite.y1_lower:>14.4e}{theory.y1_lower:>14.4e}"
    )
    print(
        f"{'e1 upper bound':24}{estimates.e1_upper:>14.4e}"
        f"{estimates_finite.e1_upper:>14.4e}{theory.e1_upper:>14.4e}"
    )
    print(f"True single-photon Y1, e1: {y1_true:.4e}, {e1_true:.4e}")
    print()

    print("Test fraction (sifted):", sifting.test_mask.sum() / max(1, sifting.sifted_mask.sum()))
    print("Data fraction (sifted):", sifting.data_mask.sum() / max(1, sifting.sifted_mask.sum()))
    print("Asymptotic key rate (per signal pulse):", key_rate.key_rate)
    print("Theoretical asymptotic key rate (per signal pulse):", theory_key_rate.key_rate)
    print("Finite-key rate (per pulse sent):", finite_key_rate.key_rate)
    print("Finite-key length (bits):", finite_key_rate.key_length)
    print("PA + EC verification penalty (bits):", finite_key_rate.pa_penalty_bits)


if __name__ == "__main__":
    main()
