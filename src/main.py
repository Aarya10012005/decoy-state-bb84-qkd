from __future__ import annotations

from .channel import simulate_channel_and_detection
from .config import QKDConfig
from .decoy_estimation import estimate_decoy_parameters
from .finite_key import estimate_decoy_parameters_finite
from .key_rate import estimate_finite_key_rate, estimate_key_rate
from .sifting import sift_and_split
from .state_preparation import prepare_states


def main() -> None:
    config = QKDConfig(num_rounds=20000)
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
    key_rate = estimate_key_rate(
        estimates,
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
    )
    finite_key_rate = estimate_finite_key_rate(
        estimates_finite,
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
        eps_pa=config.eps_privacy_amplification,
        num_pulses=config.num_rounds,
    )
    sifting = sift_and_split(prepared, measured, config.test_fraction, seed=13)

    print("Prepared rounds:", prepared.bits.size)
    print("Basis Z fraction:", prepared.bases.mean())
    for label in ("signal", "decoy", "vacuum"):
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

    print("Gain (signal):", estimates.gain["signal"])
    print("Gain (decoy):", estimates.gain["decoy"])
    print("Gain (vacuum):", estimates.gain["vacuum"])
    print("QBER (signal):", estimates.qber["signal"])
    print("QBER (decoy):", estimates.qber["decoy"])
    print("Y1 lower bound:", estimates.y1_lower)
    print("e1 upper bound:", estimates.e1_upper)
    print("Y1 lower bound (finite):", estimates_finite.y1_lower)
    print("e1 upper bound (finite):", estimates_finite.e1_upper)
    print("Test fraction (sifted):", sifting.test_mask.sum() / max(1, sifting.sifted_mask.sum()))
    print("Data fraction (sifted):", sifting.data_mask.sum() / max(1, sifting.sifted_mask.sum()))
    print("Asymptotic key rate (per pulse):", key_rate.key_rate)
    print("Finite-key rate (per pulse):", finite_key_rate.key_rate)
    print("Finite-key length (bits):", finite_key_rate.key_length)
    print("Privacy amplification penalty (bits):", finite_key_rate.pa_penalty_bits)


if __name__ == "__main__":
    main()
