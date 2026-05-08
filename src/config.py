from dataclasses import dataclass


@dataclass
class QKDConfig:
    num_rounds: int = 10000
    basis_prob_z: float = 0.5
    bob_basis_prob_z: float = 0.5
    test_fraction: float = 0.5
    intensity_prob_signal: float = 0.8
    intensity_prob_decoy: float = 0.15
    intensity_prob_vacuum: float = 0.05
    mu_signal: float = 0.5
    mu_decoy: float = 0.1
    mu_vacuum: float = 0.0
    channel_loss_db: float = 3.0
    detector_efficiency: float = 0.2
    dark_count_prob: float = 1e-6
    misalignment_error: float = 0.01
    error_correction_efficiency: float = 1.16
    eps_parameter_estimation: float = 1e-6
    eps_privacy_amplification: float = 1e-9

    def validate(self) -> None:
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if not (0.0 < self.basis_prob_z < 1.0):
            raise ValueError("basis_prob_z must be in (0, 1)")
        if not (0.0 < self.bob_basis_prob_z < 1.0):
            raise ValueError("bob_basis_prob_z must be in (0, 1)")
        if not (0.0 <= self.test_fraction <= 1.0):
            raise ValueError("test_fraction must be in [0, 1]")
        p_sum = (
            self.intensity_prob_signal
            + self.intensity_prob_decoy
            + self.intensity_prob_vacuum
        )
        if abs(p_sum - 1.0) > 1e-9:
            raise ValueError("intensity probabilities must sum to 1")
        if self.mu_signal < 0 or self.mu_decoy < 0 or self.mu_vacuum < 0:
            raise ValueError("intensities must be non-negative")
        if self.channel_loss_db < 0:
            raise ValueError("channel_loss_db must be non-negative")
        if not (0.0 <= self.detector_efficiency <= 1.0):
            raise ValueError("detector_efficiency must be in [0, 1]")
        if not (0.0 <= self.dark_count_prob <= 1.0):
            raise ValueError("dark_count_prob must be in [0, 1]")
        if not (0.0 <= self.misalignment_error <= 0.5):
            raise ValueError("misalignment_error must be in [0, 0.5]")
        if self.error_correction_efficiency < 1.0:
            raise ValueError("error_correction_efficiency must be >= 1.0")
        if not (0.0 < self.eps_parameter_estimation < 1.0):
            raise ValueError("eps_parameter_estimation must be in (0, 1)")
        if not (0.0 < self.eps_privacy_amplification < 1.0):
            raise ValueError("eps_privacy_amplification must be in (0, 1)")
