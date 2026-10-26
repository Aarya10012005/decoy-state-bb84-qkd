from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import QKDConfig

INTENSITY_LABELS = ("signal", "decoy", "vacuum")


@dataclass
class PreparedState:
    bits: np.ndarray
    bases: np.ndarray
    intensities: np.ndarray
    photon_numbers: np.ndarray


def _choose_bases(num_rounds: int, basis_prob_z: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(num_rounds) < basis_prob_z


def _choose_intensities(
    num_rounds: int,
    intensity_prob_signal: float,
    intensity_prob_decoy: float,
    intensity_prob_vacuum: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probs = [intensity_prob_signal, intensity_prob_decoy, intensity_prob_vacuum]
    return rng.choice(np.array(INTENSITY_LABELS), size=num_rounds, p=probs)


def _sample_photon_numbers(
    intensities: np.ndarray, mu_map: dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    mu_values = np.zeros(intensities.size, dtype=float)
    for label, mu in mu_map.items():
        mu_values[intensities == label] = mu
    return rng.poisson(mu_values)


def prepare_states(config: QKDConfig, seed: int | None = None) -> PreparedState:
    config.validate()
    rng = np.random.default_rng(seed)

    bits = rng.integers(0, 2, size=config.num_rounds, dtype=np.int8)
    bases = _choose_bases(config.num_rounds, config.basis_prob_z, rng)
    intensities = _choose_intensities(
        config.num_rounds,
        config.intensity_prob_signal,
        config.intensity_prob_decoy,
        config.intensity_prob_vacuum,
        rng,
    )
    photon_numbers = _sample_photon_numbers(intensities, config.mu_map(), rng)

    return PreparedState(
        bits=bits,
        bases=bases,
        intensities=intensities,
        photon_numbers=photon_numbers,
    )
