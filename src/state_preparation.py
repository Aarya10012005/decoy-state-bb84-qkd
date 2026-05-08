from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import QKDConfig


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
    return rng.choice(["signal", "decoy", "vacuum"], size=num_rounds, p=probs)


def _sample_photon_numbers(
    intensities: np.ndarray, mu_map: Dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    mu_values = np.vectorize(mu_map.get)(intensities)
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
    mu_map = {
        "signal": config.mu_signal,
        "decoy": config.mu_decoy,
        "vacuum": config.mu_vacuum,
    }
    photon_numbers = _sample_photon_numbers(intensities, mu_map, rng)

    return PreparedState(
        bits=bits,
        bases=bases,
        intensities=intensities,
        photon_numbers=photon_numbers,
    )
