from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import MeasurementResult
from .state_preparation import PreparedState


@dataclass
class SiftingResult:
    sifted_mask: np.ndarray
    test_mask: np.ndarray
    data_mask: np.ndarray


def sift_and_split(
    prepared: PreparedState,
    measured: MeasurementResult,
    test_fraction: float,
    seed: int | None = None,
) -> SiftingResult:
    rng = np.random.default_rng(seed)

    same_basis = prepared.bases == measured.bob_bases
    sifted_mask = measured.clicks & same_basis
    sifted_indices = np.flatnonzero(sifted_mask)

    if sifted_indices.size == 0:
        empty = np.zeros_like(sifted_mask, dtype=bool)
        return SiftingResult(sifted_mask=sifted_mask, test_mask=empty, data_mask=empty)

    rng.shuffle(sifted_indices)
    test_count = int(round(test_fraction * sifted_indices.size))
    test_indices = sifted_indices[:test_count]
    data_indices = sifted_indices[test_count:]

    test_mask = np.zeros_like(sifted_mask, dtype=bool)
    data_mask = np.zeros_like(sifted_mask, dtype=bool)
    test_mask[test_indices] = True
    data_mask[data_indices] = True

    return SiftingResult(sifted_mask=sifted_mask, test_mask=test_mask, data_mask=data_mask)
