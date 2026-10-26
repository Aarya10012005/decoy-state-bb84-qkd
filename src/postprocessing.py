"""Classical post-processing: turn sifted bits into an identical secret key.

    sifted data bits -> Cascade error correction -> hash verification
                     -> Toeplitz privacy amplification -> final key

The number of parity bits Cascade discloses is counted and fed into the
finite-key formula, replacing the assumed f_EC * h(E) leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import MeasurementResult, simulate_channel_and_detection
from .config import QKDConfig
from .finite_key import estimate_decoy_parameters_finite
from .key_rate import binary_entropy, estimate_finite_key_rate
from .sifting import sift_and_split
from .state_preparation import PreparedState, prepare_states


@dataclass
class CascadeResult:
    corrected: np.ndarray
    leaked_bits: int
    passes: int


def _parity(bits: np.ndarray, idx: np.ndarray) -> int:
    return int(bits[idx].sum() & 1)


class _Cascade:
    """Cascade (Brassard & Salvail, 1993). Alice answers parity queries; each
    answer is one bit disclosed to Eve."""

    def __init__(self, alice: np.ndarray, bob: np.ndarray, qber: float, passes: int, seed: int):
        self.alice = alice
        self.bob = bob.copy()
        self.leaked = 0
        n = alice.size
        rng = np.random.default_rng(seed)
        first_block = max(4, int(0.73 / max(qber, 1e-4)))

        # For each pass: the blocks (as index arrays) and, per bit, its block.
        self.blocks: list[list[np.ndarray]] = []
        self.block_of: list[np.ndarray] = []
        for p in range(passes):
            order = np.arange(n) if p == 0 else rng.permutation(n)
            size = min(n, first_block * 2**p)
            blocks = [order[i : i + size] for i in range(0, n, size)]
            block_of = np.empty(n, dtype=np.int64)
            for b, idx in enumerate(blocks):
                block_of[idx] = b
            self.blocks.append(blocks)
            self.block_of.append(block_of)
        self.alice_parity: list[dict[int, int]] = [{} for _ in range(passes)]

    def _alice_block_parity(self, p: int, b: int) -> int:
        if b not in self.alice_parity[p]:
            self.alice_parity[p][b] = _parity(self.alice, self.blocks[p][b])
            self.leaked += 1
        return self.alice_parity[p][b]

    def _binary_search(self, idx: np.ndarray) -> int:
        while idx.size > 1:
            half = idx[: idx.size // 2]
            self.leaked += 1
            if _parity(self.alice, half) != _parity(self.bob, half):
                idx = half
            else:
                idx = idx[idx.size // 2 :]
        return int(idx[0])

    def run(self) -> CascadeResult:
        for p in range(len(self.blocks)):
            pending = [(p, b) for b in range(len(self.blocks[p]))]
            while pending:
                q, b = pending.pop()
                idx = self.blocks[q][b]
                if self._alice_block_parity(q, b) == _parity(self.bob, idx):
                    continue
                flipped = self._binary_search(idx)
                self.bob[flipped] ^= 1
                # The flip changes the parity of the block holding this bit in
                # every pass run so far; recheck those (the cascade effect).
                pending.extend((r, int(self.block_of[r][flipped])) for r in range(p + 1))
        return CascadeResult(corrected=self.bob, leaked_bits=self.leaked, passes=len(self.blocks))


def cascade(
    alice: np.ndarray, bob: np.ndarray, qber: float, passes: int = 4, seed: int = 0
) -> CascadeResult:
    return _Cascade(alice, bob, qber, passes, seed).run()


def toeplitz_hash(bits: np.ndarray, out_len: int, seed_bits: np.ndarray) -> np.ndarray:
    """Multiply `bits` by the out_len x n binary Toeplitz matrix defined by
    `seed_bits` (length n + out_len - 1), modulo 2. Toeplitz matrices form a
    two-universal hash family, as the leftover hash lemma requires."""
    n = bits.size
    if out_len <= 0:
        return np.zeros(0, dtype=np.int8)
    if seed_bits.size != n + out_len - 1:
        raise ValueError("seed_bits must have length n + out_len - 1")
    conv = np.convolve(seed_bits.astype(np.int64), bits.astype(np.int64))
    return (conv[n - 1 : n - 1 + out_len] & 1).astype(np.int8)


@dataclass
class PostProcessingResult:
    sifted_bits: int
    qber_estimate: float
    errors_before: int
    errors_after: int
    leaked_bits: int
    f_ec: float
    verified: bool
    key_length: int
    alice_key: np.ndarray
    bob_key: np.ndarray


def run_postprocessing(
    config: QKDConfig,
    prepared: PreparedState,
    measured: MeasurementResult,
    seed: int = 0,
) -> PostProcessingResult:
    rng = np.random.default_rng(seed)
    sifting = sift_and_split(prepared, measured, config.test_fraction, seed=seed)
    signal = prepared.intensities == "signal"

    # QBER is estimated on the disclosed test bits; key comes from data bits.
    test = sifting.test_mask & signal
    qber = float((measured.bob_bits[test] != prepared.bits[test]).mean()) if test.any() else 0.5
    data = sifting.data_mask & signal
    alice = prepared.bits[data].astype(np.int8)
    bob = measured.bob_bits[data].astype(np.int8)

    ec = cascade(alice, bob, qber, seed=seed)
    n = alice.size
    errors_before = int((alice != bob).sum())
    # Efficiency relative to the Shannon limit n * h(E) for the key's true error rate.
    true_qber = errors_before / n if n else 0.0
    f_ec = ec.leaked_bits / (n * binary_entropy(true_qber)) if errors_before else float("nan")

    # Error verification: compare short Toeplitz hashes of both keys.
    tag_len = int(np.ceil(np.log2(1.0 / config.eps_correctness)))
    tag_seed = rng.integers(0, 2, size=n + tag_len - 1)
    verified = bool(
        np.array_equal(
            toeplitz_hash(alice, tag_len, tag_seed), toeplitz_hash(ec.corrected, tag_len, tag_seed)
        )
    )

    estimates = estimate_decoy_parameters_finite(
        prepared, measured, config.mu_signal, config.mu_decoy, config.eps_parameter_estimation
    )
    key_length = estimate_finite_key_rate(
        estimates,
        mu_signal=config.mu_signal,
        f_ec=config.error_correction_efficiency,
        sifting_factor=config.basis_match_prob(),
        test_fraction=config.test_fraction,
        eps_pa=config.eps_privacy_amplification,
        eps_cor=config.eps_correctness,
        num_signal_pulses=int(signal.sum()),
        num_pulses=config.num_rounds,
        leak_ec_bits=ec.leaked_bits,
    ).key_length
    key_length = int(min(key_length, n)) if verified else 0

    pa_seed = rng.integers(0, 2, size=n + key_length - 1) if key_length else np.zeros(0)
    return PostProcessingResult(
        sifted_bits=n,
        qber_estimate=qber,
        errors_before=errors_before,
        errors_after=int((alice != ec.corrected).sum()),
        leaked_bits=ec.leaked_bits,
        f_ec=f_ec,
        verified=verified,
        key_length=key_length,
        alice_key=toeplitz_hash(alice, key_length, pa_seed),
        bob_key=toeplitz_hash(ec.corrected, key_length, pa_seed),
    )


def main() -> None:
    config = QKDConfig()
    prepared = prepare_states(config, seed=7)
    measured = simulate_channel_and_detection(prepared, config, seed=11)
    r = run_postprocessing(config, prepared, measured)

    print(f"N = {config.num_rounds:.0e} pulses, {config.channel_loss_db} dB loss\n")
    print(f"Sifted signal data bits:       {r.sifted_bits}")
    print(f"QBER (from test bits):         {r.qber_estimate:.4f}")
    print(f"Bit errors before Cascade:     {r.errors_before}")
    print(f"Bit errors after Cascade:      {r.errors_after}")
    print(f"Parity bits disclosed:         {r.leaked_bits}")
    print(f"Measured f_EC:                 {r.f_ec:.3f}  (assumed {config.error_correction_efficiency})")
    print(f"Hash verification passed:      {r.verified}")
    print(f"Final key length (bits):       {r.key_length}")
    print(f"Alice key == Bob key:          {np.array_equal(r.alice_key, r.bob_key)}")
    if r.key_length:
        print(f"First 64 key bits:             {''.join(map(str, r.alice_key[:64]))}")


if __name__ == "__main__":
    main()
