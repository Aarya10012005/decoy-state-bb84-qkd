"""BB84 on qubits with Qiskit.

Each single-photon round is a one-qubit circuit:

    Alice: X (bit 1), H (X basis)  ->  [Eve: measure in her basis, which also
    re-prepares the state she saw]  ->  channel: RY(2θ) misalignment with
    sin²θ = e_d, plus an optional depolarising channel  ->  Bob: H (X basis), measure.

Rounds sharing (Alice basis, bit, Eve basis, Bob basis) run as one circuit, so
N rounds need at most 24 circuits. Loss and dark counts are not qubit errors and
stay in the numpy model (`src.channel`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error

from .attacks import InterceptResend
from .channel import simulate_channel_and_detection
from .config import QKDConfig
from .state_preparation import prepare_states

Z, X = True, False  # basis encoding used throughout the project


@dataclass
class QiskitBB84Result:
    rounds: int
    sifted: int
    errors: int

    @property
    def qber(self) -> float:
        return self.errors / self.sifted if self.sifted else 0.0


def bb84_circuit(
    alice_basis: bool,
    alice_bit: int,
    bob_basis: bool,
    misalignment_error: float,
    eve_basis: bool | None = None,
    depolarizing: float = 0.0,
) -> QuantumCircuit:
    qr = QuantumRegister(1, "photon")
    bob = ClassicalRegister(1, "bob")
    regs = [qr, bob]
    if eve_basis is not None:
        eve = ClassicalRegister(1, "eve")
        regs.append(eve)
    qc = QuantumCircuit(*regs)

    if alice_bit:
        qc.x(0)
    if alice_basis == X:
        qc.h(0)
    qc.barrier(label="Alice")

    if eve_basis is not None:
        # A projective measurement leaves the qubit in the state Eve observed,
        # which is exactly an intercept-resend in her basis.
        if eve_basis == X:
            qc.h(0)
        qc.measure(0, eve[0])
        if eve_basis == X:
            qc.h(0)
        qc.barrier(label="Eve")

    theta = np.arcsin(np.sqrt(misalignment_error))
    qc.ry(2.0 * theta, 0)
    if depolarizing > 0.0:
        # Appended as an instruction so the noise acts only here, not on
        # Alice's or Bob's basis gates.
        qc.append(depolarizing_error(depolarizing, 1).to_instruction(), [0])
    qc.barrier(label="channel")

    if bob_basis == X:
        qc.h(0)
    qc.measure(0, bob[0])
    return qc


def run_qiskit_bb84(
    config: QKDConfig,
    num_rounds: int = 20_000,
    eavesdropper: bool = False,
    depolarizing: float = 0.0,
    seed: int = 5,
) -> QiskitBB84Result:
    """Simulate `num_rounds` single-photon BB84 rounds on Aer and count the
    errors in rounds where Alice's and Bob's bases agree."""
    rng = np.random.default_rng(seed)
    alice_bases = rng.random(num_rounds) < config.basis_prob_z
    alice_bits = rng.integers(0, 2, size=num_rounds)
    bob_bases = rng.random(num_rounds) < config.bob_basis_prob_z
    eve_bases = rng.random(num_rounds) < 0.5 if eavesdropper else [None] * num_rounds

    buckets = Counter(
        (bool(a), int(b), bool(bb), None if e is None else bool(e))
        for a, b, bb, e in zip(alice_bases, alice_bits, bob_bases, eve_bases)
    )
    keys = list(buckets)
    circuits = [
        bb84_circuit(a, b, bb, config.misalignment_error, eve_basis=e, depolarizing=depolarizing)
        for a, b, bb, e in keys
    ]

    simulator = AerSimulator(seed_simulator=seed)
    compiled = transpile(circuits, simulator, seed_transpiler=seed)
    sifted = errors = 0
    for (alice_basis, alice_bit, bob_basis, _), circuit in zip(keys, compiled):
        shots = buckets[(alice_basis, alice_bit, bob_basis, _)]
        counts = simulator.run(circuit, shots=shots).result().get_counts()
        if alice_basis != bob_basis:
            continue
        for outcome, n in counts.items():
            bob_bit = int(outcome.split()[-1])  # bob is the first register: rightmost
            sifted += n
            errors += n * (bob_bit != alice_bit)

    return QiskitBB84Result(rounds=num_rounds, sifted=sifted, errors=errors)


def numpy_qber(config: QKDConfig, eavesdropper: bool, seed: int = 5) -> float:
    """Signal QBER from the numpy photon-level model, for cross-checking."""
    prepared = prepare_states(config, seed=seed)
    channel = InterceptResend(1.0) if eavesdropper else None
    measured = simulate_channel_and_detection(prepared, config, seed=seed + 1, channel=channel)
    sifted = measured.clicks & (measured.bob_bases == prepared.bases)
    return float((measured.bob_bits[sifted] != prepared.bits[sifted]).mean())


def main() -> None:
    config = QKDConfig()
    print("Example round: Alice sends bit 1 in the X basis, Eve measures in Z, Bob in X\n")
    print(bb84_circuit(X, 1, X, config.misalignment_error, eve_basis=Z).draw("text"))

    print(f"\n{'scenario':34}{'sifted':>8}{'qiskit QBER':>13}{'numpy QBER':>12}{'expected':>10}")
    e_d = config.misalignment_error
    scenarios = [
        ("no eavesdropper", False, 0.0, e_d),
        ("intercept-resend (all rounds)", True, 0.0, e_d / 2 + 0.25),
        ("no Eve + 2% depolarising", False, 0.02, e_d + 0.01 * (1 - 2 * e_d)),
    ]
    for name, eve, depol, expected in scenarios:
        result = run_qiskit_bb84(config, eavesdropper=eve, depolarizing=depol)
        reference = f"{numpy_qber(config, eve):>12.4f}" if depol == 0.0 else f"{'-':>12}"
        exp = f"{expected:>10.4f}" if expected is not None else f"{'-':>10}"
        print(f"{name:34}{result.sifted:>8}{result.qber:>13.4f}{reference}{exp}")


if __name__ == "__main__":
    main()
