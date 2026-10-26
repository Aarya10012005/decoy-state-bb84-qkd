# Decoy-State BB84 QKD Simulation

A practical decoy-state BB84 quantum key distribution simulation with channel effects, detector modeling, decoy-state parameter estimation, finite-key bounds, key-rate estimation, eavesdropping attacks, source-parameter optimisation, a Qiskit qubit-level implementation, and a full classical post-processing pipeline.

## Model
- **Source**: weak coherent pulses with Poisson photon number, three intensities
  (signal μ, weak decoy ν, vacuum).
- **Channel and detector**: transmittance η = 10^(−loss/10) · η_det, dark-count
  probability p_d (dark-count-only clicks give a random bit), and misalignment error e_d.
- **Decoy estimation**: vacuum + weak decoy bounds on the single-photon yield Y₁ and
  error rate e₁ (Ma, Qi, Zhao, Lo, *PRA* 72, 012326, 2005).
- **Key rate (asymptotic)**: GLLP / Lo–Ma–Chen,
  `R = q · [Q₀ + Q₁(1 − h(e₁)) − Q_μ · f · h(E_μ)]` per signal pulse, where q is the
  basis-match (sifting) probability.
- **Finite key**: multiplicative Chernoff bounds on the observed counts, with
  ε_PE split by a union bound. Only non-test signal pulses contribute key, and the
  privacy amplification (2·log₂(1/ε_PA)) and error verification (log₂(2/ε_cor))
  costs are subtracted.
- **Theory**: `src/theory.py` gives closed-form expected gains, QBERs and bounds for
  the same model, which the simulation is checked against.

All parameters live in `src/config.py` (`QKDConfig`).

## Requirements
- Python 3.10+
- numpy, scipy, matplotlib
- qiskit and qiskit-aer (only for the qubit-level simulation)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the simulation
From the project directory:
```bash
python -m src.main
```
This prints simulated vs. finite-key vs. theoretical gains, QBERs, Y₁/e₁ bounds and key rates.

## Features

### Eavesdropping attacks: `python -m src.attacks`
Simulates two attacks on the quantum channel (`src/attacks.py`):
- **Intercept-resend**: Eve measures in a random basis and resends. The QBER rises by
  about 25% × (fraction intercepted), so the attack is detected.
- **Photon-number splitting (PNS)**: Eve uses a lossless line, blocks single photons and
  keeps one photon from every multi-photon pulse. She tunes this so the signal gain looks
  exactly like normal fibre loss, and she introduces no errors.

The driver compares the secret fraction claimed by a **naive** analysis (which assumes a
single-photon source) with the **decoy-state** analysis and with what Eve really knows.
Under PNS at high loss, the naive analysis claims about 88% of the key is secret while Eve
knows over 99% of it. The decoy-state analysis notices that the decoy gain is wrong and
correctly gives zero key. This is the reason decoy states exist.

### Source-parameter optimisation: `python -m src.optimization`
Uses `scipy.optimize.differential_evolution` to choose μ, ν and the intensity
probabilities that maximise the expected finite-key rate (`src/optimization.py`). The
objective uses the noise-free expected-count model from `src/theory.py`. For example, at
6 dB with N = 10⁶ the default parameters give no key at all, while the optimised ones give
a positive key, which the Monte Carlo simulation confirms.

### Qubit-level BB84 with Qiskit: `python -m src.qiskit_bb84`
Each single-photon round is a one-qubit circuit (`src/qiskit_bb84.py`):
- Alice prepares the qubit with X and H gates.
- Eve (optional) does a mid-circuit measurement.
- The channel applies an RY rotation (misalignment), optionally with depolarising noise.
- Bob measures in Z or X.

Runs on the Aer simulator and cross-checks the QBER against the numpy model
(≈ 1% honest, ≈ 25.5% with intercept-resend).

### Classical post-processing: `python -m src.postprocessing`
Turns sifted bits into an actual shared secret key (`src/postprocessing.py`):
1. **Cascade** error correction (4 passes). Counts every disclosed parity bit and reports
   the measured efficiency f_EC.
2. **Hash verification** of Alice's and Bob's corrected keys.
3. **Toeplitz-hashing privacy amplification**, compressing to the finite-key length
   computed with the *measured* leakage.

The output shows that Alice's and Bob's final keys are identical.

## Generate plots
```bash
python -m src.plotting
```
This writes the following to `plots/`:
- `gains_qber.png`: simulated vs. theoretical gains and QBERs
- `key_rate_vs_loss.png`: theory, simulated asymptotic and finite-key rates
- `attacks.png`: QBER under intercept-resend; claimed vs. actual secrecy under PNS
- `optimization.png`: optimised vs. default key rate, and optimal parameters vs. distance

## Run tests
```bash
python -m unittest discover -s tests -t .
```

## Project structure
- `src/` : simulation modules and drivers
- `tests/` : unit tests
- `plots/` : generated figures
