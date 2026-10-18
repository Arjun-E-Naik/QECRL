# RL-MWPM: Adaptive Real-Time Surface Code Decoding

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![CI](https://github.com/Arjun-E-Naik/QECRL/actions/workflows/ci.yml/badge.svg)](https://github.com/Arjun-E-Naik/QECRL/actions/workflows/ci.yml)
[![Manuscript](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/manuscript.pdf)

Official implementation and experimental benchmark suite for **"Adaptive Reinforcement Learning-Guided Detector Error Model Reweighting for Real-Time Non-Stationary Surface Code Decoding"**.

---

## Overview

Real-time decoding of topological surface codes on superconducting quantum processors requires satisfying strict hardware deadlines: syndrome extraction rounds execute every $200\text{--}1000\ \text{ns}$, imposing a per-round decoding latency budget of $\le 1.0\ \mu\text{s}$. Standard minimum-weight perfect matching (MWPM) decoders (such as C++ PyMatching) execute in sub-microsecond timescales per round, but rely on a static Detector Error Model (DEM) calibrated at commissioning time ($t=0$).

Under realistic physical solid-state non-stationarities—including Two-Level System (TLS) defect switching, low-frequency thermal/flux drift, and correlated cosmic-ray phonon impact bursts—static DEM weights become miscalibrated, inducing up to a $5\times$ inflation in the logical error rate (LER).

Conversely:
- **Classical analytical correlation estimators** (e.g., Spitz et al.) require $10^5\text{--}10^6$ shots to converge. On low-latency telemetry windows ($W \le 250$ shots), finite binomial sampling noise ($\sigma \approx \sqrt{p_0/W}$) exceeds the physical baseline error rate, injecting severe weight distortions that degrade logical fidelity by up to $3.4\times$ relative to an unadapted static decoder.
- **Monolithic end-to-end neural decoders** face $O(2^N)$ parity state-space complexity, scale poorly beyond distance $d=3$, and violate superconducting latency budgets by orders of magnitude ($>100\ \mu\text{s}/\text{shot}$).

**RL-MWPM** resolves this trade-off via a decoupled two-timescale decoding architecture:
1. **Inner Loop (C++ PyMatching):** Executes per-shot Dijkstra/blossom matching on an active DEM graph in $0.11\text{--}0.35\ \mu\text{s}/\text{round}$, strictly below the $1.0\ \mu\text{s}/\text{round}$ superconducting hardware cycle budget.
2. **Outer Loop (Contextual Bandit Meta-Controller):** Asynchronously tracks syndrome telemetry statistics across tumbling windows of $W = 250$ shots, performing sparse non-negative deconvolution and updating graph edge weights via lock-free atomic pointer swaps in $<10\ \text{ns}$.

```
Syndrome Telemetry (W = 250 shots, tumbling window)
       │
       ▼
Non-Negative Syndrome Deconvolution (NNLS)
       │  Suppresses spatial sidelobes onto quiet qubits (residual alpha_q < 1e-3)
       ▼
Vector l_inf CUSUM-EMA Change-Point Filter
       │  Switches between fast jump tracking (tau_rec = 1) and variance smoothing
       ▼
Contextual Bandit Meta-Controller (LinUCB)
       │  Evaluates self-supervised proxy reward (explained syndrome variance)
       ▼
Continuous DEM Sensitivity Projection (Pi_cont)
       │  Incident-edge max-pooling with continuous log-odds gain modulation
       ▼
Double-Buffered C++ PyMatching Graph (Atomic swap in < 10 ns)
       │
       ▼
Real-Time Shot Decoding (0.11 - 0.35 us / round <= 1.0 us hardware budget)
```

---

## Key Algorithmic Components

1. **Sparse Non-Negative Syndrome Deconvolution (NNLS):** Formulates defect localization as a constrained linear inverse problem over precomputed spatial dictionary templates, eliminating spatial sidelobe leakage onto quiet neighboring qubits ($\alpha_q < 10^{-3}$) without false-alarm activations.
2. **Continuous Fractional Sensitivity Matrix ($\mathbf{\Pi}_{\text{cont}}$):** Derives the exact differential mapping between physical qubit error surges and DEM edge log-likelihood ratios, reducing single-defect weight prediction discrepancy from $2.616$ down to $<10^{-4}$ at calibration points with continuous mean absolute error $\text{MAE} = 0.021$ across $p \in [0.001, 0.050]$.
3. **Change-Point Vector $\ell_\infty$ CUSUM-EMA Filter:** Dynamically switches between fast tracking ($\alpha_{\text{fast}} = 0.80$) upon detecting sharp defect onsets and aggressive noise rejection ($\alpha_{\text{slow}} = 0.38$) during stationary plateaus, eliminating finite-sampling dropouts and achieving single-window recovery ($\tau_{\text{rec}} = 1$).
4. **Self-Supervised Label-Free Proxy Reward ($R_t$):** Measures total explained syndrome variance across all $W \times N_{\text{det}}$ detector measurements, providing dense non-zero gradients in every window without requiring offline labeled logical observables.

---

## Empirical Benchmark Results

### Table I: Master Multi-Seed Benchmark Across Distances $d \in \{3, 5, 7\}$
*Evaluated over 5 independent random seeds under non-stationary noise (Regime B: telegraph TLS switching between $p_{\text{low}} = 0.001$ and $p_{\text{high}} = 0.040$). Stride $W = 250$ telemetry shots; $100{,}000$ validation shots/seed for $d \le 5$, $1{,}000{,}000$ validation shots for $d=7$.*

| Distance | Decoder | Cumulative LER (Mean $\pm$ SD) | SEM | Adaptation Recovery ($\eta_{\text{adapt}}$) | Inner Decode ($\mu\text{s}/\text{shot}$) | Per-Round Latency ($\mu\text{s}/\text{round}$) | Outer Loop Algo Latency ($\mu\text{s}$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$d=3$** | Static-MWPM | $0.01265 \pm 0.00840$ | $\pm 0.00042$ | $0.0\%$ | $0.33\ \mu\text{s}$ | $0.110\ \mu\text{s}$ | — |
| **$d=3$** | recMWPM / IRMWPM | $0.01248 \pm 0.00830$ | $\pm 0.00041$ | $3.1\%$ | $0.65\ \mu\text{s}$ | $0.217\ \mu\text{s}$ | — |
| **$d=3$** | Spitz-Corr-MWPM | $0.03373 \pm 0.01720$ | $\pm 0.00086$ | $0.0\%$ (Degraded) | $0.63\ \mu\text{s}$ | $0.210\ \mu\text{s}$ | $745.0\ \mu\text{s}$ |
| **$d=3$** | EMA-Heuristic-MWPM | $0.01555 \pm 0.01037$ | $\pm 0.00052$ | $0.0\%$ (Degraded) | $0.35\ \mu\text{s}$ | $0.117\ \mu\text{s}$ | $220.0\ \mu\text{s}$ |
| **$d=3$** | Pure-RL-Shot (MLP) | $0.16190 \pm 0.04500$ | $\pm 0.00225$ | $0.0\%$ (Degraded) | $0.24\ \mu\text{s}$ | $0.080\ \mu\text{s}$ | — |
| **$d=3$** | **RL-MWPM (Proxy, Ours)** | **$0.01193 \pm 0.00792$** | **$\pm 0.00040$** | **$12.8\%$** | **$0.33\ \mu\text{s}$** | **$0.110\ \mu\text{s}$** | **$110.0\ \mu\text{s}$** |
| **$d=3$** | Oracle-MWPM (Upper Bound) | $0.00709 \pm 0.00451$ | $\pm 0.00023$ | $100.0\%$ | $0.32\ \mu\text{s}$ | $0.107\ \mu\text{s}$ | — |
| **$d=5$** | Static-MWPM | $0.00268 \pm 0.00119$ | $\pm 0.00008$ | $0.0\%$ | $1.12\ \mu\text{s}$ | $0.224\ \mu\text{s}$ | — |
| **$d=5$** | recMWPM / IRMWPM | $0.00262 \pm 0.00115$ | $\pm 0.00008$ | $3.6\%$ | $2.28\ \mu\text{s}$ | $0.456\ \mu\text{s}$ | — |
| **$d=5$** | Spitz-Corr-MWPM | $0.00512 \pm 0.00220$ | $\pm 0.00015$ | $0.0\%$ (Degraded) | $3.06\ \mu\text{s}$ | $0.612\ \mu\text{s}$ | $1250.0\ \mu\text{s}$ |
| **$d=5$** | EMA-Heuristic-MWPM | $0.00274 \pm 0.00130$ | $\pm 0.00009$ | $0.0\%$ (Degraded) | $1.79\ \mu\text{s}$ | $0.358\ \mu\text{s}$ | $480.0\ \mu\text{s}$ |
| **$d=5$** | Pure-RL-Shot (MLP) | *Failed / OOM* | — | $0.0\%$ | $>100\ \mu\text{s}$ | $>20.0\ \mu\text{s}$ | — |
| **$d=5$** | **RL-MWPM (Proxy, Ours)** | **$0.00108 \pm 0.00062$** | **$\pm 0.00004$** | **$95.4\%$** | **$1.15\ \mu\text{s}$** | **$0.230\ \mu\text{s}$** | **$290.0\ \mu\text{s}$** |
| **$d=5$** | Oracle-MWPM (Upper Bound) | $0.00101 \pm 0.00053$ | $\pm 0.00004$ | $100.0\%$ | $1.13\ \mu\text{s}$ | $0.226\ \mu\text{s}$ | — |
| **$d=7$** | Static-MWPM | $0.00038 \pm 0.00002$ | $\pm 0.00002$ | $0.0\%$ | $2.13\ \mu\text{s}$ | $0.304\ \mu\text{s}$ | — |
| **$d=7$** | recMWPM / IRMWPM | $0.00037 \pm 0.00002$ | $\pm 0.00002$ | $4.2\%$ | $4.88\ \mu\text{s}$ | $0.697\ \mu\text{s}$ | — |
| **$d=7$** | **RL-MWPM (Proxy, Ours)** | **$0.00022 \pm 0.00002$** | **$\pm 0.00002$** | **$96.3\%$** | **$2.45\ \mu\text{s}$** | **$0.350\ \mu\text{s}$** | **$740.0\ \mu\text{s}$** |
| **$d=7$** | Oracle-MWPM (Upper Bound) | $0.00022 \pm 0.00002$ | $\pm 0.00002$ | $100.0\%$ | $2.43\ \mu\text{s}$ | $0.347\ \mu\text{s}$ | — |

*Note on Latency Budgets:* One syndrome measurement shot at distance $d$ consists of $R = d$ stabilizer rounds. The superconducting cycle deadline of $\le 1.0\ \mu\text{s}$ applies per round. Across all distances, RL-MWPM per-round latencies ($0.110\ \mu\text{s}$ at $d=3$, $0.230\ \mu\text{s}$ at $d=5$, $0.350\ \mu\text{s}$ at $d=7$) remain strictly and comfortably inside this budget.

### Table II: Component Ablation & Regime C Correlated Crosstalk Bursts
*Multi-seed evaluation across 10 independent random seeds (seeds 42–51). Statistical significance confirmed via paired $t$-test ($t = 5.67, p = 0.0020 < 0.005$) and Wilcoxon signed-rank test ($W = 0.0, p = 0.0019 < 0.005$).*

| Config | Architecture Variant | $d=3$ LER (Regime B) | $d=5$ LER (Regime B) | $d=5\ \eta_{\text{adapt}}$ (Regime B) | $d=5\ \eta_{\text{adapt}}$ (Regime C Crosstalk) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Static-MWPM (No Telemetry) | $0.01265 \pm 0.00042$ | $0.00268 \pm 0.00008$ | $0.0\%$ | $0.0\%$ |
| 2 | CUSUM-EMA + Binary $\mathbf{\Pi}_{\text{bin}}$ (No NNLS) | $0.01235 \pm 0.00041$ | $0.00118 \pm 0.00005$ | $89.8\%$ | $71.2\%$ |
| 3 | NNLS + Binary $\mathbf{\Pi}_{\text{bin}}$ + CUSUM-EMA | $0.01202 \pm 0.00040$ | $0.00114 \pm 0.00004$ | $92.2\%$ | $78.4\%$ |
| 4 | NNLS + $\mathbf{\Pi}_{\text{cont}}$ + Analytical Prior (No RL) | $0.01198 \pm 0.00040$ | $0.00111 \pm 0.00004$ | $94.0\%$ | $82.6\%$ |
| 5 | **RL-MWPM (Proxy Reward, Ours)** | **$0.01193 \pm 0.00040$** | **$0.00108 \pm 0.00004$** | **$95.4\%$** | **$96.7\%$** |

Under spatially correlated $2\times 2$ crosstalk bursts and two-qubit gate depolarization (Regime C), RL residual modulation demonstrates a statistically significant **14.1% recovery gap** over pure analytical Bayesian priors ($96.7\%$ vs $82.6\%$, $p < 0.005$), confirming the necessity of the RL adaptive layer.

---

## Repository Structure

```
QECRL/
├── qecrl/                      # Core Python package
│   ├── agents/
│   │   └── linucb_bandit.py    # LinUCB bandit with Sherman-Morrison rank-1 updates
│   ├── decoders/
│   │   ├── graph_updater.py    # DynamicDEMUpdater (NNLS, Pi_cont, CUSUM-EMA)
│   │   ├── classical_adaptive.py # Spitz-Corr-MWPM & EMA-Heuristic baselines
│   │   └── mwpm_decoder.py     # C++ PyMatching wrapper
│   ├── envs/
│   │   ├── reweight_env.py     # Gymnasium environment for RL training
│   │   └── surface_code_env.py # Parameterized surface code environment
│   ├── features/
│   │   └── syndrome_features.py # 1st and 2nd order DEM edge feature extractor
│   ├── noise/
│   │   ├── noise_engine.py     # NonStationaryNoiseEngine (Thermal, TLS, Bursts)
│   │   └── noise_models.py     # Drift and hardware replay models
│   └── utils/
│       ├── stim_circuits.py    # Parameterized rotated surface code circuits
│       └── stim_utils.py       # Stim circuit inspection utilities
├── scripts/                    # Experiment reproduction scripts
│   ├── quick_start.py          # Sanity check for environment and dependencies
│   ├── static_vs_oracle.py     # Phase 1: Figure 1 generator (Static vs Oracle MWPM)
│   ├── classical_vs_spitz.py   # Phase 2: Spitz breakdown analysis under low latency
│   ├── train_rl_mwpm.py        # Meta-controller training (LinUCB)
│   ├── benchmark_suite.py      # Full 6-model benchmark suite (Figures 2 & 3)
│   ├── hardware_validation.py  # Phase 4: Experimental hardware drift replay (Figure 4)
│   ├── sweep_hyperparameters.py# Hyperparameter sensitivity sweeps (Figure 5)
│   ├── crosstalk_regime_c_ablation.py # Regime C correlated crosstalk analysis
│   └── generate_camera_ready_figures.py # Camera-ready figure pipeline
├── paper/                      # Publication assets
│   ├── manuscript.tex          # Complete APS revtex4-2 manuscript source
│   ├── manuscript.pdf          # Full 15-page compiled publication PDF
│   └── figures/                # Camera-ready publication figures (Figures 1-7)
├── tests/                      # Unit test suite
│   ├── test_noise_engine.py    # Noise models and circuit layout tests
│   └── test_components.py      # Updater, feature extractor, and bandit tests
├── results/                    # Harmonized empirical datasets (CSVs and JSON metrics)
├── pyproject.toml              # Standard PEP 517/621 packaging specification
├── requirements.txt            # Pinned production dependencies
├── requirements-dev.txt        # Development and testing dependencies
├── CITATION.cff                # Academic citation metadata
└── LICENSE                     # MIT License
```

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- C++ compiler (for PyMatching / Stim wheels)

### Setup
```bash
# Clone the repository
git clone https://github.com/Arjun-E-Naik/QECRL.git
cd QECRL

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install package in editable mode
pip install -e .
```

---

## Reproducing Experimental Results

### 1. Verify Environment & Sanity Check
Run the sanity check script to verify Stim circuit generation, PyMatching integration, and environment rollouts:
```bash
python scripts/quick_start.py
```

### 2. Run Unit Test Suite
Execute the test suite to verify circuit layouts, noise generators, feature extractors, and bandit updates:
```bash
python -m unittest discover tests
```

### 3. Figure 1: Static-MWPM vs. Oracle-MWPM Breakdown
Reproduces the fundamental failure of static commissioning weights under non-stationary noise:
```bash
python scripts/static_vs_oracle.py --windows 100 --shots 300 --regime tls --seed 123
```
*Outputs: `results/figure1_static_vs_oracle.png` and `results/figure1_metrics.json`.*

### 4. Spitz Breakdown Analysis (Figure 3 in Paper)
Demonstrates the breakdown of analytical covariance inversion under compact telemetry windows ($W \le 250$):
```bash
python scripts/classical_vs_spitz.py --distance 3 --windows 30 --seed 42
```
*Outputs: `results/spitz_breakdown_analysis.png` and `results/spitz_breakdown_metrics.json`.*

### 5. Multi-Model Benchmark Suite (Figures 2 & 4 in Paper)
Evaluates Static-MWPM, recMWPM, Spitz-Corr-MWPM, EMA-Heuristic, RL-MWPM, and Oracle-MWPM:
```bash
python scripts/benchmark_suite.py --windows 40 --shots 250 --regime all --seed 42
```
*Outputs: `results/benchmark_results.csv`, `results/figure2_pareto_frontier.png`, and `results/figure3_lambda_suppression.png`.*

### 6. Hardware Drift Validation Replay (Figure 7 in Paper)
Evaluates RL-MWPM on real-world transmon hardware drift trajectories:
```bash
python scripts/hardware_validation.py --distance 5 --windows 40 --shots 250
```
*Outputs: `results/figure4_hardware_validation.png` and `results/hardware_validation_summary.json`.*

### 7. Regime C Correlated Crosstalk Ablation
Validates the statistically significant $14.1\%$ recovery gap over pure analytical priors:
```bash
python scripts/crosstalk_regime_c_ablation.py
```
*Outputs: `results/regime_c_crosstalk_ablation.json`.*

---

## Manuscript

The full manuscript is formatted in APS `revtex4-2` style:
- **LaTeX Source:** [`paper/manuscript.tex`](paper/manuscript.tex)
- **Compiled PDF:** [`paper/manuscript.pdf`](paper/manuscript.pdf) (15 pages, self-contained bibliography, all 7 camera-ready figures embedded)

To compile the LaTeX source locally using TeXLive or MacTeX:
```bash
cd paper
pdflatex manuscript.tex
pdflatex manuscript.tex
```

---

## Citation

If you use this codebase or paper in your research, please cite:

```bibtex
@article{naik2026adaptive,
  title   = {Adaptive Reinforcement Learning-Guided Detector Error Model Reweighting for Real-Time Non-Stationary Surface Code Decoding},
  author  = {Naik, Arjun E.},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
