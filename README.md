# pMEM-EEGMEG

Generative pairwise maximum entropy model (pMEM) for continuous neuroimaging data.

## Files

| File | Description |
|------|-------------|
| `pmem_obs.py` | Core EM model for direct (source-space / ROI) observations |
| `pmem_leadfield.py` | EEG/MEG sensor-space extension with leadfield forward model |
| `__init__.py` | Package entry point — exports the public API |
| `paths.py` | Configurable output and data directory paths |
| `example.py` | Worked examples for both model classes |
| `requirements.txt` | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from pmem_obs import PMEMObsSimple
from pmem_leadfield import pMEM_Leadfield_v2
```

See `example.py` for full worked examples with simulated data.

---

## `pmem_obs.py` — PMEMObsSimple

Generative model:

```
z(t) ~ pMEM(h, J)
y(t) = β₀ + β₁ ⊙ z(t) + ε(t),   ε(t) ~ N(0, P⁻¹)
P = λ₁I + λ₂11ᵀ   (compound-symmetry precision)
```

Parameters jointly estimated: `h`, `J`, `β₀`, `β₁`, `σ²`, `ρ`.

```python
model = PMEMObsSimple(
    Y,                        # [T, N]  continuous observations
    n_iter=200,               # EM iterations
    lr_h=0.2,                 # learning rate for h
    lr_J=0.2,                 # learning rate for J
    estimate_noise=True,      # estimate σ² and ρ from data
    seed=0,
)
h, J, beta0, beta1, m = model.fit(verbose=True)
# model.sigma2, model.rho  — estimated noise parameters
```

| Output | Shape | Description |
|--------|-------|-------------|
| `h` | `[N]` | pMEM external fields |
| `J` | `[N, N]` | pMEM pairwise couplings |
| `beta0` | `[N]` | per-node baseline |
| `beta1` | `[N]` | per-node activation gain |
| `m` | `[T, N]` | posterior mean activation probability |

**Helper functions**

| Function | Returns | Description |
|----------|---------|-------------|
| `simulate_pmem_basic(T, h, J, beta0, beta1, sigma2, rho, seed)` | `(Y, Z)` | Generate synthetic observations |
| `fit_pmem_pseudolikelihood(Z, n_iter, lr)` | `(h, J)` | Demean → threshold → pseudolikelihood baseline |
| `fit_gmm_threshold_pmem(Y, n_iter, lr, seed)` | `(h, J, Z)` | GMM adaptive-threshold ablation |
| `exact_prior_moments(h, J)` | `(m0, m0_outer)` | Exact pMEM marginals via full enumeration |
| `rel_err(true, est)` | `float` | Relative RMSE |

---

## `pmem_leadfield.py` — pMEM_Leadfield_v2

Sensor-space extension. The leadfield `L ∈ ℝ^{M×N}` enters the likelihood directly — no source inversion is performed.

```
z(t) ~ pMEM(h, J)
y(t) = L(β₀ + β₁ ⊙ z(t)) + ε(t),   ε(t) ~ N(0, Σ_ε)
Σ_ε = σ²[(1−ρ)I + ρ11ᵀ]
```

```python
lf_model = pMEM_Leadfield_v2(
    Y,                        # [T, M]  sensor observations
    L,                        # [M, N]  leadfield matrix
    n_iter=80,
    estimate_noise=True,
    seed=0,
)
h, J, beta0, beta1, m = lf_model.fit(verbose=True)
# lf_model.sigma2, lf_model.rho  — estimated sensor noise parameters
```

---

## `paths.py` — path configuration

The full repository uses this folder layout:

```
mempy/
├── pmem_obs.py          # core model
├── pmem_leadfield.py    # leadfield model
├── __init__.py
├── example.py
├── scripts/             # figure generation and analysis scripts
│   ├── paths.py
│   ├── nature_style.py
│   ├── generate_fig1_fig2.py
│   ├── simulation_*.py
│   ├── physionet_analysis.py
│   ├── sedation_*.py
│   └── mne_leadfield_*.py
├── data/                # leadfield matrices and atlas files (input)
│   ├── leadfield_dmn6_64ch.npy
│   ├── leadfield_dmn6_egi91ch.npy
│   ├── leadfield_dmn6_mag102ch.npy
│   ├── leadfield_dmn6_physionet64ch.npy
│   └── dmn6_geodesic_centroids.npz
└── results/             # all outputs (auto-created on first run)
    ├── figures/         # output PNG figures
    └── cache/           # cached simulation results (.npy / .npz)
```

All paths are controlled by environment variables — no code edits needed:

```bash
export PMEM_FIGDIR=/path/to/figures        # default: results/figures
export PMEM_CACHEDIR=/path/to/cache        # default: results/cache
export PMEM_DATADIR=/path/to/data          # default: data/
export MNE_DATA=/path/to/mne_data          # default: ~/mne_data
export PMEM_SEDATION_DATA=/path/to/chennu  # default: data/Sedation-RestingState
```

---

## Citation

```bibtex
@article{jeong2026mempy,
  title   = {A generative framework for estimating brain energy landscapes
             from continuous neuroimaging observations},
  author  = {Jeong, Seok-Oh and Kim, Euisun and Kang, Jiyoung and
             Eo, Jinseok and Lee, Dongmyeong and Park, Hae-Jeong},
  year    = {2026},
  note    = {Under review}
}
```
