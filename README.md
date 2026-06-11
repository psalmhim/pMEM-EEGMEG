# mempy — pMEM observation model

Generative pairwise maximum entropy model (pMEM) for continuous neuroimaging data.
Two model classes are provided: one for direct (source-space or ROI) observations,
and one for EEG/MEG sensor-space data with an electromagnetic leadfield.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
from pmem_obs import PMEMObsSimple
from pmem_leadfield import pMEM_Leadfield_v2
```

Or, if the directory is on your Python path as a package:

```python
from mempy import PMEMObsSimple, pMEM_Leadfield_v2
```

See `example.py` for full worked examples with simulated data.

---

## PMEMObsSimple

Direct-observation EM estimator for source-space or ROI time series.

**Generative model**

```
z(t) ~ pMEM(h, J)
y(t) = β₀ + β₁ ⊙ z(t) + ε(t),   ε(t) ~ N(0, P⁻¹)
P = λ₁I + λ₂11ᵀ   (compound-symmetry precision)
```

Parameters jointly estimated: `h`, `J`, `β₀`, `β₁`, `σ²`, `ρ`.

**Constructor**

```python
PMEMObsSimple(
    Y,                        # [T, N]  continuous observations
    sigma2=None,              # noise variance  (estimated if None)
    rho=None,                 # noise correlation  (estimated if None)
    n_iter=200,               # EM iterations
    lr_h=0.2,                 # learning rate for h
    lr_J=0.2,                 # learning rate for J
    lr_beta1=0.1,             # learning rate for β₁
    use_exact_prior=True,     # exact enumeration of pMEM prior (N ≤ 14)
    use_exact_posterior=True, # exact enumeration of posterior
    sign_flip_protection=True,
    estimate_noise=True,
    seed=0,
)
```

**fit()**

```python
h, J, beta0, beta1, m = model.fit(verbose=False)
```

| Output | Shape | Description |
|--------|-------|-------------|
| `h` | `[N]` | pMEM external fields |
| `J` | `[N, N]` | pMEM pairwise couplings (symmetric, zero diagonal) |
| `beta0` | `[N]` | per-node baseline |
| `beta1` | `[N]` | per-node activation gain |
| `m` | `[T, N]` | posterior mean activation probability |

Post-fit attributes: `model.sigma2`, `model.rho`, `model.elbo_history`.

**Example**

```python
import numpy as np
from pmem_obs import PMEMObsSimple, simulate_pmem_basic

N, T = 6, 1000
h = np.array([-0.3, -0.1,  0.2, -0.25,  0.1, -0.15])
J = np.zeros((N, N))   # set couplings as needed
beta0 = np.zeros(N)
beta1 = np.ones(N)

Y, Z = simulate_pmem_basic(T, h, J, beta0, beta1, sigma2=0.5, rho=0.4)

model = PMEMObsSimple(Y, n_iter=200)
h_est, J_est, beta0_est, beta1_est, m_est = model.fit(verbose=True)

print(f"σ²={model.sigma2:.3f}  ρ={model.rho:.3f}")
```

---

## pMEM_Leadfield_v2

EEG/MEG sensor-space EM estimator. The leadfield matrix `L` enters the
likelihood directly — no source inversion is performed.

**Generative model**

```
z(t) ~ pMEM(h, J)
y(t) = L(β₀ + β₁ ⊙ z(t)) + ε(t),   ε(t) ~ N(0, Σ_ε)
Σ_ε = σ²[(1−ρ)I + ρ11ᵀ]
```

**Constructor** (dataclass)

```python
pMEM_Leadfield_v2(
    Y,                        # [T, M]  sensor observations
    L,                        # [M, N]  leadfield matrix
    n_iter=80,                # EM iterations
    lr_h=0.2,
    lr_J=0.2,
    sigma2=None,              # sensor noise variance  (estimated if None)
    rho=None,                 # sensor noise correlation  (estimated if None)
    use_exact_prior=True,
    use_exact_posterior=True,
    sign_flip_protection=True,
    estimate_noise=True,
    beta0_ridge=1e-2,         # Tikhonov ridge on β₀ (stabilises M < N case)
    seed=0,
)
```

**fit()**

```python
h, J, beta0, beta1, m = lf_model.fit(verbose=False)
```

Same output convention as `PMEMObsSimple`. Post-fit attributes:
`lf_model.h`, `lf_model.J`, `lf_model.sigma2`, `lf_model.rho`, `lf_model.elbo_history`.

**Example**

```python
import numpy as np
from pmem_leadfield import pMEM_Leadfield_v2

# Y_eeg : [T, M]  preprocessed EEG (e.g. alpha-band envelope)
# L     : [M, N]  leadfield from MNE or FieldTrip
L = np.load("leadfield_dmn6_64ch.npy")      # [64, 6]
lf_model = pMEM_Leadfield_v2(Y_eeg, L, n_iter=150)
lf_model.fit(verbose=True)

h, J = lf_model.h, lf_model.J
rho_hat = lf_model.rho
```

---

## Helper functions  (`pmem_obs.py`)

| Function | Returns | Description |
|----------|---------|-------------|
| `simulate_pmem_basic(T, h, J, beta0, beta1, sigma2, rho, seed)` | `(Y [T,N], Z [T,N])` | Generate synthetic observations from the generative model |
| `fit_pmem_pseudolikelihood(Z, n_iter, lr)` | `(h, J)` | Demean → threshold → pseudolikelihood baseline |
| `fit_gmm_threshold_pmem(Y, n_iter, lr, seed)` | `(h, J, Z)` | Per-node GMM adaptive-threshold ablation |
| `exact_prior_moments(h, J)` | `(m0 [N], m0_outer [N,N])` | Exact pMEM marginals via full enumeration |
| `rel_err(true, est)` | `float` | Relative RMSE between two arrays |

---

## Path configuration  (`paths.py`)

Output directories are controlled by environment variables — no code edits needed:

```bash
export PMEM_FIGDIR=/path/to/figures        # default: ./figures
export PMEM_CACHEDIR=/path/to/cache        # default: ./cache
export PMEM_DATADIR=/path/to/data          # default: ./data
export MNE_DATA=/path/to/mne_data          # default: ~/mne_data
export PMEM_SEDATION_DATA=/path/to/chennu  # default: ./data/Sedation-RestingState
```

---

## Citation

```bibtex
@article{jeong2026mempy,
  title   = {A generative framework for estimating brain energy landscapes
             from continuous neuroimaging observations},
  author  = {Jeong, Seok-Oh and Kim, Euisun and Kang, Jiyoung and
             Eo, Jinseok and Lee, Dongmyeong and Park, Hae-Jeong},
  journal = {NeuroImage},
  year    = {2026},
  note    = {Under review}
}
```
