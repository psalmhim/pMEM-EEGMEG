"""
example.py — minimal worked examples for PMEMObsSimple and pMEM_Leadfield_v2.

All data are synthetically generated; no external datasets are required.

Run:
    python example.py
"""
import numpy as np
import itertools
from pmem_obs import (
    PMEMObsSimple,
    simulate_pmem_basic,
    fit_pmem_pseudolikelihood,
    exact_prior_moments,
    rel_err,
)
from pmem_leadfield import pMEM_Leadfield_v2


# ──────────────────────────────────────────────────────────────────────────────
# Shared ground-truth pMEM parameters (N=6, tractable for exact enumeration)
# ──────────────────────────────────────────────────────────────────────────────
N = 6

h_true = np.array([-0.30, -0.10,  0.20, -0.25,  0.10, -0.15])

J_true = np.array([
    [ 0.00,  0.40, -0.10,  0.20,  0.00,  0.05],
    [ 0.40,  0.00,  0.30, -0.05,  0.10,  0.00],
    [-0.10,  0.30,  0.00,  0.25,  0.05, -0.10],
    [ 0.20, -0.05,  0.25,  0.00,  0.35,  0.10],
    [ 0.00,  0.10,  0.05,  0.35,  0.00,  0.30],
    [ 0.05,  0.00, -0.10,  0.10,  0.30,  0.00],
])


# ══════════════════════════════════════════════════════════════════════════════
# Example 1 — PMEMObsSimple  (direct-observation β-model)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Example 1: PMEMObsSimple  (direct-observation β-model)")
print("=" * 60)

# 1a. Simulate continuous observations from the generative model
#     y(t) = β₀ + β₁ ⊙ z(t) + ε(t),   ε ~ N(0, P⁻¹)
beta0_true = np.array([-0.5,  0.3, -0.2,  0.4, -0.1,  0.2])
beta1_true = np.array([ 1.2,  0.9,  1.1,  0.8,  1.3,  1.0])
sigma2, rho = 0.5, 0.4          # compound-symmetry noise parameters
T = 1000

Y, Z_true = simulate_pmem_basic(
    T=T, h=h_true, J=J_true,
    beta0=beta0_true, beta1=beta1_true,
    sigma2=sigma2, rho=rho, seed=0,
)
print(f"Simulated data:  Y shape={Y.shape},  Z shape={Z_true.shape}")
print(f"True firing rates:  {Z_true.mean(axis=0).round(2)}")

# 1b. Demean-and-threshold baseline
Y_dm   = Y - Y.mean(axis=0)
Z_naive = (Y_dm > 0).astype(int)
h_naive, J_naive = fit_pmem_pseudolikelihood(Z_naive, n_iter=300, lr=0.05)

# 1c. Fit the β-model
#     sigma2 and rho are estimated from data (no oracle information)
model = PMEMObsSimple(
    Y,
    n_iter=200,
    lr_h=0.2, lr_J=0.2,
    use_exact_prior=True,
    use_exact_posterior=True,   # exact enumeration (N=6 → 64 states)
    sign_flip_protection=True,
    estimate_noise=True,
    seed=0,
)
h_est, J_est, beta0_est, beta1_est, m_est = model.fit(verbose=True)

print(f"\nEstimated noise:  σ²={model.sigma2:.3f} (true {sigma2})  "
      f"ρ={model.rho:.3f} (true {rho})")

# 1d. Compare errors
print("\n--- Parameter recovery ---")
print(f"  h  error:  demean={rel_err(h_true, h_naive):.3f}  "
      f"β-model={rel_err(h_true, h_est):.3f}")
print(f"  J  error:  demean={rel_err(J_true, J_naive):.3f}  "
      f"β-model={rel_err(J_true, J_est):.3f}")
print(f"  β₀ error:  {rel_err(beta0_true, beta0_est):.3f}")
print(f"  β₁ error:  {rel_err(beta1_true, beta1_est):.3f}")

Z_est = (m_est > 0.5).astype(int)
print(f"  Latent accuracy:  demean={np.mean(Z_naive == Z_true):.3f}  "
      f"β-model={np.mean(Z_est == Z_true):.3f}")

# 1e. Energy landscape from estimated parameters
#     E(z) = -h^T z - z^T J z   (lower = more probable)
Zs = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)
J_up = np.triu(J_est, 1)
E = -(Zs @ h_est + np.einsum('ki,ij,kj->k', Zs, J_up, Zs))

# Local minima: states where all single-bit-flip neighbours have higher energy
def find_local_minima(E, Zs):
    minima = []
    for idx in range(len(Zs)):
        z = Zs[idx]
        is_min = True
        for i in range(N):
            z_flip = z.copy(); z_flip[i] = 1.0 - z_flip[i]
            dists = np.sum((Zs - z_flip) ** 2, axis=1)
            nbr_idx = np.argmin(dists)
            if E[nbr_idx] < E[idx]:
                is_min = False
                break
        if is_min:
            minima.append(idx)
    return minima

minima_idx = find_local_minima(E, Zs)
print(f"\n--- Energy landscape ({2**N} states) ---")
print(f"  Number of local minima: {len(minima_idx)}")
for idx in minima_idx:
    state = Zs[idx].astype(int)
    print(f"  Minimum  z={state}  E={E[idx]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Example 2 — pMEM_Leadfield_v2  (EEG/MEG sensor-space β-model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Example 2: pMEM_Leadfield_v2  (EEG/MEG leadfield β-model)")
print("=" * 60)

# 2a. Synthetic leadfield  L ∈ R^{M x N}
#     In practice: load a real leadfield from leadfield_dmn6_64ch.npy
#     Here we use a random Gaussian matrix (M=32 sensors, N=6 sources)
M_sensors = 32
rng = np.random.default_rng(42)
L_raw = rng.standard_normal((M_sensors, N))
# Column-normalize so each source column has unit norm (matches real leadfields)
L = L_raw / (np.linalg.norm(L_raw, axis=0, keepdims=True) + 1e-9)
print(f"Leadfield shape: L={L.shape}  ({M_sensors} sensors × {N} sources)")

# 2b. Simulate sensor-space observations
#     y(t) = L(β₀ + β₁ ⊙ z(t)) + ε(t),   ε ~ N(0, Σ_ε)
#     Σ_ε = σ²[(1−ρ)I + ρ11ᵀ]
rho_sensor = 0.4
sigma2_sensor = 0.5
cov_sensor = sigma2_sensor * (
    (1 - rho_sensor) * np.eye(M_sensors) +
    rho_sensor * np.ones((M_sensors, M_sensors))
)

Z_sim = np.zeros((T, N), dtype=int)
Y_sensor = np.zeros((T, M_sensors))
z = rng.integers(0, 2, size=N).astype(float)
for t in range(T):
    # Gibbs sweep for pMEM state
    for _ in range(5):
        for i in range(N):
            field = h_true[i] + J_true[i].dot(z) - J_true[i, i] * z[i]
            z[i] = float(rng.random() < 1.0 / (1.0 + np.exp(-field)))
    Z_sim[t] = z.astype(int)
    source_signal = beta0_true + beta1_true * z
    Y_sensor[t] = L @ source_signal + rng.multivariate_normal(
        np.zeros(M_sensors), cov_sensor
    )
print(f"Sensor data shape: Y={Y_sensor.shape}")

# 2c. Fit pMEM_Leadfield_v2
lf_model = pMEM_Leadfield_v2(
    Y=Y_sensor,
    L=L,
    n_iter=100,
    lr_h=0.2, lr_J=0.2,
    use_exact_prior=True,
    use_exact_posterior=True,
    sign_flip_protection=True,
    estimate_noise=True,
    seed=0,
)
h_lf, J_lf, beta0_lf, beta1_lf, m_lf = lf_model.fit(verbose=True)

print(f"\nEstimated sensor noise:  "
      f"σ²={lf_model.sigma2:.3f} (true {sigma2_sensor})  "
      f"ρ={lf_model.rho:.3f} (true {rho_sensor})")

# 2d. Compare errors
print("\n--- Parameter recovery ---")
print(f"  h  error:  {rel_err(h_true, h_lf):.3f}")
print(f"  J  error:  {rel_err(J_true, J_lf):.3f}")
print(f"  β₀ error:  {rel_err(beta0_true, beta0_lf):.3f}")
print(f"  β₁ error:  {rel_err(beta1_true, beta1_lf):.3f}")

Z_lf = (m_lf > 0.5).astype(int)
print(f"  Latent accuracy: {np.mean(Z_lf == Z_sim):.3f}")

# 2e. Load a real leadfield instead of synthetic
print("\n--- To use a real leadfield ---")
print("  L = np.load('leadfield_dmn6_64ch.npy')   # [64, 6] BioSemi-64 EEG")
print("  L = np.load('leadfield_dmn6_egi91ch.npy') # [91, 6] EGI-91 EEG")
print("  L = np.load('leadfield_dmn6_mag102ch.npy')# [102, 6] MEG magnetometers")
print("  lf_model = pMEM_Leadfield_v2(Y_eeg, L)")
print("  lf_model.fit()")
print("  rho_hat = lf_model.rho  # estimated residual sensor correlation")
