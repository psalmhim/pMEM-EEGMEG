"""
Hierarchical pMEM for EEG with a lead-field forward model.

Model
-----
    z(t) ~ pMEM(h, J)           binary state at each of N cortical sources
    s(t) = beta0 + beta1 ⊙ z(t) source amplitude
    y(t) = L s(t) + eps(t),     eps ~ N(0, Sigma_eps)
    Sigma_eps = sigma2 * (I + rho*(J_M - I))    compound-symmetry sensor noise (background EEG)

What this file fixes relative to `pmem.pMEM_Leadfield`
------------------------------------------------------
1. Mean-field reference for the M-step uses EXACT enumeration of the pMEM
   when N <= MAX_EXACT_N, else block-Gibbs marginals.  No more chasing a
   biased single-mode mean-field fixed point.
2. Closed-form WLS update for (beta0, beta1) jointly given m, accounting for
   the lead-field mixing (i.e. L^T P L is dense).  Original code used
   element-wise heuristics that ignored cross-source coupling through L.
3. Gauge / sign-flip protection: enforces the canonical convention "active
   state corresponds to higher signal in the sensor-projected source space"
   so that latent identifiability is resolved consistently across runs.
4. Sensor-space noise sigma2 estimated from data.
5. ELBO tracking, MAP latent extraction, source-projected warm start using a
   Tikhonov-regularized minimum-norm inverse.
6. Ridge prior on beta0 to handle the null space of L (M < N).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit


MAX_EXACT_N = 14    # 2^14 = 16384 states; beyond this fall back to Gibbs


# ------------------------------------------------------------
# pMEM prior helpers
# ------------------------------------------------------------
def _enumerate_states(N: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)


def exact_prior_moments(h, J, Zs=None):
    N = len(h)
    if Zs is None:
        Zs = _enumerate_states(N)
    J_up = np.triu(J, 1)
    E = Zs @ h + np.einsum('ki,ij,kj->k', Zs, J_up, Zs)
    E -= E.max()
    w = np.exp(E)
    w /= w.sum()
    m0 = w @ Zs
    m0_outer = (Zs.T * w) @ Zs
    return m0, m0_outer


def gibbs_prior_moments(h, J, n_burn=200, n_samples=2000, rng=None):
    """Fallback prior moment estimator for N > MAX_EXACT_N."""
    if rng is None:
        rng = np.random.default_rng()
    N = len(h)
    z = rng.integers(0, 2, size=N).astype(float)
    m1 = np.zeros(N)
    m2 = np.zeros((N, N))
    for k in range(n_burn + n_samples):
        for i in range(N):
            field = h[i] + np.dot(J[i], z) - J[i, i] * z[i]
            z[i] = float(rng.random() < expit(field))
        if k >= n_burn:
            m1 += z
            m2 += np.outer(z, z)
    m1 /= n_samples
    m2 /= n_samples
    return m1, m2


def make_sym_upper(M):
    M = np.triu(M, 1)
    M = M + M.T
    np.fill_diagonal(M, 0.0)
    return M


# ------------------------------------------------------------
# Lead-field pMEM
# ------------------------------------------------------------
@dataclass
class pMEM_Leadfield_v2:
    Y: np.ndarray                       # [T, M]
    L: np.ndarray                       # [M, N]
    n_iter: int = 80
    lr_h: float = 0.2
    lr_J: float = 0.2

    sigma2: float | None = None          # sensor noise variance; estimated if None
    rho:    float | None = None          # compound-symmetry correlation; estimated if None

    use_exact_prior: bool = True
    use_exact_posterior: bool = True    # exact enumeration for E-step (N<=MAX_EXACT_N)
    sign_flip_protection: bool = True
    estimate_noise: bool = True
    beta0_ridge: float = 1e-2           # Tikhonov on beta0 (handles M<N null space)
    seed: int = 0

    # populated in fit
    h: np.ndarray = field(init=False)
    J: np.ndarray = field(init=False)
    beta0: np.ndarray = field(init=False)
    beta1: np.ndarray = field(init=False)
    m: np.ndarray = field(init=False)
    elbo_history: list = field(default_factory=list)

    def __post_init__(self):
        self.T, self.M = self.Y.shape
        _, self.N = self.L.shape
        rng = np.random.default_rng(self.seed)

        # ---- sensor noise: initialize rho from signal-subtracted residuals ----
        # Raw Y includes source signal (L s) which is NOT compound-symmetric and
        # would bias rho upward.  A quick Tikhonov inverse removes the bulk of
        # the signal before fitting the compound-symmetry noise model.
        if self.sigma2 is None:
            Yc  = self.Y - self.Y.mean(axis=0, keepdims=True)
            LtL = self.L.T @ self.L
            lam_init = 0.1 * np.trace(LtL) / max(self.N, 1)
            W   = np.linalg.solve(LtL + lam_init * np.eye(self.N), self.L.T)
            R0  = Yc - (Yc @ W.T) @ self.L.T   # [T, M] approx noise residual
            S   = (R0.T @ R0) / self.T
            self.sigma2 = float(np.diag(S).mean())
            off = (S.sum() - np.trace(S)) / (self.M * (self.M - 1))
            self.rho = float(np.clip(off / (self.sigma2 + 1e-9), -0.3, 0.95))
        if self.rho is None:
            self.rho = 0.0

        self._update_P()

        # ---- pMEM init ------------------------------------------------
        self.h = np.zeros(self.N)
        J0 = 0.01 * rng.standard_normal((self.N, self.N))
        self.J = make_sym_upper(J0)

        # ---- warm start: MNE-style Tikhonov inverse -------------------
        Yc = self.Y - self.Y.mean(axis=0, keepdims=True)
        # noise-normalized leadfield gain ≈ scale of L^T y_t
        LtL = self.L.T @ self.L
        lam_w = 1e-2 * np.trace(LtL) / max(self.N, 1)
        W = np.linalg.solve(LtL + lam_w * np.eye(self.N), self.L.T)
        S_hat = Yc @ W.T                                # [T, N] source estimates
        med = np.median(S_hat, axis=0, keepdims=True)
        z_warm = (S_hat > med).astype(float)
        self.m = 0.05 + 0.90 * z_warm

        # beta1 init by sign-aware regression of S_hat on z_warm
        b1 = np.zeros(self.N)
        for i in range(self.N):
            zi = z_warm[:, i] - z_warm[:, i].mean()
            num = ((S_hat[:, i] - S_hat[:, i].mean()) * zi).sum()
            den = (zi ** 2).sum() + 1e-9
            b1[i] = num / den
        self.beta1 = np.maximum(np.abs(b1), 0.1)

        # beta0 closed form below; placeholder
        self.beta0 = S_hat.mean(axis=0) - self.beta1 * z_warm.mean(axis=0)

        if self.N <= MAX_EXACT_N:
            self._Zs = _enumerate_states(self.N)
        else:
            self._Zs = None

    # --------------------------------------------------------
    # Precision and projected leadfield quantities
    # --------------------------------------------------------
    def _update_P(self):
        a = max(1.0 - self.rho, 1e-4)
        b = self.rho
        denom = a * (a + self.M * b)
        I = np.eye(self.M)
        O = np.ones((self.M, self.M))
        self.P = (1.0 / max(self.sigma2, 1e-6)) * ((1.0 / a) * I - (b / denom) * O)
        self.LtPL = self.L.T @ self.P @ self.L              # [N, N]
        self.LtP  = self.L.T @ self.P                       # [N, M]
        self.LtPL_diag = np.diag(self.LtPL)

    # --------------------------------------------------------
    # Posterior natural parameters at time t
    # --------------------------------------------------------
    def _posterior_nat_params(self, y_t):
        # residual to baseline drive Lβ0
        r = y_t - self.L @ self.beta0
        # ∂/∂z_i of -0.5 (y - Lβ0 - L diag(β1) z)^T P (...) yields:
        #   β1_i * [L^T P r]_i for the linear term
        #  -0.5 * β1_i^2 * (L^T P L)_ii for the diagonal quadratic
        #  -β1_i β1_j (L^T P L)_ij for the cross term in tilde_J
        Lp_r = self.LtP @ r                                 # [N]
        tilde_h = self.h + self.beta1 * Lp_r \
                  - 0.5 * (self.beta1 ** 2) * self.LtPL_diag
        B1B1 = np.outer(self.beta1, self.beta1)
        cross = B1B1 * self.LtPL
        np.fill_diagonal(cross, 0.0)
        tilde_J = self.J - cross
        return tilde_h, tilde_J

    # --------------------------------------------------------
    # E-step (mean-field OR exact enumeration)
    #
    # When exact posterior is used, we also store the pairwise posterior
    # <z_i z_j> per time step so the M-step J update is not biased by the
    # mean-field independence assumption.
    # --------------------------------------------------------
    def E_step(self, sweeps: int = 20):
        if self.use_exact_posterior and self.N <= MAX_EXACT_N:
            self._E_step_exact()
        else:
            self._E_step_meanfield(sweeps=sweeps)

    def _E_step_meanfield(self, sweeps: int = 20):
        # Sweep i = N-1, N-2, ..., 0 (downward) so that j > i nodes (already
        # updated in this sweep) feed into node i's field — consistent with the
        # manuscript's sequential update direction.
        for t in range(self.T):
            tilde_h, tilde_J = self._posterior_nat_params(self.Y[t])
            m_t = self.m[t].copy()
            for _ in range(sweeps):
                m_old = m_t.copy()
                for i in range(self.N - 1, -1, -1):
                    field = tilde_h[i] + np.dot(tilde_J[i], m_t)   # tilde_J diagonal is 0
                    m_t[i] = expit(field)
                if np.max(np.abs(m_t - m_old)) < 1e-6:
                    break
            self.m[t] = np.clip(m_t, 1e-4, 1.0 - 1e-4)
        # mean-field has no exact pairwise moments stored
        self._m_outer_post = None

    def _E_step_exact(self):
        """
        Vectorised exact E-step: no Python loop over T.

        tilde_J is constant across time (depends only on fixed params).
        tilde_h(t) varies linearly with Y[t].  We batch all T at once:
          E_all[k,t] = z_k · tilde_h(t) + z_k · tJ_up · z_k   (K×T)
        then softmax over k, and accumulate m_post and m_outer_post via
        matrix multiplies — loop-free and GPU-ready.
        """
        Zs = self._Zs                                   # (K, N), K=2^N

        # ---- constant part: tilde_J and its quadratic contribution ----
        cross = np.outer(self.beta1, self.beta1) * self.LtPL
        np.fill_diagonal(cross, 0.0)
        tJ_up = np.triu(self.J - cross, 1)              # (N, N) constant
        E_quad = np.einsum('ki,ij,kj->k', Zs, tJ_up, Zs)  # (K,) constant

        # ---- batch tilde_h over all T ----
        # r(t) = Y[t] - L @ beta0  →  R = Y - L@beta0  shape (T, M)
        # LtP @ r(t) = LtP @ R[t]  →  batch: R @ LtP.T  shape (T, N)
        R = self.Y - (self.L @ self.beta0)              # (T, M)
        LtP_R = R @ self.LtP.T                         # (T, N)
        h_const = self.h - 0.5 * (self.beta1 ** 2) * self.LtPL_diag  # (N,)
        tilde_H = h_const + self.beta1 * LtP_R         # (T, N)

        # ---- energy matrix ----
        E_lin = Zs @ tilde_H.T                         # (K, T)
        E_all = E_lin + E_quad[:, None]                 # (K, T)
        E_all -= E_all.max(axis=0)                      # numerical stability

        # ---- softmax weights ----
        w_all = np.exp(E_all)                           # (K, T)
        w_all /= w_all.sum(axis=0)                      # normalise over K

        # ---- posterior statistics ----
        m_post = w_all.T @ Zs                           # (T, N)
        w_sum  = w_all.sum(axis=1)                      # (K,)  Σ_t w[k,t]
        m_outer_post = (Zs.T * w_sum) @ Zs / self.T    # (N, N)

        self.m = np.clip(m_post, 1e-4, 1.0 - 1e-4)
        self._m_outer_post = m_outer_post               # for M-step

    # --------------------------------------------------------
    # Gauge protection: canonical "active ⇒ stronger projected source"
    # --------------------------------------------------------
    def _fix_sign_flips(self):
        # use Tikhonov-projected source estimate as the reference signal per node
        Yc = self.Y - self.Y.mean(axis=0, keepdims=True)
        LtL = self.L.T @ self.L
        lam_w = 1e-2 * np.trace(LtL) / max(self.N, 1)
        S_hat = Yc @ np.linalg.solve(LtL + lam_w * np.eye(self.N), self.L.T).T
        for i in range(self.N):
            if self.m[:, i].std() < 1e-6 or S_hat[:, i].std() < 1e-6:
                continue
            c = np.corrcoef(self.m[:, i], S_hat[:, i])[0, 1]
            if np.isnan(c) or c >= -0.05:
                continue
            self.m[:, i] = 1.0 - self.m[:, i]
            self.beta0[i] = self.beta0[i] + self.beta1[i]
            self.beta1[i] = -self.beta1[i]
        neg = self.beta1 < 0
        if np.any(neg):
            self.beta1[neg] = -self.beta1[neg]
            self.m[:, neg] = 1.0 - self.m[:, neg]
            self.beta0[neg] = self.beta0[neg] - self.beta1[neg]
        self.beta1 = np.maximum(self.beta1, 0.05)

    # --------------------------------------------------------
    # M-step
    # --------------------------------------------------------
    def M_step(self):
        # ---- (beta0, beta1): two alternating closed-form sweeps ----------
        # Both updates are optimal given the other fixed; two sweeps of the
        # outer EM loop are enough for practical convergence.
        for _ in range(2):
            # β0 — Tikhonov-regularized source-space WLS (δI handles null(L))
            r_bar = self.Y.mean(axis=0) - self.L @ (self.beta1 * self.m.mean(axis=0))
            A0 = self.LtPL + self.beta0_ridge * np.eye(self.N)
            self.beta0 = np.linalg.solve(A0, self.LtP @ r_bar)

            # β1 — N×N linear system A β1 = b,  A_{kk'} = K_{kk'} * sum_t C_{kk'}(t)
            # K = L^T P L.  Diagonal exact: C_{kk}(t) = m_k(t) (binary identity).
            # Off-diagonal: exact posterior E_q[z_k z_k'] when available (enumeration),
            # mean-field m_k m_k' + diagonal V(t) correction otherwise.
            R0 = self.Y - (self.L @ self.beta0)[None, :]            # [T, M]
            G = R0 @ self.LtP.T                                     # [T, N]
            numer = (self.m * G).sum(axis=0)                        # [N]
            if self._m_outer_post is not None:
                sum_C = self._m_outer_post * self.T                 # [N, N]: sum_t C(t)
                A1 = self.LtPL * sum_C                              # diagonal already exact
            else:
                mm    = self.m.T @ self.m                           # [N, N]: sum_t m_i m_j
                sum_V = (self.m * (1.0 - self.m)).sum(axis=0)       # [N]
                A1 = self.LtPL * mm + np.diag(self.LtPL_diag * sum_V)
            A1 += 1e-9 * np.eye(self.N)
            self.beta1 = np.maximum(np.linalg.solve(A1, numer), 0.05)

        # ---- h and J moment matching ---------------------------------
        m_bar = self.m.mean(axis=0)
        # Use exact posterior pairwise <z_i z_j> if available; otherwise fall
        # back to the (biased) mean-field outer product.
        if getattr(self, "_m_outer_post", None) is not None:
            m_outer = self._m_outer_post
        else:
            m_outer = (self.m.T @ self.m) / self.T

        if self.use_exact_prior and self.N <= MAX_EXACT_N:
            m0, m0_outer = exact_prior_moments(self.h, self.J, Zs=self._Zs)
        else:
            m0, m0_outer = gibbs_prior_moments(self.h, self.J)
        self.h += self.lr_h * (m_bar - m0)
        grad_J = np.triu(m_outer - m0_outer, 1)
        self.J += self.lr_J * grad_J
        self.J = make_sym_upper(self.J)

        # ---- sensor noise update (compound-symmetry) ------------------
        # E[r̄ r̄^T] = Σ_ε + L diag(β₁) Cov_q[z] diag(β₁) L^T
        # Subtract the sensor-space posterior-variance contamination to recover Σ_ε.
        if self.estimate_noise:
            S_drive = self.beta0[None, :] + self.beta1[None, :] * self.m
            mu = S_drive @ self.L.T                                 # [T, M]
            R  = self.Y - mu                                        # [T, M]
            S  = (R.T @ R) / self.T                                 # [M, M]
            if self._m_outer_post is not None:
                avg_cov_q = self._m_outer_post - (self.m.T @ self.m) / self.T  # [N, N]
            else:
                avg_cov_q = np.diag((self.m * (1.0 - self.m)).mean(axis=0))    # diagonal
            B = self.L * self.beta1[None, :]                        # [M, N]: L diag(β₁)
            S = S - B @ avg_cov_q @ B.T                             # [M, M]
            diag_var = max(float(np.diag(S).mean()), 1e-4)
            off_cov  = float((S.sum() - np.trace(S)) / (self.M * (self.M - 1)))
            sigma2_new = diag_var
            rho_new    = float(np.clip(off_cov / (sigma2_new + 1e-9), -0.3, 0.95))
            self.sigma2 = 0.5 * self.sigma2 + 0.5 * sigma2_new
            self.rho    = 0.5 * self.rho    + 0.5 * rho_new
            self._update_P()

    # --------------------------------------------------------
    # ELBO (exact prior log-Z when N small)
    # --------------------------------------------------------
    def _elbo(self):
        m_bar = self.m.mean(axis=0)
        m_outer = (self.m.T @ self.m) / self.T
        if self._Zs is not None:
            E = self._Zs @ self.h + np.einsum('ki,ij,kj->k', self._Zs, np.triu(self.J, 1), self._Zs)
            logZ = E.max() + np.log(np.exp(E - E.max()).sum())
        else:
            logZ = 0.0
        prior_term = self.T * (m_bar @ self.h
                               + np.einsum('ij,ij->', np.triu(self.J, 1), m_outer)
                               - logZ)
        mu = (self.beta0 + self.beta1 * self.m) @ self.L.T
        R = self.Y - mu
        obs = -0.5 * np.einsum('ti,ij,tj->', R, self.P, R)
        eps = 1e-12
        mc = np.clip(self.m, eps, 1 - eps)
        H = -(mc * np.log(mc) + (1 - mc) * np.log(1 - mc)).sum()
        return float(prior_term + obs + H)

    # --------------------------------------------------------
    def fit(self, verbose=False):
        for it in range(self.n_iter):
            self.E_step()
            if self.sign_flip_protection and (it < 5 or it % 20 == 0):
                self._fix_sign_flips()
            self.M_step()
            if verbose and ((it + 1) % 10 == 0 or it == self.n_iter - 1):
                elbo = self._elbo()
                self.elbo_history.append(elbo)
                print(f"[pMEM_Leadfield_v2] it {it+1}/{self.n_iter}  ELBO={elbo:.2f}  "
                      f"σ²={self.sigma2:.3f}  ρ={self.rho:.3f}")
        return self.h, self.J, self.beta0, self.beta1, self.m
