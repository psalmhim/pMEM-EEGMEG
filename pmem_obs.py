"""
Core pMEM observation model for direct (non-leadfield) settings.

Public API
----------
PMEMObsSimple          — EM estimator: jointly fits (h, J, β₀, β₁, σ², ρ)
simulate_pmem_basic    — generate (Y, Z) from the generative model
fit_pmem_pseudolikelihood — demean-and-threshold baseline
fit_gmm_threshold_pmem — GMM-threshold ablation baseline
exact_prior_moments    — enumerate all 2^N pMEM states exactly
rel_err                — relative L2 error utility
"""
import itertools

import numpy as np
from scipy.special import expit


# ==========================================================
# Utility
# ==========================================================

def rel_err(true, est, eps=1e-12):
    denom = np.linalg.norm(true)
    if denom < eps:
        return np.linalg.norm(est - true)
    return np.linalg.norm(est - true) / denom


def make_sym_upper(M):
    M = np.triu(M, 1)
    M = M + M.T
    np.fill_diagonal(M, 0.0)
    return M


# ==========================================================
# Compound symmetry precision helper
# ==========================================================
def compound_symmetry_precision(N, sigma2, rho):
    a = 1.0 - rho
    b = rho
    I = np.eye(N)
    O = np.ones((N, N))
    denom = a * (a + N * b)
    return (1.0 / sigma2) * ((1.0 / a) * I - (b / denom) * O)


# ==========================================================
# Exact prior moments via enumeration (feasible for N up to ~14)
# ==========================================================
def _enumerate_states(N):
    return np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)


def exact_prior_moments(h, J, Zs=None):
    """
    Compute exact <z_i> and <z_i z_j> under the pMEM prior
        p(z) ∝ exp(h^T z + sum_{i<j} J_ij z_i z_j)
    by enumerating all 2^N configurations.
    """
    N = len(h)
    if Zs is None:
        Zs = _enumerate_states(N)
    J_up = np.triu(J, 1)
    E = Zs @ h + np.einsum('ki,ij,kj->k', Zs, J_up, Zs)
    E -= E.max()
    w = np.exp(E)
    w /= w.sum()
    m0 = w @ Zs                              # <z_i>
    m0_outer = (Zs.T * w) @ Zs               # <z_i z_j>; diag = <z_i>
    return m0, m0_outer


# ==========================================================
# Simulator (unchanged)
# ==========================================================
def simulate_pmem_basic(T, h, J, beta0, beta1, sigma2=0.5, rho=0.4, seed=0):
    rng = np.random.RandomState(seed)
    N = len(h)

    Z = np.zeros((T, N), dtype=int)
    Y = np.zeros((T, N), dtype=float)

    cov = sigma2 * ((1.0 - rho) * np.eye(N) + rho * np.ones((N, N)))

    z_prev = rng.binomial(1, 0.5, size=N).astype(int)

    for t in range(T):
        z = z_prev.copy()
        for _ in range(5):
            for i in range(N):
                field = h[i] + np.dot(J[i], z) - J[i, i] * z[i]
                p = expit(field)
                z[i] = rng.rand() < p
        Z[t] = z
        mean = beta0 + beta1 * z
        Y[t] = rng.multivariate_normal(mean, cov)
        z_prev = z

    return Y, Z


# ==========================================================
# Baseline (unchanged)
# ==========================================================
def fit_pmem_pseudolikelihood(Z, n_iter=300, lr=0.05):
    T, N = Z.shape

    h = np.zeros(N)
    J = np.zeros((N, N))

    for _ in range(n_iter):
        grad_h = np.zeros(N)
        grad_J = np.zeros((N, N))

        for t in range(T):
            z = Z[t]
            for i in range(N):
                eta = h[i] + np.dot(J[i], z) - J[i, i] * z[i]
                p = expit(eta)
                diff = z[i] - p
                grad_h[i] += diff
                grad_J[i] += diff * z

        h += lr * grad_h / T
        J += lr * grad_J / T
        J = make_sym_upper(J)

    return h, J


# ==========================================================
# GMM-threshold baseline
# ==========================================================
def fit_gmm_threshold_pmem(Y, n_iter=300, lr=0.05, seed=0):
    """
    Per-node 2-Gaussian mixture → adaptive threshold → pseudolikelihood pMEM.

    Each node is binarized independently: the EM-fitted 2-Gaussian boundary
    replaces the per-channel mean, adapting to heterogeneous (β₀, β₁) without
    modelling cross-channel noise correlations.  Serves as a 'best adaptive
    threshold' ablation baseline between demean and the full β-model.
    """
    from sklearn.mixture import GaussianMixture
    T, N = Y.shape
    Z = np.zeros((T, N), dtype=int)
    for i in range(N):
        yi = Y[:, i].reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=seed, n_init=3)
        gmm.fit(yi)
        labels = gmm.predict(yi)
        # enforce: higher-mean component → z=1
        if gmm.means_[0, 0] > gmm.means_[1, 0]:
            labels = 1 - labels
        Z[:, i] = labels
    h, J = fit_pmem_pseudolikelihood(Z, n_iter=n_iter, lr=lr)
    return h, J, Z


# ==========================================================
# Proposed method (FIXED)
#
# Changes vs. original:
#   1) M-step prior reference uses EXACT enumeration of the pMEM,
#      not mean-field, so h and J no longer chase a biased target.
#   2) WLS closed-form update for beta1 instead of slow gradient.
#   3) Sign-flip detection after every E-step: if corr(m_i, y_i) < 0
#      then (m_i, beta0_i, beta1_i) are flipped to the matching gauge.
#   4) ELBO tracked for convergence diagnostics.
# ==========================================================

class PMEMObsSimple:
    def __init__(self, Y,
                 sigma2=None, rho=None,
                 n_iter=200, lr_h=0.2, lr_J=0.2, lr_beta1=0.1,
                 use_exact_prior=True,
                 use_exact_posterior=True,
                 sign_flip_protection=True,
                 estimate_noise=True,
                 seed=0):
        """
        sigma2, rho:
            If None (default) AND estimate_noise=True, the noise (sigma2, rho)
            is estimated from the data via the M-step.
            If provided AND estimate_noise=False, these are treated as known
            (oracle mode — useful for ablations, NOT for a fair comparison
            against demean+pseudolikelihood).
        """
        self.Y = Y
        self.T, self.N = Y.shape

        self.estimate_noise = estimate_noise

        if sigma2 is None or estimate_noise:
            # data-driven start: pooled within-channel variance and
            # off-diagonal correlation of the demeaned signal
            Yc = Y - Y.mean(axis=0, keepdims=True)
            S = (Yc.T @ Yc) / self.T
            self.sigma2 = float(np.diag(S).mean())
            off = (S.sum() - np.trace(S)) / (self.N * (self.N - 1))
            self.rho = float(np.clip(off / (self.sigma2 + 1e-9), -0.49, 0.95))
        else:
            self.sigma2 = float(sigma2)
            self.rho = float(rho)

        self.P = compound_symmetry_precision(self.N, self.sigma2, self.rho)
        self.P_diag = np.diag(self.P)

        self.h = np.zeros(self.N)
        rng = np.random.RandomState(seed)
        J0 = 0.01 * rng.randn(self.N, self.N)
        self.J = make_sym_upper(J0)

        # Warm start beta1 by sign-aware regression of (Y - Ybar) on (Y > median).
        # This anchors beta1 with the correct sign per node, avoiding the
        # symmetric initialization at beta1 = 1.
        Yc = Y - Y.mean(axis=0, keepdims=True)
        z_warm = (Yc > 0).astype(float)
        b1 = np.zeros(self.N)
        for i in range(self.N):
            num = (Yc[:, i] * (z_warm[:, i] - z_warm[:, i].mean())).sum()
            den = ((z_warm[:, i] - z_warm[:, i].mean()) ** 2).sum() + 1e-9
            b1[i] = num / den
        self.beta1 = np.maximum(np.abs(b1), 0.1)

        # m initialization from sign-aware warm threshold
        self.m = 0.05 + 0.90 * z_warm
        self.beta0 = Y.mean(axis=0) - self.beta1 * self.m.mean(axis=0)

        self.n_iter = n_iter
        self.lr_h = lr_h
        self.lr_J = lr_J
        self.lr_beta1 = lr_beta1

        self.use_exact_prior = use_exact_prior
        self.use_exact_posterior = use_exact_posterior
        self.sign_flip_protection = sign_flip_protection

        if self.use_exact_prior or self.use_exact_posterior:
            self._Zs = _enumerate_states(self.N)
        else:
            self._Zs = None

        self._m_outer_post = None
        self.elbo_history = []

    # ------------------------------------------------------
    # Posterior natural parameters (paper Eq. 13, 14)
    # ------------------------------------------------------
    def _posterior_nat_params(self, y):
        resid = y - self.beta0
        Py = self.P @ resid
        tilde_h = self.h + self.beta1 * Py - 0.5 * (self.beta1 ** 2) * self.P_diag

        # tilde_J upper-tri only, then symmetrized for convenient field calc
        B1 = self.beta1
        tilde_J = self.J - np.triu(np.outer(B1, B1) * self.P, 1) \
                          - np.tril(np.outer(B1, B1) * self.P, -1)
        # (Note: self.J is already symmetric with zero diag; we subtract the
        #  symmetric beta1 beta1^T P off-diagonal, equivalent to the paper.)
        return tilde_h, tilde_J

    # ------------------------------------------------------
    # E-step — exact enumeration when feasible, else mean-field
    # ------------------------------------------------------
    def E_step(self):
        if self.use_exact_posterior and self._Zs is not None:
            self._E_step_exact()
        else:
            self._E_step_meanfield()

    def _E_step_meanfield(self):
        for t in range(self.T):
            tilde_h, tilde_J = self._posterior_nat_params(self.Y[t])
            m_t = self.m[t].copy()
            for _ in range(25):
                m_old = m_t.copy()
                for i in range(self.N):
                    field = tilde_h[i] + np.dot(tilde_J[i], m_t) - tilde_J[i, i] * m_t[i]
                    m_t[i] = expit(field)
                if np.max(np.abs(m_t - m_old)) < 1e-6:
                    break
            self.m[t] = np.clip(m_t, 1e-4, 1.0 - 1e-4)
        self._m_outer_post = None

    def _E_step_exact(self):
        Zs = self._Zs
        m_post = np.zeros((self.T, self.N))
        m_outer_post = np.zeros((self.N, self.N))
        for t in range(self.T):
            tilde_h, tilde_J = self._posterior_nat_params(self.Y[t])
            tJ_up = np.triu(tilde_J, 1)
            E = Zs @ tilde_h + np.einsum('ki,ij,kj->k', Zs, tJ_up, Zs)
            E -= E.max()
            w = np.exp(E)
            w /= w.sum()
            m_post[t] = w @ Zs
            m_outer_post += (Zs.T * w) @ Zs
        m_outer_post /= self.T
        self.m = np.clip(m_post, 1e-4, 1.0 - 1e-4)
        self._m_outer_post = m_outer_post

    # ------------------------------------------------------
    # Sign-flip protection
    #
    # For each node check if posterior m[:, i] is anti-correlated with the
    # observation channel (after removing baseline). If so, flip:
    #     m_i        -> 1 - m_i
    #     beta1_i    -> -beta1_i        (then clipped positive)
    #     beta0_i    -> beta0_i + beta1_i_old
    # This leaves the observation model invariant and produces the canonical
    # gauge "active state => higher signal".
    # ------------------------------------------------------
    def _fix_sign_flips(self):
        Yc = self.Y - self.Y.mean(axis=0, keepdims=True)
        for i in range(self.N):
            mi = self.m[:, i]
            if mi.std() < 1e-6:
                continue
            c = np.corrcoef(mi, Yc[:, i])[0, 1]
            if c < -0.05:                       # clearly flipped
                self.m[:, i] = 1.0 - self.m[:, i]
                self.beta0[i] = self.beta0[i] + self.beta1[i]
                self.beta1[i] = -self.beta1[i]
        # Enforce positive beta1 gauge
        neg = self.beta1 < 0
        if np.any(neg):
            # flip everything back to positive beta1 convention
            self.beta1[neg] = -self.beta1[neg]
            self.m[:, neg] = 1.0 - self.m[:, neg]
            self.beta0[neg] = self.beta0[neg] - self.beta1[neg]  # reverse of above
        self.beta1 = np.maximum(self.beta1, 0.05)

    # ------------------------------------------------------
    # M-step
    # ------------------------------------------------------
    def M_step(self):
        # closed-form beta0
        self.beta0 = (self.Y - self.beta1[None, :] * self.m).mean(axis=0)

        # Exact closed-form for beta1 with full compound-symmetry P.
        # Setting d ELBO / d beta1_k = 0 for all k gives the N×N system A β1 = b:
        #   A_{k,k'}  = P_{k,k'} * sum_t E_q[z_k z_k']   (exact posterior pairwise)
        #   A_{k,k}   = P_{k,k} * sum_t m_k(t)           (binary identity E[z²]=m)
        #   b_k       = sum_t m_k(t) [P(y(t)-β0)]_k
        # Use exact posterior pairwise moments when available (exact E-step);
        # fall back to mean-field outer product otherwise.
        if self._m_outer_post is not None:
            m_outer = self._m_outer_post * self.T   # sum_t E_q[z z^T]
        else:
            m_outer = self.m.T @ self.m             # mean-field fallback: sum_t m m^T
        A = m_outer * self.P                        # N×N, P-weighted pairwise moments
        np.fill_diagonal(A, self.m.sum(axis=0) * self.P_diag)   # binary-z diagonal fix
        b = (self.m * ((self.Y - self.beta0[None, :]) @ self.P)).sum(axis=0)
        try:
            self.beta1 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # fallback to diagonal closed-form if A is singular
            self.beta1 = b / (np.diag(A) + 1e-9)
        self.beta1 = np.maximum(self.beta1, 0.05)

        # moment matching for h and J using exact prior moments
        m_bar = self.m.mean(axis=0)
        if self._m_outer_post is not None:
            m_outer = self._m_outer_post           # exact posterior pairwise
        else:
            m_outer = (self.m.T @ self.m) / self.T  # mean-field outer (biased)

        if self.use_exact_prior:
            m0, m0_outer = exact_prior_moments(self.h, self.J, Zs=self._Zs)
        else:
            m0 = np.full(self.N, 0.5)
            for _ in range(50):
                m0_old = m0.copy()
                for i in range(self.N):
                    field = self.h[i] + np.dot(self.J[i], m0) - self.J[i, i] * m0[i]
                    m0[i] = expit(field)
                if np.max(np.abs(m0 - m0_old)) < 1e-6:
                    break
            m0_outer = np.outer(m0, m0)

        self.h += self.lr_h * (m_bar - m0)
        grad_J = np.triu(m_outer - m0_outer, 1)
        self.J += self.lr_J * grad_J
        self.J = make_sym_upper(self.J)

        # noise update (compound-symmetry assumption):
        # E[r̄ r̄^T] = Σ_ε + β₁β₁^T ⊙ Cov_q[z], so subtract the posterior-
        # variance contamination to recover Σ_ε before fitting σ² and ρ.
        if self.estimate_noise:
            R = self.Y - self.beta0 - self.beta1 * self.m
            S = (R.T @ R) / self.T
            # exact Cov_q from enumeration when available; diagonal V(t) under mean-field
            if self._m_outer_post is not None:
                avg_cov_q = self._m_outer_post - (self.m.T @ self.m) / self.T
            else:
                avg_cov_q = np.diag((self.m * (1.0 - self.m)).mean(axis=0))
            S = S - np.outer(self.beta1, self.beta1) * avg_cov_q
            diag_var = max(float(np.diag(S).mean()), 1e-4)
            off_cov = float((S.sum() - np.trace(S)) / (self.N * (self.N - 1)))
            sigma2_new = diag_var
            rho_new = float(np.clip(off_cov / (sigma2_new + 1e-9), -0.49, 0.95))
            self.sigma2 = sigma2_new
            self.rho = rho_new
            self.P = compound_symmetry_precision(self.N, self.sigma2, self.rho)
            self.P_diag = np.diag(self.P)

    # ------------------------------------------------------
    # ELBO (for monitoring only)
    # ------------------------------------------------------
    def _elbo(self):
        # prior term
        m_bar = self.m.mean(axis=0)
        m_outer = (self.m.T @ self.m) / self.T
        # log Z computed exactly
        Zs = self._Zs if self._Zs is not None else _enumerate_states(self.N)
        E = Zs @ self.h + np.einsum('ki,ij,kj->k', Zs, np.triu(self.J, 1), Zs)
        logZ = E.max() + np.log(np.exp(E - E.max()).sum())
        prior_term = self.T * (m_bar @ self.h
                               + np.einsum('ij,ij->', np.triu(self.J, 1), m_outer)
                               - logZ)

        # observation term
        R = self.Y - self.beta0 - self.beta1 * self.m
        obs = -0.5 * np.einsum('ti,ij,tj->', R, self.P, R)

        # entropy
        eps = 1e-12
        m = np.clip(self.m, eps, 1 - eps)
        H = -(m * np.log(m) + (1 - m) * np.log(1 - m)).sum()
        return float(prior_term + obs + H)

    # ------------------------------------------------------
    def fit(self, verbose=False):
        for it in range(self.n_iter):
            self.E_step()
            if self.sign_flip_protection and (it < 5 or it % 20 == 0):
                self._fix_sign_flips()
            self.M_step()

            if verbose and ((it + 1) % 50 == 0 or it == self.n_iter - 1):
                elbo = self._elbo()
                self.elbo_history.append(elbo)
                print(f"[PMEMObsSimple] iter {it + 1}/{self.n_iter}  ELBO={elbo:.2f}")

        return self.h, self.J, self.beta0, self.beta1, self.m


# ==========================================================
# Main experiment
# ==========================================================
if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.RandomState(0)

    N = 8
    T = 1000
    n_rep = 30

    h_true = np.array([
        -0.1485, -0.0646, -0.2941, -0.4084,
        -0.1705, -0.3360, -0.1381, -0.1385
    ], dtype=float)

    J_true = np.array([
        [0,        0.3051, -0.1035, -0.0325,  0.0089,  0.0148,  0.0299,  0.0912],
        [0.3051,   0,      -0.0633,  0.0551,  0.0102, -0.1042,  0.0252, -0.0942],
        [-0.1035, -0.0633,  0,       0.3468,  0.0151,  0.2371,  0.1649, -0.0121],
        [-0.0325,  0.0551,  0.3468,  0,       0.2343,  0.1791,  0.0497, -0.0195],
        [0.0089,   0.0102,  0.0151,  0.2343,  0,       0.0747, -0.0182,  0.0156],
        [0.0148,  -0.1042,  0.2371,  0.1791,  0.0747,  0,      -0.0058,  0.2953],
        [0.0299,   0.0252,  0.1649,  0.0497, -0.0182, -0.0058,  0,       0.0287],
        [0.0912,  -0.0942, -0.0121, -0.0195,  0.0156,  0.2953,  0.0287,  0]
    ], dtype=float)

    sigma2 = 0.5
    rho = 0.4
    beta_sd_list = [0.0, 0.1, 0.3, 0.5, 0.8]

    print("\n===== EEG/fMRI-motivated simulation: demean vs beta-estimation =====")
    print(f"N={N}, T={T}, n_rep={n_rep}, sigma2={sigma2}, rho={rho}")
    print("Fixes: exact-prior M-step, WLS beta1, sign-flip protection,")
    print("       noise (sigma2, rho) ESTIMATED from data (no oracle).\n")

    all_results = []

    for beta_sd in beta_sd_list:
        err_h_naive, err_J_naive = [], []
        err_h_obs,   err_J_obs   = [], []
        err_b0, err_b1 = [], []
        acc_naive_list, acc_obs_list = [], []

        for rep in range(n_rep):
            beta0_true = rng.normal(0.0, 0.7, size=N)

            if beta_sd == 0.0:
                beta1_true = np.ones(N)
            else:
                beta1_true = rng.normal(1.0, beta_sd, size=N)
                beta1_true = np.maximum(beta1_true, 0.1)

            Y, Z_true = simulate_pmem_basic(
                T=T, h=h_true, J=J_true,
                beta0=beta0_true, beta1=beta1_true,
                sigma2=sigma2, rho=rho, seed=1000 + rep
            )

            # Baseline
            Y_demean = Y - Y.mean(axis=0, keepdims=True)
            Z_naive = (Y_demean > 0).astype(int)
            h_naive, J_naive = fit_pmem_pseudolikelihood(Z_naive, n_iter=300, lr=0.05)

            # Proposed (fixed) — fair comparison: sigma2 and rho are
            # ESTIMATED from data, just like the baseline gets no oracle info.
            obs_model = PMEMObsSimple(
                Y,
                n_iter=200, lr_h=0.2, lr_J=0.2,
                use_exact_prior=True,
                sign_flip_protection=True,
                estimate_noise=True,
                seed=rep,
            )
            h_obs, J_obs, beta0_obs, beta1_obs, m_obs = obs_model.fit(verbose=False)
            Z_obs = (m_obs > 0.5).astype(int)

            acc_naive_list.append(np.mean(Z_naive == Z_true))
            acc_obs_list.append(np.mean(Z_obs == Z_true))

            err_h_naive.append(rel_err(h_true, h_naive))
            err_J_naive.append(rel_err(J_true, J_naive))
            err_h_obs.append(rel_err(h_true, h_obs))
            err_J_obs.append(rel_err(J_true, J_obs))
            err_b0.append(rel_err(beta0_true, beta0_obs))
            err_b1.append(rel_err(beta1_true, beta1_obs))

        res = {
            "beta_sd":   float(beta_sd),
            "demean_h":  float(np.mean(err_h_naive)),
            "demean_J":  float(np.mean(err_J_naive)),
            "obs_h":     float(np.mean(err_h_obs)),
            "obs_J":     float(np.mean(err_J_obs)),
            "beta0_err": float(np.mean(err_b0)),
            "beta1_err": float(np.mean(err_b1)),
            "acc_naive": float(np.mean(acc_naive_list)),
            "acc_obs":   float(np.mean(acc_obs_list)),
        }
        all_results.append(res)

        print("--------------------------------------------------")
        print(f"beta_sd = {beta_sd:.2f}")
        print("demean:   h_err={:.3f}  J_err={:.3f}  latent_acc={:.3f}".format(
            res["demean_h"], res["demean_J"], res["acc_naive"]
        ))
        print("beta-mod: h_err={:.3f}  J_err={:.3f}  latent_acc={:.3f}".format(
            res["obs_h"], res["obs_J"], res["acc_obs"]
        ))
        print("beta recovery: b0_err={:.3f}  b1_err={:.3f}".format(
            res["beta0_err"], res["beta1_err"]
        ))
        print("delta latent_acc (beta - demean) = {:.3f}".format(
            res["acc_obs"] - res["acc_naive"]
        ))

    print("\n===== FINAL SUMMARY =====")
    for r in all_results:
        print(r)
