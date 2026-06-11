"""
mempy — pMEM observation model for continuous neuroimaging data.

Core classes
------------
PMEMObsSimple      Direct-observation EM estimator (ROI / source space)
pMEM_Leadfield_v2  EEG/MEG sensor-space EM estimator with leadfield forward model

Quick example
-------------
    from mempy import PMEMObsSimple

    model = PMEMObsSimple(Y)          # Y: [T, N] continuous observations
    h, J, beta0, beta1, m = model.fit()
    # h, J  — pMEM energy parameters
    # beta0, beta1 — per-node baseline and gain
    # m — posterior mean activation  [T, N]

    from mempy import pMEM_Leadfield_v2
    lf_model = pMEM_Leadfield_v2(Y_sensor, L)   # L: [M, N] leadfield matrix
    lf_model.fit()
"""

from .pmem_obs import (
    PMEMObsSimple,
    simulate_pmem_basic,
    fit_pmem_pseudolikelihood,
    fit_gmm_threshold_pmem,
    exact_prior_moments,
    rel_err,
)
from .pmem_leadfield import pMEM_Leadfield_v2

__all__ = [
    "PMEMObsSimple",
    "pMEM_Leadfield_v2",
    "simulate_pmem_basic",
    "fit_pmem_pseudolikelihood",
    "fit_gmm_threshold_pmem",
    "exact_prior_moments",
    "rel_err",
]
