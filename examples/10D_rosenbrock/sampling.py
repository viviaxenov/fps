

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np
import emcee
import scipy.optimize as sp 
import os
import matplotlib.pyplot as plt
# implement jax_log_prob for each distribution 
# as a function of single x (vectorization is done in the solver)
# then pass to operatorSteinGradient 
import jax.scipy as jsp


jax_log_prob = lambda x: -sp.rosen(np.moveaxis(x, -1, 0))

# tune COV in the MH-MCMC

dim = 10
n_step_tune = 700
n_ensemble_tune = 32

ar_cur = [1.0]


def _test_mcmc_ar(cov):
    # run short MCMC chains with different covariance so that acceptance rate becomes close to 0.23
    if cov <= 0.0:
        return 1.0 - 0.23

    init = np.random.randn(n_ensemble_tune, dim)
    sampler = emcee.EnsembleSampler(
        n_ensemble_tune,
        dim,
        jax_log_prob,
        moves=[(emcee.moves.GaussianMove(cov=cov), 1.0)],
    )
    state = sampler.run_mcmc(init, n_step_tune, progress=False)
    ar = sampler.acceptance_fraction.mean()
    ar_cur[0] = ar

    return ar - 0.23  # rule of thumb optimal AR


print(
    "Tuning stepsize parameter in MCMC",
    flush=True,
)
tune_res = sp.root_scalar(
    _test_mcmc_ar, bracket=(0.0, 100.0), method="bisect", xtol=0.005
)
mcmc_cov = tune_res.root
print(f"Stepsize {mcmc_cov:.2e}; acceptance fraction {ar_cur[0]:.3f}", flush=True)


# Run mcmc to estimate AC time and (hopefully) skip the burn-in phase
# Run mcmc to estimate AC time ...

# Run mcmc to estimate AC time and (hopefully) skip the burn-in phase

print("Running burn-in phase", flush=True)
fname_h5 = "/Home/optimier/berkowsky/Documents/fps/examples/datasets/10D_rosenbrock/sample.h5"
fname_old = fname_h5
fname_new = fname_h5.replace(".h5", "_restart.h5")

N_ensemble_mcmc = 400
N_mcmc = 2000000

old_backend = emcee.backends.HDFBackend(fname_old, read_only=True)
if old_backend.shape != (N_ensemble_mcmc, dim):
    raise ValueError(f"H5 shape {old_backend.shape} != {(N_ensemble_mcmc, dim)}")
init = old_backend.get_last_sample()
del old_backend

mcmc_cov = 0.5 * mcmc_cov
print(f"Using smaller stepsize (cov): {mcmc_cov:.2e}", flush=True)

backend_new = emcee.backends.HDFBackend(fname_new)
backend_new.reset(N_ensemble_mcmc, dim)

sampler = emcee.EnsembleSampler(
    N_ensemble_mcmc,
    dim,
    jax_log_prob,
    moves=[(emcee.moves.GaussianMove(cov=mcmc_cov), 1.0)],
    backend=backend_new,
)
#state = sampler.run_mcmc(
#   init,
#    N_mcmc,
#    progress=True,
##    store=True,
#)

sampler.run_mcmc(init, N_mcmc, progress=True, store=True)

# tau can raise AutocorrError -> make it non-fatal
try:
    ac_times = sampler.get_autocorr_time(quiet=True)
    t_ac = np.max(ac_times)
    print(f"Autocorrelation time {t_ac:.2e}", flush=True)
except emcee.autocorr.AutocorrError as e:
    t_ac = float(np.max(e.tau))
    print("WARNING:", e, flush=True)
    print(f"Autocorrelation time (estimate) {t_ac:.2e}", flush=True)

reader = emcee.backends.HDFBackend(fname_new, read_only=True)
try:
    ac_times = reader.get_autocorr_time(discard=2000, thin=2, quiet=True)
    t_ac = np.max(ac_times)
except emcee.autocorr.AutocorrError as e:
    t_ac = float(np.max(e.tau))
    print("WARNING (reader):", e, flush=True)

t_ac_int = int(t_ac) if np.isfinite(t_ac) and t_ac > 0 else 1
discard = int(50 * t_ac_int)

chain = reader.get_chain(discard=discard, thin=t_ac_int)

# run the chain; make large so that there is no warning that AC time is larger than N_mcmc/50
#ac_times = sampler.get_autocorr_time()
#t_ac = np.max(ac_times)
#print(f"Autocorrelation time {t_ac:.2e}", flush=True)
#
# loading chain (use this snippet in test_kRAM.py 

#reader = emcee.backends.HDFBackend(fname_h5, read_only=True)
# can make discard and n_thin larger if it takes too long to compute
#ac_times = reader.get_autocorr_time(discard=2000, thin=2)
#t_ac = np.max(ac_times)

#t_ac_int = int(t_ac)
#discard = int(50 * t_ac_int)

#chain = reader.get_chain(discard=discard, thin=t_ac_int)



