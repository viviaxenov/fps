import os
import time
import numpy as np
import emcee
import scipy.optimize as sp

jax_log_prob = lambda x: -sp.rosen(np.moveaxis(x, -1, 0))

dim = 10

# --- tune covariance (stepsize) to target acceptance rate ---
n_step_tune = 700
n_ensemble_tune = 32
target_ar = 0.25
ar_cur = [1.0]

def _test_mcmc_ar(cov):
    if cov <= 0.0:
        return 1.0 - target_ar

    init = np.random.randn(n_ensemble_tune, dim)
    sampler = emcee.EnsembleSampler(
        n_ensemble_tune,
        dim,
        jax_log_prob,
        moves=[(emcee.moves.GaussianMove(cov=cov), 1.0)],
    )
    sampler.run_mcmc(init, n_step_tune, progress=False)
    ar = sampler.acceptance_fraction.mean()
    ar_cur[0] = ar
    return ar - target_ar

print("Tuning stepsize parameter in MCMC", flush=True)
tune_res = sp.root_scalar(_test_mcmc_ar, bracket=(0.0, 100.0), method="bisect", xtol=0.005)
mcmc_cov = float(tune_res.root)
print(f"Stepsize {mcmc_cov:.2e}; acceptance fraction {ar_cur[0]:.3f}", flush=True)

# --- reuse existing sample as init and write to a NEW h5 ---
fname_old = "/Home/optimier/berkowsky/Documents/fps/examples/datasets/10D_rosenbrock/sample.h5"
fname_new = fname_old.replace(".h5", f"_restart_{int(time.time())}.h5")

N_ensemble_mcmc = 400
N_mcmc = 2_000_000

old_backend = emcee.backends.HDFBackend(fname_old, read_only=True)
if old_backend.shape != (N_ensemble_mcmc, dim):
    raise ValueError(f"H5 shape {old_backend.shape} != {(N_ensemble_mcmc, dim)}")
init = old_backend.get_last_sample()
del old_backend

cov_scale = 1.0
mcmc_cov = cov_scale * mcmc_cov
print(f"Using restart stepsize (cov): {mcmc_cov:.2e}", flush=True)

os.makedirs(os.path.dirname(fname_new), exist_ok=True)
backend_new = emcee.backends.HDFBackend(fname_new)
backend_new.reset(N_ensemble_mcmc, dim)

sampler = emcee.EnsembleSampler(
    N_ensemble_mcmc,
    dim,
    jax_log_prob,
    moves=[(emcee.moves.GaussianMove(cov=mcmc_cov), 1.0)],
    backend=backend_new,
)

sampler.run_mcmc(init, N_mcmc, progress=True, store=True)

print("saved:", fname_new, flush=True)
print("mean acceptance:", float(np.mean(sampler.acceptance_fraction)), flush=True)

try:
    ac_times = sampler.get_autocorr_time(quiet=True)
    print("tau max:", float(np.max(ac_times)), flush=True)
except emcee.autocorr.AutocorrError as e:
    print("WARNING:", str(e), flush=True)
    print("tau max (estimate):", float(np.max(e.tau)), flush=True)
