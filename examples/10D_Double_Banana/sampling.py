

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



def jax_log_prob(x: jnp.ndarray) -> jnp.ndarray:
    r = jnp.sqrt(jnp.sum(x**2))
    term1 = -2.0 * (r - 3.0)**2
    t1 = -2.0 * (x[0] - 3.0)**2
    t2 = -2.0 * (x[0] + 3.0)**2
    term2 = jsp.special.logsumexp(jnp.array([t1, t2]))
    return term1 + term2 


log_prob_vec = jax.vmap(jax_log_prob, in_axes=0)

# Grid im (x0, x1)-Raum
x_min, x_max = -6.0, 6.0
y_min, y_max = -6.0, 6.0
n_grid = 200

xs = jnp.linspace(x_min, x_max, n_grid)
ys = jnp.linspace(y_min, y_max, n_grid)
X, Y = jnp.meshgrid(xs, ys, indexing="xy")

points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
logp = log_prob_vec(points).reshape(X.shape)

plt.figure(figsize=(6, 5))

# Nur Isolinien:
cs = plt.contour(X, Y, logp, levels=20)  # levels = Anzahl der Isolinien
plt.clabel(cs, inline=True, fontsize=8)  # optional: Werte an die Linien schreiben

plt.xlabel(r"$x_0$")
plt.ylabel(r"$x_1$")
plt.title(r"Isolinien von $\log \pi(x)$")
plt.axis("equal")
plt.grid(False)

plt.show()

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

print("Running burn-in phase", flush=True)
fname_h5 = "/Home/optimier/berkowsky/Documents/fps/examples/datasets/10D_p4/sample.h5"
dirname = os.path.dirname(fname_h5)
os.makedirs(dirname, exist_ok=True)

N_ensemble_mcmc = 400 # number of parallel chains
N_mcmc = 300000 # Number of iterations; should be large

init = np.random.randn(N_ensemble_mcmc, dim)
sampler = emcee.EnsembleSampler(
    N_ensemble_mcmc,
    dim,
    jax_log_prob,
    moves=[(emcee.moves.GaussianMove(cov=mcmc_cov), 1.0)], # move == method to generate proposals; can experiment with others => see documentation
    backend=emcee.backends.HDFBackend(fname_h5),
)
state = sampler.run_mcmc(
    init,
    N_mcmc,
    progress=True,
    store=True,
)
# run the chain; make large so that there is no warning that AC time is larger than N_mcmc/50
ac_times = sampler.get_autocorr_time()
t_ac = np.max(ac_times)
print(f"Autocorrelation time {t_ac:.2e}", flush=True)

# loading chain (use this snippet in test_kRAM.py 

reader = emcee.backends.HDFBackend(fname_h5, read_only=True)
# can make discard and n_thin larger if it takes too long to compute
ac_times = reader.get_autocorr_time(discard=2000, thin=2)
t_ac = np.max(ac_times)

t_ac_int = int(t_ac)
discard = int(50 * t_ac_int)

chain = reader.get_chain(discard=discard, thin=t_ac_int)
#chain = reader.get_chain(discard=50*t_ac, thin=t_ac) # numpy array os shape (N_samples, sample_size, dim)
#samples = chain.reshape(-1, dim)  
# each chain[i] is an i.i.d. sample from the distribution




