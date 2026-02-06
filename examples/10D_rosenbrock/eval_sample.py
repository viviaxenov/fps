import numpy as np
import emcee
import h5py

fname = "/Home/optimier/berkowsky/Documents/fps/examples/datasets/10D_rosenbrock/sample.h5"
discard, thin = 0,1 

reader = emcee.backends.HDFBackend(fname, read_only=True)

# total iterations + shape
n_iter = reader.iteration
n_walkers, ndim = reader.shape
print("iterations (per walker):", n_iter)
print("n_walkers, ndim:", (n_walkers, ndim))

# acceptance rate from H5 ("accepted" counts)
with h5py.File(fname, "r") as f:
    g = f["mcmc"]  # default group name in emcee HDFBackend
    accepted = np.array(g["accepted"])          # shape (n_walkers,)
    acc_frac = accepted / max(1, n_iter)
    print("acceptance fraction mean/min/max:",
          acc_frac.mean(), acc_frac.min(), acc_frac.max())

# empirical covariance of (post-processed) samples
samples = reader.get_chain(discard=discard, thin=thin, flat=True)  # (N, ndim)
print("N used samples:", samples.shape[0])

# t_ac (computed, not stored) — use thin>1 to make it faster
tau = reader.get_autocorr_time(discard=0, thin=1, quiet=True)
tau = np.array(tau, dtype=float)
print("tau per dim thinned :", tau)
print("tau max thinned:", float(tau.max()))

