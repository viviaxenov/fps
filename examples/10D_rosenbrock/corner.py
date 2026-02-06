import os
import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt

fname = "./sample_restart_1769524224.h5"
discard, thin = 0, 1

out_dir = "./"
os.makedirs(out_dir, exist_ok=True)

r = emcee.backends.HDFBackend(fname, read_only=True)
#print("Nsteps", r.iteration)
#print("tau", r.get_autocorr_time(discard=discard, thin=1))
samples = r.get_chain(discard=2000, thin=50, flat=True)  # (N, dim)

ndim = samples.shape[1]

max_points = 100_000
if samples.shape[0] > max_points:
    idx = np.random.choice(samples.shape[0], size=max_points, replace=False)
    samples = samples[idx]

labels = [rf"$x_{{{i}}}$" for i in range(ndim)]

fig = corner.corner(
    samples,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)

out = "corner_10D.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)

print("saved:", os.path.abspath(out))








