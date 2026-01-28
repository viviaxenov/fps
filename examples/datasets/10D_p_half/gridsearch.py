import os
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import emcee
import matplotlib.pyplot as plt

from fps.kernel_ram_solver import *
from fps.picard_solver import *
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence


H5_PATH = "./sample.h5"
out_h5  = "./svgd_gridsearch_h.h5"
os.makedirs(os.path.dirname(out_h5), exist_ok=True)

MCMC_Samples = 4000
reader = emcee.backends.HDFBackend(H5_PATH, read_only=True)
chain = reader.get_chain(discard=0, thin=1, flat=False)
sample_targ = jnp.array(chain.reshape(-1, chain.shape[-1])[-MCMC_Samples:])


def jax_log_prob(x: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[0]
    m = (-1.0) ** (jnp.arange(d) + 1)
    norm4 = jnp.sum(jnp.abs(x - m) ** 0.5) ** 2
    return -norm4


N_samples = 3000
dim = 10
N_iter = 1000

key = jax.random.PRNGKey(5)
x0 = jax.random.normal(key, (N_samples, dim))

bandwidth = float(bandwidth_median(x0))
def kern(_x1, _x2):
    return jnp.exp(-((_x1 - _x2) ** 2).sum() / (bandwidth**2))

def sinkhorn_metric(X: jnp.ndarray) -> jnp.ndarray:
    div, _ = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        X,
        sample_targ,
        epsilon = 5e-2,
        solve_kwargs=dict(threshold = 5e-2, max_iterations = 30000, inner_iterations=50),
    )
    return div


h_min, h_max = 0.01, 3.0
n_h = 20
h_list = np.linspace(h_min, h_max, n_h).tolist()
h_list


def h_to_group_name(h: float) -> str:
    return f"h_{h:.10g}".replace(".", "p").replace("-", "m")

extra_h = np.linspace(0.01, 0.1, 4).tolist()
extra_h = sorted(set(extra_h + [h_list[1]]))

redo_h = extra_h
redo_groups = {h_to_group_name(h) for h in redo_h}

h_list = sorted(set(h_list + extra_h))

with h5py.File(out_h5, "a") as f:
    for h in h_list:
        print(f"=== h={h} ===", flush=True)

        gname = h_to_group_name(h)

        if gname in f and gname not in redo_groups:
            print("skip (already exists):", gname, flush=True)
            continue

        if gname in f and gname in redo_groups:
            print("redo (overwrite):", gname, flush=True)
            del f[gname]
            f.flush()

        oper = getOperatorSteinGradKL(jax_log_prob, -float(h))
        solver_svgd = PicardSolver(oper, kern, metrics=(sinkhorn_metric,))
        solver_svgd, (d_rkhs, d_l2, d_sinkhorn) = solver_svgd.iterate(x0.copy(), max_iter=N_iter)
        x_SVGD = solver_svgd._x_cur

        score = float(np.sum(np.asarray(d_sinkhorn, dtype=float).reshape(-1)))

        g = f.create_group(gname)
        g.attrs["h"] = float(h)
        g.attrs["score_sinkhorn_sum"] = score

        g.create_dataset("d_rkhs", data=np.asarray(d_rkhs), compression="gzip")
        g.create_dataset("d_l2", data=np.asarray(d_l2), compression="gzip")
        g.create_dataset("d_sinkhorn", data=np.asarray(d_sinkhorn), compression="gzip")
        g.create_dataset("x_SVGD_last", data=np.asarray(x_SVGD), compression="gzip")

        f.flush()
        print("score (sum d_sinkhorn):", score, flush=True)


hs = []
scores = []
hs2 = []
curves = []

with h5py.File(out_h5, "r") as f:
    for gname in f.keys():
        if not gname.startswith("h_"):
            continue
        if "d_sinkhorn" not in f[gname]:
            continue

        h = float(f[gname].attrs["h"])
        s = np.array(f[gname]["d_sinkhorn"]).reshape(-1)

        hs.append(h)
        scores.append(float(np.sum(s)))

        hs2.append(h)
        curves.append(s)

order = np.argsort(hs)
hs_sorted = np.array(hs)[order]
scores_sorted = np.array(scores)[order]

plt.figure()
plt.plot(hs_sorted, scores_sorted, marker="o")
plt.xlabel("h")
plt.ylabel("score = sum(d_sinkhorn)")
plt.tight_layout()
out_png = out_h5.replace(".h5", "_score_vs_h.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print("saved plot:", out_png)

order2 = np.argsort(hs2)
hs2 = np.array(hs2)[order2]
curves = [curves[i] for i in order2]

plt.figure()
for h, s in zip(hs2, curves):
    t = np.arange(len(s))
    plt.plot(t, s, label=f"h={h:g}")

plt.xlabel("iteration")
plt.ylabel("d_sinkhorn")
plt.legend()
plt.tight_layout()
out_png2 = out_h5.replace(".h5", "_all_sinkhorn_curves.png")
plt.savefig(out_png2, dpi=300, bbox_inches="tight")
plt.close()
print("saved plot:", out_png2)

print("saved h5:", out_h5)
