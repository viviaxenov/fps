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


# -----------------------------
# IO paths
# -----------------------------
H5_PATH = "./sample.h5"
out_h5  = "./svgd_gridsearch_h.h5"
os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)

# -----------------------------
# Load target samples (MCMC)
# -----------------------------
MCMC_Samples = 4000
reader = emcee.backends.HDFBackend(H5_PATH, read_only=True)
chain = reader.get_chain(discard=0, thin=1, flat=False)
sample_targ = jnp.array(chain.reshape(-1, chain.shape[-1])[-MCMC_Samples:])

# -----------------------------
# Target log prob
# -----------------------------
def jax_log_prob(x: jnp.ndarray) -> jnp.ndarray:
    r = jnp.sqrt(jnp.sum(x**2))
    term1 = -2.0 * (r - 3.0)**2
    t1 = -2.0 * (x[0] - 3.0)**2
    t2 = -2.0 * (x[0] + 3.0)**2
    term2 = jsp.special.logsumexp(jnp.array([t1, t2]))
    return term1 + term2 


# -----------------------------
# Setup
# -----------------------------
N_samples = 3000
dim = 10
N_iter = 10000

# Sinkhorn nur jede n-te Iteration auswerten
sinkhorn_every = 100

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
        epsilon=5e-2,
        solve_kwargs=dict(threshold=5e-2, max_iterations=30000, inner_iterations=50),
    )
    return div


# -----------------------------
# h grid
# -----------------------------
h_min, h_max = 0.1, 12
n_h = 50
h_list = np.linspace(h_min, h_max, n_h).tolist()
#h_list = []
def h_to_group_name(h: float) -> str:
    return f"h_{h:.10g}".replace(".", "p").replace("-", "m")

#extra_h = np.linspace(0.01, 0.1, 4).tolist()
#extra_h = sorted(set(extra_h + [h_list[1]]))
extra_h = []
redo_h = extra_h
redo_groups = {h_to_group_name(h) for h in redo_h}

h_list = sorted(set(h_list + extra_h))


# -----------------------------
# Run gridsearch
# -----------------------------
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

        # WICHTIG: sinkhorn_metric NICHT in metrics, sonst wird es jede Iteration gerechnet
        solver_svgd = PicardSolver(oper, kern, metrics=())

        x = x0.copy()

        # >>> NEU: initiale Sinkdiv (Iteration 0) einmal berechnen und speichern
        sink0 = float(sinkhorn_metric(x))

        d_rkhs_chunks = []
        d_l2_chunks   = []

        # Länge N_iter, nur alle sinkhorn_every Schritte gefüllt, sonst NaN
        d_sinkhorn = np.full((N_iter,), np.nan, dtype=np.float64)

        it_done = 0
        while it_done < N_iter:
            n = min(sinkhorn_every, N_iter - it_done)

            solver_svgd, outs = solver_svgd.iterate(x, max_iter=n)

            # outs enthält nur (d_rkhs, d_l2), weil metrics=() ist
            d_rkhs_blk, d_l2_blk = outs

            d_rkhs_chunks.append(np.asarray(d_rkhs_blk))
            d_l2_chunks.append(np.asarray(d_l2_blk))

            x = solver_svgd._x_cur
            it_done += n

            # Sinkhorn nur am Blockende (100,200,... bzw. am Ende)
            if (it_done % sinkhorn_every == 0) or (it_done == N_iter):
                d_sinkhorn[it_done - 1] = float(sinkhorn_metric(x))

        d_rkhs = np.concatenate(d_rkhs_chunks, axis=0)
        d_l2   = np.concatenate(d_l2_chunks, axis=0)
        x_SVGD = x

        # Score inkl. initialer Sinkdiv + sparse Werte
        score = float(sink0 + np.nansum(d_sinkhorn))
        finite = d_sinkhorn[np.isfinite(d_sinkhorn)]
        sink_last = float(finite[-1]) if finite.size > 0 else float("nan")
        sink_best = float(np.min(finite)) if finite.size > 0 else float("nan")

        print(f"sinkhorn init={sink0:.6g}  last={sink_last:.6g}  best={sink_best:.6g}  score={score:.6g}", flush=True)
        g = f.create_group(gname)
        g.attrs["h"] = float(h)
        g.attrs["sinkhorn_every"] = int(sinkhorn_every)
        g.attrs["sinkhorn_init"] = float(sink0)
        g.attrs["score_sinkhorn_sum"] = score

        g.create_dataset("d_rkhs", data=np.asarray(d_rkhs), compression="gzip")
        g.create_dataset("d_l2", data=np.asarray(d_l2), compression="gzip")
        g.create_dataset("d_sinkhorn", data=np.asarray(d_sinkhorn), compression="gzip")
        g.create_dataset("x_SVGD_last", data=np.asarray(x_SVGD), compression="gzip")

        f.flush()
        #print("score (init + sparse sum d_sinkhorn):", score, flush=True)


# -----------------------------
# Read results + plots (score + last + best + curves)
# -----------------------------
hs = []
scores = []
lasts = []
bests = []

hs2 = []
curves = []

with h5py.File(out_h5, "r") as f:
    for gname in f.keys():
        if not gname.startswith("h_"):
            continue
        if "d_sinkhorn" not in f[gname]:
            continue

        h = float(f[gname].attrs["h"])

        # sparse sinkdiv (NaNs between eval steps)
        s = np.array(f[gname]["d_sinkhorn"]).reshape(-1)

        # prepend initial sinkdiv at iteration 0 (stored above)
        s0 = f[gname].attrs.get("sinkhorn_init", np.nan)
        s = np.concatenate([[s0], s])   # length = N_iter + 1

        hs.append(h)
        scores.append(float(np.nansum(s)))

        finite = s[np.isfinite(s)]
        if finite.size == 0:
            lasts.append(np.nan)
            bests.append(np.nan)
        else:
            lasts.append(float(finite[-1]))
            bests.append(float(np.min(finite)))

        hs2.append(h)
        curves.append(s)

# sort by h
order = np.argsort(hs)
hs_sorted     = np.array(hs)[order]
scores_sorted = np.array(scores)[order]
lasts_sorted  = np.array(lasts)[order]
bests_sorted  = np.array(bests)[order]

# Plot: score vs h
plt.figure()
plt.plot(hs_sorted, scores_sorted, marker="o")
plt.xlabel("h")
plt.ylabel("score = sum(d_sinkhorn) (init + only every n; ignore NaNs)")
plt.tight_layout()
out_png = out_h5.replace(".h5", "_score_vs_h.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print("saved plot:", out_png)

# Optional: last + best together
plt.figure()
plt.plot(
    hs_sorted, lasts_sorted,
    linestyle="-", marker="o", markersize=5, linewidth=2,
    label="last sinkdiv"
)
plt.plot(
    hs_sorted, bests_sorted,
    linestyle="--", marker="s", markersize=5, linewidth=2,
    label="best sinkdiv (min)"
)
plt.xlabel("h")
plt.ylabel("sinkdiv")
plt.legend()
plt.tight_layout()
out_lb = out_h5.replace(".h5", "_last_best_sinkdiv_vs_h.png")
plt.savefig(out_lb, dpi=300, bbox_inches="tight")
plt.close()
print("saved plot:", out_lb)


# Plot: sparse sinkhorn curves (only finite points, incl. init at t=0)
order2 = np.argsort(hs2)
hs2 = np.array(hs2)[order2]
curves = [curves[i] for i in order2]

fig, ax = plt.subplots(figsize=(12, 5))
for h, s in zip(hs2, curves):
    t = np.arange(len(s))   # includes t=0 init point
    mask = np.isfinite(s)
    ax.plot(t[mask], s[mask], marker="o", markersize=3, linewidth=1, label=f"h={h:g}")

ax.set_xlabel("iteration")
ax.set_ylabel("d_sinkhorn (sparse + init)")

# Legend kompakt machen (mehrere Spalten, kleine Schrift, oben)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.25),
    ncol=4,          # ggf. 5-8 je nach Anzahl Kurven
    fontsize=7,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.9])  # Platz für Legende oben lassen

out_png2 = out_h5.replace(".h5", "_all_sinkhorn_curves.png")
fig.savefig(out_png2, dpi=300)         # <<< kein bbox_inches="tight"
plt.close(fig)
print("saved plot:", out_png2)

print("saved h5:", out_h5)

# --- NEW: plot 3 selected curves ---
# 1) curve with lowest "last sinkdiv"
# 2) curve with lowest "best/min sinkdiv"
# 3) curve with lowest "score"
# (Falls du "best last" als größtes last meinst -> np.argmax statt np.argmin.)

scores_arr = np.asarray(scores, dtype=float)
lasts_arr  = np.asarray(lasts, dtype=float)
bests_arr  = np.asarray(bests, dtype=float)

def safe_argmin(a):
    m = np.isfinite(a)
    if not np.any(m):
        return None
    tmp = np.where(m, a, np.inf)
    return int(np.argmin(tmp))

i_last  = safe_argmin(lasts_arr)
i_best  = safe_argmin(bests_arr)
i_score = safe_argmin(scores_arr)

fig, ax = plt.subplots(figsize=(10, 5))

def plot_curve(i, label, linestyle, marker):
    if i is None:
        return
    s = curves[i]
    t = np.arange(len(s))
    mask = np.isfinite(s)
    ax.plot(
        t[mask], s[mask],
        linestyle=linestyle, marker=marker,
        markersize=4, linewidth=2,
        label=label
    )

if i_last is not None:
    plot_curve(
        i_last,
        f"best last: h={hs2[i_last]:g}, last={lasts_arr[i_last]:.4g}",
        "-", "o"
    )

if i_best is not None:
    plot_curve(
        i_best,
        f"best min:  h={hs2[i_best]:g}, min={bests_arr[i_best]:.4g}",
        "--", "s"
    )

if i_score is not None:
    plot_curve(
        i_score,
        f"best score: h={hs2[i_score]:g}, score={scores_arr[i_score]:.4g}",
        ":", "D"
    )

ax.set_xlabel("iteration")
ax.set_ylabel("d_sinkhorn (sparse + init)")
ax.legend()
fig.tight_layout()

out_sel = out_h5.replace(".h5", "_selected_3_sinkhorn_curves.png")
fig.savefig(out_sel, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved plot:", out_sel)
