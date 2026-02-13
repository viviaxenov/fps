import os
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import emcee
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fps.kernel_ram_solver import *
from fps.picard_solver import *
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
import corner



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
    d = x.shape[0]
    m = (-1.0) ** (jnp.arange(d) + 1)
    norm4 = jnp.sum(jnp.abs(x - m) ** 4) ** 0.25
    return -norm4


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
h_min, h_max = 20.0, 50
n_h = 75
h_list = np.linspace(h_min, h_max, n_h).tolist()
h_list = []
def h_to_group_name(h: float) -> str:
    return f"h_{h:.10g}".replace(".", "p").replace("-", "m")

#extra_h = np.linspace(0.01, 0.1, 4).tolist()
#extra_h = sorted(set(extra_h + [h_list[1]]))
extra_h = []
redo_h = []
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
        print("score (init + sparse sum d_sinkhorn):", score, flush=True)


# -----------------------------
# Read results + plots (score + last + best + curves)
# -----------------------------
import matplotlib.ticker as mticker

eps = 1e-6  # for shift-log safety

def safe_argmin(a: np.ndarray):
    a = np.asarray(a, float)
    m = np.isfinite(a)
    if not np.any(m):
        return None
    tmp = np.where(m, a, np.inf)
    return int(np.argmin(tmp))

def shift_for_log(y, y0, eps=1e-6):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan, dtype=float)
    m = np.isfinite(y)
    out[m] = np.maximum(y[m] - y0 + eps, eps)
    return out

def curve_shift(s, y0, eps=1e-6):
    s = np.asarray(s, float)
    out = np.full_like(s, np.nan, dtype=float)
    m = np.isfinite(s)
    out[m] = np.maximum(s[m] - y0 + eps, eps)
    return out

def apply_shiftlog_axis_with_original_labels(ax, y0, eps=1e-6):
    """
    Axis is log in shifted coordinates (y - y0 + eps),
    but tick labels are shown in ORIGINAL y-units.
    """
    ax.set_yscale("log")

    def fmt(v, pos):
        # v is in shifted units; map back to original y
        y = y0 + v - eps
        return f"{y:.4g}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))

# -----------------------------
# Read results from out_h5
# -----------------------------
# Read results from out_h5
hs = []
lasts = []
bests = []
curves = []
gnames = []
samples = []  

with h5py.File(out_h5, "r") as f:
    for gname in f.keys():
        if not gname.startswith("h_"):
            continue
        g = f[gname]
        if "d_sinkhorn" not in g:
            continue
        if "x_SVGD_last" not in g:   
            continue

        h = float(g.attrs.get("h", np.nan))
        s = np.array(g["d_sinkhorn"]).reshape(-1)

        s0 = g.attrs.get("sinkhorn_init", np.nan)
        s = np.concatenate([[s0], s])

        finite = s[np.isfinite(s)]
        last = float(finite[-1]) if finite.size else np.nan
        best = float(np.min(finite)) if finite.size else np.nan

        hs.append(h)
        curves.append(s)
        lasts.append(last)
        bests.append(best)
        gnames.append(gname)
        samples.append(np.array(g["x_SVGD_last"])) 
hs = np.asarray(hs, float)
lasts = np.asarray(lasts, float)
bests = np.asarray(bests, float)

order = np.argsort(hs)
hs = hs[order]
lasts = lasts[order]
bests = bests[order]
curves = [curves[i] for i in order]
gnames = [gnames[i] for i in order]
samples = [samples[i] for i in order]


# =============================
# 1) Plot: 2 selected curves
#    - curve with minimal last
#    - curve with minimal best/min
# =============================
i_last = safe_argmin(lasts)
i_best = safe_argmin(bests)

# y0 = global min among the plotted (selected) curves
sel = [i for i in [i_last, i_best] if i is not None]
vals = []
for i in sel:
    v = curves[i][np.isfinite(curves[i])]
    if v.size:
        vals.append(v)
y0_sel = float(np.min(np.concatenate(vals))) if vals else 0.0
ymax_sel = float(np.max(np.concatenate(vals))) if vals else 1.0

fig, ax = plt.subplots(figsize=(12, 5))

if i_last is not None:
    s = curves[i_last]
    t = np.arange(len(s))
    y = curve_shift(s, y0_sel, eps)         # plot in shifted coords
    m = np.isfinite(y)
    ax.plot(t[m], y[m], "-", marker="o", markersize=4, linewidth=2,
            label=f"min last: h={hs[i_last]:g}, last={lasts[i_last]:.4g}")

if i_best is not None:
    s = curves[i_best]
    t = np.arange(len(s))
    y = curve_shift(s, y0_sel, eps)
    m = np.isfinite(y)
    ax.plot(t[m], y[m], "--", marker="s", markersize=4, linewidth=2,
            label=f"min best: h={hs[i_best]:g}, min={bests[i_best]:.4g}")

apply_shiftlog_axis_with_original_labels(ax, y0_sel, eps)
ax.set_ylim(eps, (ymax_sel - y0_sel + eps))    # starts at min (mapped), not 0
ax.set_xlabel("iteration")
ax.set_ylabel("d_sinkhorn (shift-log scaling, original labels)")
ax.legend()
fig.tight_layout()

out1 = out_h5.replace(".h5", "_selected_minlast_minbest_curves_shiftlog_labels.png")
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved plot:", out1)

# =============================
# 2) Plot: all sinkdiv curves (shift-log, original labels)
# =============================
vals = []
for s in curves:
    v = s[np.isfinite(s)]
    if v.size:
        vals.append(v)

y0_all = float(np.min(np.concatenate(vals))) if vals else 0.0
ymax_all = float(np.max(np.concatenate(vals))) if vals else 1.0

fig, ax = plt.subplots(figsize=(12, 7))

# show only ~12 legend entries
step = max(1, len(curves) // 12)

for idx, (h, s) in enumerate(zip(hs, curves)):
    t = np.arange(len(s))
    y = curve_shift(s, y0_all, eps)
    m = np.isfinite(y)
    label = f"h={h:g}" if (idx % step == 0) else None
    ax.plot(t[m], y[m], linewidth=1, label=label)

apply_shiftlog_axis_with_original_labels(ax, y0_all, eps)
ax.set_ylim(eps, (ymax_all - y0_all + eps))
ax.set_xlabel("iteration")
ax.set_ylabel("d_sinkhorn (shift-log scaling, original labels)")

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=4,
    fontsize=8,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.92])
out2 = out_h5.replace(".h5", "_all_sinkhorn_curves_shiftlog_labels.png")
fig.savefig(out2, dpi=300)
plt.close(fig)
print("saved plot:", out2)

# =============================
# 3) Plot: best(min) + last over h (shift-log, original labels)
# =============================
finite_last = lasts[np.isfinite(lasts)]
finite_best = bests[np.isfinite(bests)]
if finite_last.size or finite_best.size:
    y0_h = float(np.min(np.concatenate([finite_last, finite_best])))
    ymax_h = float(np.max(np.concatenate([finite_last, finite_best])))
else:
    y0_h, ymax_h = 0.0, 1.0

lasts_shift = shift_for_log(lasts, y0_h, eps)
bests_shift = shift_for_log(bests, y0_h, eps)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hs, lasts_shift, "-",  marker="o", markersize=5, linewidth=2, label="last sinkdiv")
ax.plot(hs, bests_shift, "--", marker="s", markersize=5, linewidth=2, label="best sinkdiv (min)")

apply_shiftlog_axis_with_original_labels(ax, y0_h, eps)
ax.set_ylim(eps, (ymax_h - y0_h + eps))
ax.set_xlabel("h")
ax.set_ylabel("sinkdiv (shift-log scaling, original labels)")
ax.legend()
fig.tight_layout()

out3 = out_h5.replace(".h5", "_last_best_vs_h_shiftlog_labels.png")
fig.savefig(out3, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved plot:", out3)

i_best = int(np.nanargmin(lasts))
data = samples[i_best]
best_gname = gnames[i_best]
best_h = hs[i_best]
best_last = lasts[i_best]

# ggf. auf 4 dims schneiden, falls data mehr Dimensionen hat:
# data = data[:, [0,1,2,3]]

D = data.shape[1]
labels = [fr"$x_{{{i}}}$" for i in range(D)]

figure = corner.corner(
    data,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)

figure.suptitle(
    f"Best run (min sink_last) | {best_gname} | h={best_h:g} | sink_last={best_last:.4g}",
    y=1.02
)

out_png = out_h5.replace(".h5", f"_corner_minlast.png")
figure.savefig(out_png, dpi=250, bbox_inches="tight")
plt.close(figure)
print("saved:", out_png)
