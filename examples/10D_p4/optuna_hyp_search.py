import optuna
import numpy as np
import jax
import jax.numpy as jnp
import emcee
import optuna.visualization as vis
from fps.kernel_ram_solver import *
from fps.picard_solver import *
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence

H5_PATH = "./sample.h5"
MCMC_Samples = 3000

reader = emcee.backends.HDFBackend(H5_PATH, read_only=True)
chain = reader.get_chain(discard=0, thin=1, flat=False)
sample_targ = jnp.array(chain.reshape(-1, chain.shape[-1])[-MCMC_Samples:])

def jax_log_prob(x: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[0]
    m = (-1.0) ** (jnp.arange(d) + 1)
    norm4 = jnp.sum(jnp.abs(x - m) ** 4) ** 0.25
    return -norm4

N_samples = 3000
dim = 10
N_iter = 10000

key = jax.random.PRNGKey(5)
x0 = jax.random.normal(key, (N_samples, dim))

bw0 = float(bandwidth_median(x0))

def solver(h: float, bw: float) -> jnp.ndarray:
    kern = lambda _x1, _x2: jnp.exp(-((_x1 - _x2) ** 2).sum() / (bw**2))
    oper = getOperatorSteinGradKL(jax_log_prob, -float(h))

    solver_svgd = PicardSolver(oper, kern)
    solver_svgd, _ = solver_svgd.iterate(x0.copy(), max_iter=N_iter)
    return solver_svgd._x_cur  # x_SVGD

def sinkhorn(X: jnp.ndarray) -> float:
    div, _ = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        X,
        sample_targ,
        epsilon=5e-2,
        solve_kwargs=dict(threshold=5e-2, max_iterations=30000, inner_iterations=50),
    )
    return float(div)


def objective(trial):
    h = trial.suggest_float("h", 1e-2, 1e2, log=True)
    bw_mult = trial.suggest_float("bw_mult", 0.1, 50.0, log=True)
    bw = bw0 * bw_mult

    val = sinkhorn(solver(h, bw))
    if not np.isfinite(val):
        return float("inf")
    return val

study = optuna.create_study(
    direction="minimize",
    study_name="svgd_sinkhorn",
    storage="sqlite:///optuna.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=15)


print("best value:", study.best_value)
print("best params:", study.best_params)
print("best bandwidth:", bw0 * study.best_params["bw_mult"])

# 2D contour (h vs bw_mult)
fig = vis.plot_contour(study, params=["h", "bw_mult"])
fig.write_html("optuna_contour_h_bw.html")
print("saved: optuna_contour_h_bw.html")

# optional: auch history
fig2 = vis.plot_optimization_history(study)
fig2.write_html("optuna_history.html")
print("saved: optuna_history.html")

