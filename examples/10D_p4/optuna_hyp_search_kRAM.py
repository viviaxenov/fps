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

study = optuna.load_study(study_name="svgd_sinkhorn", storage="sqlite:///optuna.db")
best_h  = study.best_params["h"]
best_bw = study.best_params["bw_mult"]
bw = bw0*best_bw
def solver(relax: float, reg: float) -> jnp.ndarray:
    kern = lambda _x1, _x2: jnp.exp(-((_x1 - _x2) ** 2).sum() / (bw**2))
    oper = getOperatorSteinGradKL(jax_log_prob, -float(best_h))

    solver = KernelRAMSolver(
        oper,
        kern,
        relaxation= relax,
        l2_regularization= reg,
        history_len=6,
        )
    solver, _ = solver.iterate(x0.copy(), max_iter=N_iter)
    return solver._x_cur

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
    relax = trial.suggest_float("relax", 0.1, 2.0)          # oder 0.05..1.0
    reg   = trial.suggest_float("reg", 1e-10, 1e0, log=True) # reg > 0

    val = sinkhorn(solver(relax, reg))
    if not np.isfinite(val):
        return float("inf")
    return val

study = optuna.create_study(
    direction="minimize",
    study_name="kram_sinkhorn",
    storage="sqlite:///optuna_kRAM.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=15)


print("best value:", study.best_value)
print("best params:", study.best_params)

# 2D contour (h vs bw_mult)
fig = vis.plot_contour(study, params=["h", "bw_mult"])
fig.write_html("optuna_contour_kram.html")
print("saved: optuna_contour_kram.html")

# optional: auch history
fig2 = vis.plot_optimization_history(study)
fig2.write_html("optuna_history_kram.html")
print("saved: optuna_history_kram.html")

