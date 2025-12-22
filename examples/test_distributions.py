import os

os.makedirs("./outputs", exist_ok=True)

import jax
import jax.numpy as jnp

from fps import *

import matplotlib.pyplot as plt

# Define the main parameters of the problem
N_samples = 50
dim = 3
N_iter = 100

# define the target distribution
m_targ = 0.0
p = 4.0
log_density_targ = lambda _x: -(((jnp.abs(_x - m_targ) ** p).sum()) ** (1.0 / p))

# Produce initial sample
key = jax.random.key(seed=5)
x0 = jax.random.normal(key, (N_samples, dim))

# Define a Gaussian RBF kernel with bandwith estimated from the median heuristic
bandwidth = bandwidth_median(x0)
kern = lambda _x1, _x2: jnp.exp(-((_x1 - _x2) ** 2).sum() / bandwidth**2)

# Define the operator as the Stein Gradient of KL divergence to the target distribution
stepsize_SVGD = 0.5
oper = getOperatorSteinGradKL(log_density_targ, -stepsize_SVGD)

solver = KernelRAMSolver(
    oper,
    kern,
    relaxation=3.00,
    l2_regularization=8e-3,
    history_len=6,
)


solver, (d_rkhs_kram, d_l2_kram, ) = solver.iterate(x0.copy(), max_iter=N_iter)
x_kRAM = solver._x_cur

solver_svgd = PicardSolver(
    oper,
    kern,
)
solver_svgd, (d_rkhs, d_l2, ) = solver_svgd.iterate(x0.copy(), max_iter=N_iter)

x_SVGD = solver_svgd._x_cur

label_RAM = f"$k$RAM, m={solver._m}"

fig, axs = plt.subplots(2, 1, sharex=True)

ax = axs[0]
ax.plot(d_rkhs_kram)
ax.plot(d_rkhs, color="r")
ax.set_ylabel(r"$\|r_t\|_{\mathcal{H}^d_k}$")

ax = axs[1]
ax.plot(d_l2_kram, label=label_RAM)
ax.plot(d_l2, label=f"SVGD (Picard), $h$ = {stepsize_SVGD:.2f}", color="r")
ax.set_ylabel(r"$\|\Delta x_t \|_{L_2(\rho_t)}$")

for ax in axs:
    ax.set_yscale("log")
    ax.grid()

axs[-1].legend()

fig.savefig("./outputs/test_kRAM_convergence.pdf")

fig, axs = plt.subplots(1, 1)
axs.scatter(*x_kRAM[:, :2].T, label=r"$x_\text{ kRAM}$")
axs.scatter(*x_SVGD[:, :2].T, label=r"$x_\text{ SVGD }$")
# axs.scatter(*sample_targ[:, :2].T, label=r"$x_{\infty}$")
axs.legend()

fig.savefig("./outputs/test_kRAM_scatter.pdf")
