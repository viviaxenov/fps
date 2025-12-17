"""Contains all the relevant functions for the geometric computaitons on the Stein manifold.

A density is represented by a sample

.. math::

    \\left\{x_i \\right\}_{i = \overline{1, N_{\\text{samples}}}},\\ x_i \\sim \\rho_* \\text{ i.i.d.}


"""
from typing import Callable, Union, List, Dict, Generator, Literal, Generator, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial

@jax.jit
def bandwidth_median(X: jnp.array) -> float:
    """Estimate a suitable bandwidth for the kernel using the median heuristic.

    Args:
        X: array of shape `(N_samples, dim)`, representing the sample

    Returns:
        float: the bandwidth    
    """
    N, d = X.shape
    X_diffs = X[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]
    idx = jnp.triu_indices(N, k=1)
    X_diffs = X_diffs[*idx, :]
    pairwise_dists = (X_diffs**2).sum(axis=-1)
    h = jnp.median(pairwise_dists)
    h = jnp.sqrt(0.5 * h / jnp.log(d + 1))

    return h


# @jax.jit
def norm_rkhs(x: jnp.ndarray, G: jnp.ndarray) -> float:
    """Evaluates the RKHS norm of the tangent vector, given the current Gram matrix.

    The tangent vector is represented in the basis

    .. math::
        
        \\left\\{k(\\cdot, x_i)e_j, \\nabla k(\cdot, x_i) \\right\\}_{ i = \\overline{1, N_{\\text{samples}}}, j = \\overline{1, d}}


    Args:
        x: array of shape `(N_samples, dim + 1)` with vector's coefficients

    Returns:
        The value of the vector's norm
    """
    rkhs_norm_sq = jnp.einsum("ij,ijkl,kl", x, G, x)
    return jnp.sqrt(jnp.maximum(rkhs_norm_sq, 0.0))


def norm_l2(v: jnp.ndarray) -> float:
    """Evaluates the (Monte-Carlo estimate of the) l2 norm of the tangent vector.

    The tangent vector is represented as

    .. math::

        v(x_i),\\ v \in L_2(\\rho_{\\text{cur}})

    Args:
        v: the array of shape `(N_samples, dim)` with the values of :math:`v` 

    Returns:
        The value of the vector's norm
    """
    norm_l2 = jnp.linalg.norm(v, axis=-1).mean()
    return norm_l2


def getOperatorSteinGradKL(log_density_target: Callable, stepsize: jnp.float64) -> Callable:
    """Given a funciton for the log density of the target distribution :math:`\\rho_*`, returns a function for the Stein gradient of the :math:`\operatorname{KL(\cdot | \\rho_*)}`

    Args:
        log_density_target: the log density of the target distribution (up to an additive constant).
        stepsize: scaling parameter for the gradient

    Returns:
        A function computing the gradient

    """
    grad_log_fn = jax.vmap(jax.grad(log_density_target))

    def steinGradKL(x: jnp.ndarray):
        N_samples, dim = x.shape
        sg = jnp.zeros((N_samples, dim + 1))
        sg = sg.at[:, :-1].set(grad_log_fn(x))

        sg = sg.at[:, -1].set(1.0)

        return -sg / N_samples * stepsize

    return steinGradKL


# @partial(jax.jit, static_argnums=-1)
def evalTangent(
    x_eval: jnp.ndarray, tangent: jnp.ndarray, x_basis: jnp.ndarray, kern: Callable
):
    # TODO: kern vectorized? how to broadcast?
    grad_of_kern = jax.grad(kern, argnums=1)
    vect_kern = jnp.vectorize(kern, signature="(i),(i)->()")
    vect_grad = jnp.vectorize(grad_of_kern, signature="(i),(i)->(i)")

    k = vect_kern(x_eval[:, jnp.newaxis, :], x_basis[jnp.newaxis, :, :])
    gk = vect_grad(x_eval[:, jnp.newaxis, :], x_basis[jnp.newaxis, :, :])

    comp_T0 = (k[:, :, jnp.newaxis] * tangent[jnp.newaxis, :, :-1]).sum(axis=1)
    comp_T1 = (gk * tangent[jnp.newaxis, :, -1, jnp.newaxis]).sum(axis=1)

    return comp_T0 + comp_T1


# to do: reimplement as matvec?
# @partial(jax.jit, static_argnums=-1)
def pairwiseScalarProductOfBasisVectors(x1: jnp.array, x2: jnp.array, kern: Callable):
    N, d = x1.shape
    M, d1 = x2.shape
    assert d1 == d

    grad_of_kern = jax.grad(kern, argnums=1)
    vect_kern = jnp.vectorize(kern, signature="(i),(i)->()")
    vect_grad = jnp.vectorize(grad_of_kern, signature="(i),(i)->(i)")
    jac_of_kern = jax.jacrev(grad_of_kern, argnums=0)

    mixed_deriv_term = jnp.vectorize(
        lambda _x1, _x2: jnp.trace(jac_of_kern(_x1, _x2)), signature="(i),(i)->()"
    )

    res_mat = jnp.zeros((N, d + 1, M, d + 1))

    k = vect_kern(x1[:, jnp.newaxis, :], x2[jnp.newaxis, :, :])
    for i in range(d):
        res_mat = res_mat.at[:, i, :, i].set(k)

    res_mat = res_mat.at[:, :-1, :, -1].set(
        vect_grad(x1[:, jnp.newaxis, :], x2[jnp.newaxis, :, :]).swapaxes(1, 2)
    )
    res_mat = res_mat.at[:, -1, :, :-1].set(
        vect_grad(x2[jnp.newaxis, :, :], x1[:, jnp.newaxis, :])
    )
    res_mat = res_mat.at[:, -1, :, -1].set(
        mixed_deriv_term(x1[:, jnp.newaxis, :], x2[jnp.newaxis, :, :])
    )

    return res_mat


# @jax.jit
def vectorTransport(
    x_cur: jnp.ndarray,
    G_cur: jnp.ndarray,
    x_prev: jnp.ndarray,
    tang_prev: jnp.ndarray,
    T_cur: jnp.ndarray,
    reg_proj: jnp.float64 = 1e-6,
):
    N, d = x_cur.shape
    M, d1 = x_prev.shape
    assert d1 == d
    if len(tang_prev.shape) == 2:
        tang_prev = tang_prev[:, :, jnp.newaxis]
    mk = tang_prev.shape[-1]

    # matTransition = pairwiseScalarProductOfBasisVectors(x_cur, x_prev)
    rhs = jnp.einsum("ijkl,klm->ijm", T_cur, tang_prev)

    tang_cur = jsp.sparse.linalg.cg(
        lambda _x: jnp.einsum("ijkl,klm->ijm", G_cur, _x) + reg_proj * _x,
        rhs,
        x0=rhs,
        tol=1e-5,
    )[0]

    return tang_cur



