from typing import Callable, Union, List, Dict, Generator, Literal, Generator, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial
from .stein_manifold import *
from ._utility import _as_generator


class PicardSolver:
    def __init__(
        self,
        operator: Callable,
        kernel: Callable,
        metrics: Tuple[Callable] = None,
    ):
        self._operator = operator

        self._kernel = kernel

        self._x_cur, self._v_cur = (None,) * 2
        self._metrics = metrics

    def _get_update(self, ):
        r = self._operator(self._x_cur)
        G = pairwiseScalarProductOfBasisVectors(self._x_cur, self._x_cur, self._kernel)
        v = evalTangent(self._x_cur, r, self._x_cur, self._kernel)
        residual_rkhs = norm_rkhs(r, G)
        dx_l2 = norm_l2(v)

        return v, residual_rkhs, dx_l2


    # TODO: do I really a separate initialize here?
    def _initialize_iteration(self, x0: jnp.ndarray):
        self._x_cur = x0.copy()  # x_k
        v, residual_rkhs, dx_l2 = self._get_update()
        self._v_cur = v
        metric_vals = (residual_rkhs, dx_l2)

        if self._metrics is not None:
            metric_vals += tuple(m(self._x_cur) for m in self._metrics)

        return self, metric_vals

    def _step(
        self,
    ):
        self._x_cur += self._v_cur
        v, residual_rkhs, dx_l2 = self._get_update()
        self._v_cur = v

        # TODO: inherit?
        metric_vals = (residual_rkhs, dx_l2)
        if self._metrics is not None:
            metric_vals += tuple(m(self._x_cur) for m in self._metrics)

        return self, metric_vals

    @partial(jax.jit, static_argnames="max_iter")
    def iterate(
        self,
        x0: jnp.ndarray,
        max_iter: int = 10,
    ):
        solver, metric_vals_orig = self._initialize_iteration(x0)
        metric_vals = [metric_vals_orig]
        metric_vals = jnp.array(metric_vals)

        def body_fn(solver, *args):
            solver, metric_vals = solver._step()
            return solver, metric_vals

        solver, metric_vals_run = jax.lax.scan(body_fn, solver, length=max_iter)

        metric_vals_run = jnp.array(metric_vals_run)
        metric_vals = jnp.concatenate((metric_vals.T, metric_vals_run), axis=1)

        return solver, metric_vals

    def _tree_flatten(self):
        children = (
            self._x_cur,
            self._v_cur,
        )
        aux_data = {
            "operator": self._operator,
            "kernel": self._kernel,
            "metrics": self._metrics,
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        sol = cls(**aux_data)
        sol._x_cur, sol._v_cur = children
        return sol


jax.tree_util.register_pytree_node(
    PicardSolver, PicardSolver._tree_flatten, PicardSolver._tree_unflatten
)
