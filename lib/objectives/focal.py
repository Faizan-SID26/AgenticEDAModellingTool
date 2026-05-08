"""Focal-loss objective for LightGBM / XGBoost.

Focal loss [Lin et al. 2017, https://arxiv.org/abs/1708.02002] down-weights
the loss contribution from well-classified examples, focusing the gradient
on hard-to-classify ones. For severe class imbalance it often outperforms
class weighting.

Formula (binary):

    p_t = p   if y == 1 else (1 - p)
    L = -alpha_t * (1 - p_t)^gamma * log(p_t)

LightGBM expects an objective callable returning `(grad, hess)` arrays at
the *raw score* (logit) level, not at the probability level. We compute
the gradient and Hessian analytically.

Usage:
    from lib.objectives.focal import lgbm_focal_objective
    obj = lgbm_focal_objective(alpha=0.25, gamma=2.0)
    LGBMClassifier(objective=obj, ...)
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def lgbm_focal_objective(alpha: float = 0.25, gamma: float = 2.0) -> Callable[..., Any]:
    """Return a LightGBM-compatible objective callable.

    Parameters mirror the focal-loss paper:
    - `alpha`: class-balance weight in [0, 1]; positives get `alpha`, negatives get `1 - alpha`.
    - `gamma`: focusing parameter; gamma=0 reduces to weighted log-loss.
    """
    a = float(alpha)
    g = float(gamma)

    def _obj(y_true, y_pred):
        # y_pred here is the raw score (logit) from LightGBM.
        z = np.asarray(y_pred, dtype=float)
        y = np.asarray(y_true, dtype=float)
        p = _sigmoid(z)
        # Per-row alpha: alpha for positives, 1-alpha for negatives.
        alpha_t = np.where(y > 0.5, a, 1.0 - a)
        # Per-row p_t.
        p_t = np.where(y > 0.5, p, 1.0 - p)
        eps = 1e-9
        log_p_t = np.log(np.clip(p_t, eps, 1.0))
        # Gradient w.r.t. raw score z (derivation in the focal-loss paper / many blogs).
        # dL/dz = alpha_t * (1 - p_t)^gamma * (gamma * p_t * log(p_t) + p_t - 1) * sign(y - 0.5)
        # Equivalent simpler form using y as 0/1:
        sign = np.where(y > 0.5, 1.0, -1.0)
        grad = (
            alpha_t
            * np.power(1.0 - p_t, g)
            * (g * p_t * log_p_t + p_t - 1.0)
            * sign
        )
        # Hessian — second derivative. Bounded approximation that keeps
        # LightGBM stable; exact form is unwieldy and not needed for tree
        # split scoring.
        hess = (
            alpha_t
            * np.power(1.0 - p_t, g)
            * (g * (g + 1.0) * p_t * (1.0 - p_t) * log_p_t + p_t * (1.0 - p_t) * (g + 1.0))
        )
        # Numerical safety: hessian must be strictly positive for LightGBM.
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return _obj
