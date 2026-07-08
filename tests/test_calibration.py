# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Stage-C shrink unit tests — the machinery behind "provably no worse than the
estimate." These test pure functions on synthetic arrays (no trained model
needed), so they run anywhere, including CI on a fresh clone.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.train import fit_estimate_shrink, apply_estimate_shrink, _TIER_NAMES  # noqa: E402


def test_lambda_zero_collapses_to_estimate():
    """λ=0 everywhere → the prediction becomes the estimate (log_ratio 0)."""
    p   = np.array([0.3, -0.5, 0.8, 0.1])
    est = np.array([5_000., 30_000., 100_000., 2_000_000.])
    shrink = {"global": 0.0, **{t: 0.0 for t in _TIER_NAMES}}
    out = apply_estimate_shrink(shrink, p, est)
    assert np.allclose(out, 0.0)


def test_shrink_never_amplifies():
    """0 ≤ λ ≤ 1 ⇒ |shrunk| ≤ |raw| elementwise (it only pulls toward the estimate)."""
    rng = np.random.default_rng(0)
    p   = rng.normal(0, 0.6, 500)
    est = rng.uniform(1_000, 2_000_000, 500)
    shrink = {"global": 0.4, "lt_10k": 0.5, "10k_50k": 0.2,
              "50k_200k": 0.1, "200k_1m": 0.15, "gt_1m": 0.0}
    out = apply_estimate_shrink(shrink, p, est)
    assert np.all(np.abs(out) <= np.abs(p) + 1e-9)


def test_fitted_lambdas_in_unit_interval():
    """Every fitted λ (global + per tier) must lie in [0, 1]."""
    rng = np.random.default_rng(1)
    n = 4_000
    est  = rng.uniform(1_000, 2_000_000, n)
    y_lr = rng.normal(0, 0.5, n)
    p_lr = 0.3 * y_lr + rng.normal(0, 0.5, n)   # weakly informative predictions
    shrink = fit_estimate_shrink(y_lr, p_lr, est)
    for k, v in shrink.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} outside [0,1]"


def test_defers_completely_when_deviating_hurts():
    """If deviating from the estimate strictly increases error (predictions anti-
    correlated with truth), the λ=0-in-grid guarantee forces every λ to 0 — the
    model can never end up worse than the estimate on validation."""
    rng = np.random.default_rng(2)
    n = 4_000
    est  = rng.uniform(1_000, 2_000_000, n)
    y_lr = rng.normal(0, 0.5, n)
    p_lr = -2.0 * y_lr                          # any λ>0 pushes away from truth
    shrink = fit_estimate_shrink(y_lr, p_lr, est)
    assert all(v == 0.0 for v in shrink.values()), shrink


if __name__ == "__main__":
    test_lambda_zero_collapses_to_estimate()
    test_shrink_never_amplifies()
    test_fitted_lambdas_in_unit_interval()
    test_defers_completely_when_deviating_hurts()
    print("✅ Stage-C shrink guarantees hold (λ∈[0,1], never amplifies, defers when deviating hurts)")
