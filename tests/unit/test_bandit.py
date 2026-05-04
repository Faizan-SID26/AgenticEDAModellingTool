"""Thompson sampling tests."""
from __future__ import annotations

from lib.bandit import BanditState, pick_arm, posterior_means, sample, update


def test_initial_posteriors_uniform():
    s = BanditState()
    means = posterior_means(s)
    assert all(abs(v - 0.5) < 1e-9 for v in means.values())


def test_update_shifts_posterior():
    s = BanditState()
    s = update(s, "boosted_tree", info_gain_actual=0.9)
    means = posterior_means(s)
    assert means["boosted_tree"] > 0.5


def test_pick_arm_returns_known_arm():
    s = BanditState()
    arm = pick_arm(s, seed=42)
    assert arm in s.alpha


def test_repeated_high_reward_dominates():
    s = BanditState()
    for _ in range(50):
        s = update(s, "boosted_tree", 1.0)
        s = update(s, "linear", 0.0)
    means = posterior_means(s)
    assert means["boosted_tree"] > means["linear"]
