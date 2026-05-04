"""Doom-loop detection tests."""
from __future__ import annotations

from lib.doom_loop import check
from lib.schemas.experiment import ExperimentResult, FitMetrics, SkepticResult
from lib.schemas.plan import PlanDict, PriorEvidence


def _plan(model: str = "logreg", area: str = "baseline") -> PlanDict:
    return PlanDict(
        id="P-x-aaaaaa",
        iteration=0,
        hypothesis_id="H-seed-1",
        model=model,
        features=["a", "b"],
        prior_evidence=PriorEvidence(kind="hypothesis_seed", reference="H-seed-1", summary="x"),
        technique_family="linear",
        area=area,
        expected_info_gain=0.5,
    )


def _exp(value: float) -> ExperimentResult:
    return ExperimentResult(
        id="P-x-aaaaaa",
        iteration=0,
        hypothesis_id="H-seed-1",
        model="logreg",
        features_used=["a", "b"],
        params={},
        calibrated=False,
        technique_family="linear",
        area="baseline",
        metrics=FitMetrics(),
        primary_metric="roc_auc",
        primary_metric_value=value,
        skeptic=SkepticResult(verdict="ACCEPT"),
    )


def test_no_fire_when_only_two():
    v = check([_plan(), _plan()], [_exp(0.5), _exp(0.5)], window=3)
    assert v.fired is False


def test_fires_on_repeat_with_flat_metric():
    plans = [_plan(), _plan(), _plan()]
    exps = [_exp(0.50), _exp(0.50), _exp(0.500001)]
    v = check(plans, exps, window=3)
    assert v.fired is True


def test_no_fire_when_metric_moves():
    plans = [_plan(), _plan(), _plan()]
    exps = [_exp(0.50), _exp(0.55), _exp(0.60)]
    v = check(plans, exps, window=3)
    assert v.fired is False


def test_no_fire_when_plan_changes():
    plans = [_plan(model="logreg"), _plan(model="lgbm_binary"), _plan(model="ridge")]
    exps = [_exp(0.50), _exp(0.50), _exp(0.50)]
    v = check(plans, exps, window=3)
    assert v.fired is False
