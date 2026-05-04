"""Researcher eval.

The researcher's output is a `PlanDict`. We test the *constraints* a
real plan dict must satisfy: validates against the schema, references
real (or sketch-derivable) features, has a non-empty prior_evidence,
and lands on a known model + technique family.
"""
from __future__ import annotations

import pytest

from lib.schemas.plan import PlanDict, PriorEvidence


def test_well_formed_plan_validates():
    p = PlanDict(
        id="P-3-abcdef",
        iteration=3,
        hypothesis_id="H-seed-3",
        model="lgbm_binary",
        features=["+all_allowed", "engineered:interactions_top5"],
        params={"num_leaves": 31, "learning_rate": 0.05},
        calibrate=True,
        prior_evidence=PriorEvidence(
            kind="sketch_query",
            reference="top_interactions:abc",
            summary="(temp,press) MI=0.42 — highest pair",
        ),
        technique_family="boosted_tree",
        area="interactions",
        expected_info_gain=0.7,
    )
    again = PlanDict.model_validate_json(p.model_dump_json())
    assert again.id == p.id


def test_missing_prior_evidence_rejected():
    with pytest.raises(Exception):
        PlanDict(
            id="P-3-abcdef",
            iteration=3,
            hypothesis_id="H-seed-3",
            model="lgbm_binary",
            features=["+all_allowed"],
            # prior_evidence omitted
            technique_family="boosted_tree",
            area="interactions",
            expected_info_gain=0.7,
        )


def test_id_must_start_with_p_dash():
    with pytest.raises(Exception):
        PlanDict(
            id="X-3-abc",
            iteration=3,
            hypothesis_id="H-seed-3",
            model="logreg",
            features=["a"],
            prior_evidence=PriorEvidence(kind="hypothesis_seed", reference="H-seed-3", summary="x"),
            technique_family="linear",
            area="baseline",
            expected_info_gain=0.5,
        )


def test_features_no_empty_token():
    with pytest.raises(Exception):
        PlanDict(
            id="P-3-abc",
            iteration=3,
            hypothesis_id="H-seed-3",
            model="logreg",
            features=["a", "  "],
            prior_evidence=PriorEvidence(kind="hypothesis_seed", reference="H-seed-3", summary="x"),
            technique_family="linear",
            area="baseline",
            expected_info_gain=0.5,
        )
