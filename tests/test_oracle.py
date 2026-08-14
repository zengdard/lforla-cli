"""Tests for the oracle recruitment benchmark logic."""

import json

import pytest

from lforla_eval.oracle import (
    RecruitmentOracle,
    parse_final_team,
    score_team,
)

SAMPLE = {
    "num_seats": 3,
    "total_budget": 420000,
    "role_needs": [
        {"role": "Machine Learning Engineer", "skills": ["python", "mlops", "fraud"], "critical": True},
        {"role": "Data Engineer", "skills": ["python", "spark", "sql", "elt"], "critical": True},
        {"role": "Data Product Owner", "skills": ["regulation", "rgpd", "dora"], "critical": True},
    ],
    "candidates": [
        {"id": "c01", "name": "A", "role": "Machine Learning Engineer", "seniority": "senior",
         "skills": ["python", "mlops", "fraud detection"], "expected_salary": 150000},
        {"id": "c03", "name": "B", "role": "Data Engineer", "seniority": "senior",
         "skills": ["python", "spark", "sql", "elt"], "expected_salary": 135000},
        {"id": "c05", "name": "C", "role": "Data Product Owner", "seniority": "senior",
         "skills": ["regulation", "rgpd", "dora"], "expected_salary": 145000},
        {"id": "c09", "name": "D", "role": "Data Product Owner", "seniority": "junior",
         "skills": ["product"], "expected_salary": 70000},
    ],
}


def make_oracle(sample=SAMPLE):
    return RecruitmentOracle(
        sample["candidates"],
        role_needs=sample["role_needs"],
        num_seats=sample["num_seats"],
        total_budget=sample["total_budget"],
    )


def test_oracle_tool_specs_openai():
    specs = make_oracle().get_tool_specs("openai")
    names = [s["function"]["name"] for s in specs]
    assert names == ["get_context", "get_candidates", "get_candidate"]


def test_oracle_tool_specs_anthropic():
    specs = make_oracle().get_tool_specs("anthropic")
    names = [s["name"] for s in specs]
    assert "get_candidates" in names and "get_context" in names


def test_get_candidates_filter():
    oracle = make_oracle()
    res = oracle.get_tool_callers()["get_candidates"]({"role": "Data Engineer"})
    assert res["count"] == 1
    assert res["candidates"][0]["id"] == "c03"


def test_get_candidate():
    oracle = make_oracle()
    res = oracle.get_tool_callers()["get_candidate"]({"id": "c05"})
    assert res["found"] is True
    assert res["candidate"]["role"] == "Data Product Owner"


def test_get_context():
    res = make_oracle().get_tool_callers()["get_context"]({})
    assert res["num_seats"] == 3
    assert res["total_budget"] == 420000
    assert len(res["roles"]) == 3


def test_score_full_coverage():
    """Best-fitting team within budget scores high."""
    oracle = make_oracle()
    team = [{"id": "c01"}, {"id": "c03"}, {"id": "c05"}]
    res = score_team(team, oracle)
    assert res["criteria"]["role_coverage"] == 1.0
    assert res["criteria"]["seat_limit"] == 1.0
    assert res["criteria"]["role_coverage"] == 1.0
    assert res["overall_score"] > 60


def test_score_junior_critical_penalised():
    """A junior filling a critical role is penalised on meets_min_seniority."""
    oracle = make_oracle()
    team = [{"id": "c01"}, {"id": "c03"}, {"id": "c09"}]
    res = score_team(team, oracle)
    assert res["criteria"]["meets_min_seniority"] == 0.5


def test_score_over_seats_penalised():
    oracle = make_oracle()
    team = [{"id": "c01"}, {"id": "c03"}, {"id": "c05"}, {"id": "c09"}]
    res = score_team(team, oracle)
    assert res["criteria"]["seat_limit"] < 1.0


def test_parse_final_team():
    assert parse_final_team('{"team": [{"id": "c01"}, {"id": "c03"}], "justification": "ok"}') == ["c01", "c03"]
    assert parse_final_team('[{"id": "c01"}]') == ["c01"]
    assert parse_final_team("```json\n{\"team\": [{\"id\": \"c01\"}]}\n```") == ["c01"]
    assert parse_final_team("no json here") is None
