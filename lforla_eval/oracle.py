"""Oracle benchmark support for LFORLA.

An "oracle" benchmark turns evaluation into an agentic task: instead of a single
prompt, the model queries one or more oracle functions (tool-calling) to gather
information before producing a final answer.

This module ships the recruitment oracle used by the ``recruit-equipe`` benchmark:
a pool of fake CVs + a hiring context (available seats, budget, required roles).
The model queries candidates, then submits a team + justification. The composition
is scored against the CV criteria (no single "correct" team).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================================
# Oracle tool spec
# ============================================================================

def openai_tool_spec(name: str, description: str, parameters: dict) -> dict:
    """Build an OpenAI-style ``tools`` entry."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def anthropic_tool_spec(name: str, description: str, input_schema: dict) -> dict:
    """Build an Anthropic-style ``tools`` entry."""
    return {
        "name": name,
        "description": description,
        "input_schema": input_schema,
    }


# ============================================================================
# Recruitment oracle
# ============================================================================

RoleNeed = dict  # {role, skills: [..], critical: bool}


def _candidate(obj: Any) -> dict:
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    elif not isinstance(obj, dict):
        obj = json.loads(obj)
    return obj


class RecruitmentOracle:
    """An oracle that exposes fake CVs and hiring context to the model.

    Use ``get_tool_specs()`` to build the provider tool list and
    ``handle_call(name, args)`` to service tool calls.
    """

    def __init__(
        self,
        candidates: list[dict],
        *,
        role_needs: list[RoleNeed],
        num_seats: int,
        total_budget: float | None = None,
        context: str = "",
    ):
        self.candidates = [_candidate(c) for c in candidates]
        self.role_needs = role_needs
        self.num_seats = num_seats
        self.total_budget = total_budget
        self.context = context

    # -- tools -------------------------------------------------------------
    def get_tool_specs(self, provider: str = "openai") -> list[dict]:
        def fn(name, description, properties, required_=()):
            params = {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            }
            if required_:
                params["required"] = list(required_)
            if provider == "anthropic":
                return anthropic_tool_spec(name, description, params)
            return openai_tool_spec(name, description, params)

        specs = [
            fn(
                "get_context",
                "Return the hiring context: number of seats available, total salary budget, "
                "and the list of roles the team must cover (with required skills).",
                {},
            ),
            fn(
                "get_candidates",
                "List candidates available in the talent pool, optionally filtered by role and skill.",
                {
                    "role": {"type": "string", "description": "Job role to filter by"},
                    "skill": {"type": "string", "description": "Skill to filter by"},
                },
            ),
            fn(
                "get_candidate",
                "Return the full profile (CV) of one candidate by id.",
                {"id": {"type": "string", "description": "Candidate id"}},
                ("id",),
            ),
        ]
        return specs

    def get_tool_callers(self) -> dict[str, Callable]:
        return {
            "get_context": self._call_context,
            "get_candidates": self._call_candidates,
            "get_candidate": self._call_candidate,
        }

    # -- implementations ----------------------------------------------------
    def _call_context(self, args: dict) -> dict:
        return {
            "num_seats": self.num_seats,
            "total_budget": self.total_budget,
            "roles": [
                {"role": n.get("role"), "skills": n.get("skills", []), "critical": n.get("critical", False)}
                for n in self.role_needs
            ],
            "context": self.context,
        }

    def _call_candidates(self, args: dict) -> dict:
        role = (args.get("role") or "").lower()
        skill = (args.get("skill") or "").lower()
        out = []
        for c in self.candidates:
            if role and role not in (c.get("role", "") or "").lower():
                continue
            if skill and not any(
                skill in (s or "").lower() for s in c.get("skills", [])
            ):
                continue
            out.append(self.public_profile(c))
        return {"count": len(out), "candidates": out}

    def _call_candidate(self, args: dict) -> dict:
        cid = args.get("id", "")
        for c in self.candidates:
            if str(c.get("id")) == str(cid):
                return {"candidate": c, "found": True}
        return {"found": False, "error": f"No candidate with id {cid}"}

    @staticmethod
    def public_profile(c: dict) -> dict:
        keys = {
            "id",
            "name",
            "role",
            "seniority",
            "years_experience",
            "skills",
            "expected_salary",
            "profile",
            "strengths",
            "weaknesses",
        }
        return {k: c[k] for k in keys if k in c}


# ============================================================================
# Scoring — composition evaluated on CV criteria
# ============================================================================

WEIGHTS = {
    "role_coverage": 0.30,
    "skill_match": 0.25,
    "budget": 0.15,
    "seat_limit": 0.10,
    "seniority_balance": 0.10,
    "meets_min_seniority": 0.10,
}


def _skill_match_score(candidate: dict, role_need: dict) -> float:
    """Fraction of required skills that the candidate covers for a role."""
    required = [s.lower() for s in role_need.get("skills", [])]
    if not required:
        return 1.0
    have = set((s or "").lower() for s in candidate.get("skills", []))
    covered = sum(1 for s in required if s in have)
    return covered / len(required)


def choose_role(candidate: dict, role_needs: list[RoleNeed]) -> RoleNeed | None:
    """Best matching role for a candidate, favouring critical roles on ties."""
    best: RoleNeed | None = None
    best_score = -1.0
    for n in role_needs:
        score = _skill_match_score(candidate, n)
        if n.get("critical"):
            score += 0.01
        if score > best_score:
            best_score = score
            best = n
    return best if best_score > 0 else None


def score_team(
    team: list[dict],
    oracle: RecruitmentOracle,
    *,
    include_debug: bool = False,
) -> dict[str, Any]:
    """Score a composed team (list of candidate dicts) against the oracle context.

    Returns a dict of 0-1 sub-scores, a weighted aggregate in [0, 1], and
    a 0-100 ``overall_score``.
    """
    all_candidates = {str(c.get("id")): c for c in oracle.candidates}
    chosen = [
        all_candidates[str(c.get("id"))] for c in team if str(c.get("id")) in all_candidates
    ]
    num_seats = max(1, oracle.num_seats or 1)
    needs = oracle.role_needs or []

    # Role coverage & skill match
    covered_roles: set[str] = set()
    skill_scores: list[float] = []
    for cand in chosen:
        match = choose_role(cand, needs)
        if match:
            covered_roles.add(match.get("role"))
            skill_scores.append(_skill_match_score(cand, match))
    role_coverage = len(covered_roles) / len(needs) if needs else 1.0
    skill_match = sum(skill_scores) / len(skill_scores) if skill_scores else 0.0

    # Budget
    budget_ok = 1.0
    if oracle.total_budget is not None:
        spend = sum(float(c.get("expected_salary", 0) or 0) for c in chosen)
        max_spend = max(1.0, oracle.total_budget)
        budget_ok = min(1.0, (max_spend - spend) / max_spend + 1)
        if spend > oracle.total_budget:
            budget_ok = max(0.0, budget_ok)

    # Seat limit
    seat_ok = min(1.0, num_seats / max(1, len(chosen))) if chosen else 0.0

    # Seniority balance: prefer a mix of senior & junior (penalise all-senior/all-junior)
    if chosen:
        seniors = sum(
            1 for c in chosen if (c.get("seniority", "") or "").lower() in ("senior", "lead")
        )
        ratio = seniors / len(chosen)
        seniority_balance = 1.0 - abs(ratio - 0.5) * 2
    else:
        seniority_balance = 0.0

    # Minimum seniority: every critical role should at least be filled by a non-junior
    meets_min = 1.0
    if needs:
        for cand in chosen:
            match = choose_role(cand, needs)
            if match and match.get("critical") and (cand.get("seniority", "") or "").lower() == "junior":
                meets_min = 0.5
                break

    sub = {
        "role_coverage": round(role_coverage, 3),
        "skill_match": round(skill_match, 3),
        "budget": round(budget_ok, 3),
        "seat_limit": round(seat_ok, 3),
        "seniority_balance": round(seniority_balance, 3),
        "meets_min_seniority": round(meets_min, 3),
    }
    aggregate = sum(sub[k] * WEIGHTS[k] for k in WEIGHTS)

    result: dict[str, Any] = {
        "overall_score": round(aggregate * 100, 2),
        "score": round(aggregate * 100, 2),
        "team_size": len(chosen),
        "num_seats": num_seats,
        "criteria": sub,
    }
    if include_debug:
        result["chosen_ids"] = [c.get("id") for c in chosen]
    return result


def parse_final_team(output: str) -> list[str] | None:
    """Extract a list of chosen candidate ids from the model's final JSON answer.

    Accepts both a pure-JSON document and a ``{"team": [...], "justification": ...}``
    object embedded in markdown/multi-line prose.
    """
    if not output:
        return None
    text = output.strip()
    # Strip markdown fences
    if "```" in text:
        code_blocks = [b for b in text.split("```") if b.strip().lstrip().startswith("{")]
        if code_blocks:
            text = code_blocks[0].strip()

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            decoded = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    team = decoded.get("team") if isinstance(decoded, dict) else None
    if isinstance(team, list):
        return [str(row.get("id") or row) for row in team if isinstance(row, (dict, str))]
    if isinstance(decoded, list):
        return [str(row.get("id") or row) for row in decoded]
    return None
