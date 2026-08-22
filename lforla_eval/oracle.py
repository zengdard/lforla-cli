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
    decoded = parse_final_json(output)
    if isinstance(decoded, dict):
        team = decoded.get("team")
        if isinstance(team, list):
            return [str(row.get("id") or row) for row in team if isinstance(row, (dict, str))]
        return None
    if isinstance(decoded, list):
        return [str(row.get("id") or row) for row in decoded]
    return None


def parse_final_json(output: str) -> Any:
    """Extract the first JSON object/array from the model's final answer.

    Accepts pure JSON, JSON embedded in markdown fences, or JSON buried in
    prose. Falls back to ``json_repair`` for structurally broken LLM JSON
    (unescaped quotes/brackets in embedded code are common). Returns
    ``None`` when nothing salvageable is found.
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
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    try:
        from json_repair import repair_json

        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, (dict, list)):
            return repaired
    except ImportError:
        pass
    return None


# ============================================================================
# Drone build oracle
# ============================================================================

class DroneOracle:
    """An oracle that exposes a drone design mission and its component catalog.

    Backs the ``drone-build`` benchmark. Two information tiers mirror the
    backend contract: ``get_catalog_summary`` lists parts with prices only,
    while full datasheets (masses, KV, currents, thrust tables) are revealed
    one part at a time via ``get_component`` — forcing the model to explore
    and do its own engineering math.

    Use ``get_tool_specs()`` to build the provider tool list and
    ``handle_call(name, args)`` to service tool calls.
    """

    def __init__(self, sample: dict):
        self.sample_id = sample.get("id", "")
        self.bom_mode = sample.get("bom_mode", "catalog")
        self.mission: dict = sample.get("mission", {}) or {}
        self.catalog: list[dict] = sample.get("catalog", []) or []
        self._by_id = {str(c.get("id")): c for c in self.catalog}

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
                "get_mission",
                "Return the mission brief: description, hard constraints (budget USD, minimum "
                "thrust-to-weight, flight time target, wheelbase class, max prop size...), the "
                "required and optional component types, and whether an engineering analysis "
                "block is required.",
                {},
            ),
            fn(
                "get_catalog_summary",
                "List every component in the catalog with ID, type, name, manufacturer, "
                "part number and unit price ONLY. Full specs (mass, KV, currents, thrust "
                "table, dimensions) require per-part get_component calls.",
                {
                    "component_type": {
                        "type": "string",
                        "description": "Optional filter by component type (e.g. motor, esc, battery)",
                    },
                },
            ),
            fn(
                "get_component",
                "Return the FULL datasheet of one component by id: mass_g, unit_price_usd "
                "and the complete specs object (KV, currents, thrust table, dimensions...).",
                {"part_id": {"type": "string", "description": "Component id from get_catalog_summary"}},
                ("part_id",),
            ),
        ]
        return specs

    def get_tool_callers(self) -> dict[str, Callable]:
        return {
            "get_mission": self._call_mission,
            "get_catalog_summary": self._call_catalog_summary,
            "get_component": self._call_component,
        }

    # -- implementations ----------------------------------------------------
    def _call_mission(self, args: dict) -> dict:
        m = self.mission
        out: dict[str, Any] = {
            "mission_id": self.sample_id,
            "description": m.get("description", ""),
            "constraints": m.get("constraints", {}),
            "required_component_types": m.get("required_component_types", []),
            "optional_component_types": m.get("optional_component_types", []),
            "requires_analysis": bool(m.get("requires_analysis")),
            "bom_mode": self.bom_mode,
        }
        if self.bom_mode == "open":
            out["note"] = (
                "OPEN BOM MODE: no catalog is provided. Cite real commercially available parts "
                "by name/part number in part_id."
            )
        return out

    def _call_catalog_summary(self, args: dict) -> dict:
        ctype = (args.get("component_type") or "").lower()
        out = []
        for c in self.catalog:
            if ctype and ctype not in (c.get("type", "") or "").lower():
                continue
            out.append(
                {
                    k: c[k]
                    for k in ("id", "type", "name", "manufacturer", "part_number", "unit_price_usd")
                    if k in c
                }
            )
        return {"count": len(out), "components": out}

    def _call_component(self, args: dict) -> dict:
        pid = args.get("part_id", "")
        comp = self._by_id.get(str(pid))
        if comp is None:
            return {"found": False, "error": f"No component with id {pid}"}
        return {"found": True, "component": comp}


DRONE_SYSTEM_PROMPT = """You are an experienced multirotor engineer. Query the oracle to inspect the mission \
constraints, then explore the component catalog: get_catalog_summary lists parts with \
prices only — full specs (masses, KV, currents, thrust tables) require per-part \
get_component calls. Design the complete build within budget, verify electrical \
compatibility (KV vs cells, ESC/battery current margins), and compute YOURSELF the \
all-up weight, total thrust, thrust-to-weight, hover current and estimated flight time. \
Declare your math in the analysis block. Author parametric OpenSCAD source for the frame \
structure. Return STRICT JSON:
{"architecture": "...",
 "bom": [{"component_type": "...", "part_id": "...", "quantity": N}],
 "total_cost_usd": F,
 "analysis": {...},
 "cad": {"format": "openscad", "files": [{"name": "frame.scad", "content": "..."}]},
 "justification": "..."}"""


DRONE_USER_PROMPT = """Design the drone for this mission. You MUST query get_mission first, then \
explore the catalog with get_catalog_summary and inspect candidate components with \
get_component before finalizing your BOM. Compute your own engineering analysis and author \
the OpenSCAD frame source. Return your final answer as a single STRICT JSON object with keys \
architecture, bom, total_cost_usd, analysis, cad, justification."""


# Keys accepted by the backend's strict droneOutputSchema (Zod .strict()).
_DRONE_ANALYSIS_KEYS = (
    "component_masses",
    "all_up_weight_g",
    "thrust_per_motor_g",
    "max_thrust_total_g",
    "thrust_to_weight",
    "hover_current_a",
    "battery_sustained_current_a",
    "estimated_flight_time_min",
)


def _num(value: Any) -> float | None:
    """Best-effort numeric coercion ('531 g' -> 531.0)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().rstrip("gGAaVWwW ").replace(",", ""))
        except ValueError:
            return None
    return None


def sanitize_drone_output(parsed: Any) -> dict | None:
    """Conform a model's parsed drone-build answer to the strict output contract.

    Real models routinely add bonus keys (dry_weight_g, ...), emit numbers as
    strings, or overshoot length caps — the backend Zod schema is ``strict()``
    and would reject the whole submission. This drops unknown keys, coerces
    numerics and clamps lengths so the deterministic scorer sees a valid
    contract without rewarding or punishing anything beyond it.
    """
    if not isinstance(parsed, dict):
        return None

    out: dict[str, Any] = {}

    arch = parsed.get("architecture")
    if isinstance(arch, str) and arch.strip():
        out["architecture"] = arch.strip()[:120]

    bom: list[dict] = []
    for line in parsed.get("bom", []) or []:
        if not isinstance(line, dict) or len(bom) >= 100:
            continue
        entry: dict[str, Any] = {}
        for key in ("component_type", "part_id", "part_number", "name"):
            value = line.get(key)
            if isinstance(value, str) and value.strip():
                entry[key] = value.strip()[:200]
        qty = _num(line.get("quantity"))
        if qty is not None:
            entry["quantity"] = int(min(max(qty, 0), 100))
        # A usable line must identify a part
        if any(k in entry for k in ("part_id", "part_number", "name")):
            bom.append(entry)
    if bom:
        out["bom"] = bom

    cost = _num(parsed.get("total_cost_usd"))
    if cost is not None:
        out["total_cost_usd"] = min(max(cost, 0), 1_000_000)

    analysis_raw = parsed.get("analysis")
    if isinstance(analysis_raw, dict):
        analysis: dict[str, Any] = {}
        for key in _DRONE_ANALYSIS_KEYS:
            value = analysis_raw.get(key)
            if key == "component_masses":
                masses: list[dict] = []
                for m in value or []:
                    if not isinstance(m, dict) or len(masses) >= 200:
                        continue
                    mass_entry: dict[str, Any] = {}
                    pid = m.get("part_id")
                    if isinstance(pid, str) and pid.strip():
                        mass_entry["part_id"] = pid.strip()[:120]
                    unit_mass = _num(m.get("unit_mass_g"))
                    if unit_mass is not None:
                        mass_entry["unit_mass_g"] = min(max(unit_mass, 0), 100_000)
                    qty = _num(m.get("qty"))
                    if qty is not None:
                        mass_entry["qty"] = int(min(max(qty, 0), 100))
                    if mass_entry:
                        masses.append(mass_entry)
                if masses:
                    analysis[key] = masses
            elif key == "thrust_per_motor_g" and isinstance(value, dict):
                prop_inch = _num(value.get("prop_inch"))
                thrust_g = _num(value.get("thrust_g"))
                thrust_entry: dict[str, Any] = {}
                if prop_inch is not None:
                    thrust_entry["prop_inch"] = min(max(prop_inch, 0), 60)
                if thrust_g is not None:
                    thrust_entry["thrust_g"] = min(max(thrust_g, 0), 50_000)
                if thrust_entry:
                    analysis[key] = thrust_entry
            else:
                num = _num(value)
                if num is not None:
                    analysis[key] = num
        if analysis:
            out["analysis"] = analysis

    cad_raw = parsed.get("cad")
    if isinstance(cad_raw, dict):
        cad: dict[str, Any] = {}
        fmt = cad_raw.get("format")
        if isinstance(fmt, str) and fmt.strip():
            cad["format"] = fmt.strip()[:40]
        files: list[dict] = []
        for f in cad_raw.get("files", []) or []:
            if not isinstance(f, dict) or len(files) >= 50:
                continue
            file_entry: dict[str, Any] = {}
            name = f.get("name")
            if isinstance(name, str) and name.strip():
                file_entry["name"] = name.strip()[:200]
            content = f.get("content")
            if isinstance(content, str) and content.strip():
                file_entry["content"] = content[:500_000]
            if file_entry:
                files.append(file_entry)
        if files:
            cad["files"] = files
        if cad:
            out["cad"] = cad

    justification = parsed.get("justification")
    if isinstance(justification, str) and justification.strip():
        out["justification"] = justification.strip()[:20_000]

    # Contract requires at least one BOM line to be scoreable
    return out if "bom" in out else None


def run_drone_sample(runner: Any, sample: dict) -> dict:
    """Run one drone-build sample through the agentic oracle loop.

    Returns the raw runner result enriched with the parsed output contract
    (``output_parsed``), oracle tool-call bookkeeping and the sample id.
    Scoring is NOT done here — the parsed output must be sent to the
    deterministic backend scorer (POST /scoring/drone-build).
    """
    oracle = DroneOracle(sample)

    tool_specs = oracle.get_tool_specs(provider=runner.provider)
    callers = oracle.get_tool_callers()

    def handle_call(name: str, args: dict):
        if name not in callers:
            return {"error": f"Unknown tool: {name}"}
        return callers[name](args)

    result = runner.run_oracle_sample(
        DRONE_USER_PROMPT,
        tool_specs,
        handle_call,
        system=DRONE_SYSTEM_PROMPT,
    )

    parsed = parse_final_json(result.get("output", ""))
    result["output_parsed"] = parsed if isinstance(parsed, dict) else None
    result["oracle_calls"] = len(result.get("tool_calls", []))
    result["sample_id"] = sample.get("id", sample.get("sample_id", ""))
    return result
