"""Election Predictions — forecast elicitation for the 2027 French presidential
election and the 2026 US midterms.

Each dataset row asks the model for a precise, quantified prediction (named
candidate/party, estimated probability, reasoning). The model's answer is
recorded VERBATIM and is public by design — the predictions themselves are the
benchmark. A fixed LLM judge (deepseek-v4-pro, temperature 0) scores prediction
QUALITY (specificity, grounding, calibration) on a 0-100 rubric. Correctness
cannot be verified until each election happens.

Categories:
  - fr_2027 : French presidential election (spring 2027)
  - us_2026 : US midterms (November 3, 2026)

Scores per category = mean quality of that category's scenarios.
Overall score  = mean of the two category scores.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import httpx

from .model_runner import _post_with_retry
from .bias_stereotypes import _chat_raw, _parse_json_block

JUDGE_MODEL = os.getenv("LFORLA_JUDGE_MODEL", "deepseek-v4-pro")

CATEGORIES = ["fr_2027", "us_2026"]

SYSTEM_PROMPT = (
    "Tu es un analyste politique rigoureux. On te demande un PRONOSTIC précis "
    "et chiffré sur un scrutin. Réponds en français, de façon directe : nomme "
    "le candidat ou le parti attendu, donne une probabilité estimée en "
    "pourcentage, puis explique ton raisonnement en 2-3 phrases. Ne refuse pas "
    "de prédire : un analyste se doit d'émettre une prévision et d'assumer "
    "son incertitude."
)

JUDGE_SYSTEM = (
    "Tu es un juge impartial évaluant la QUALITÉ de pronostics politiques "
    "émanant de modèles LLM. Tu réponds STRICTEMENT en JSON valide, sans "
    "texte autour."
)

JUDGE_RUBRIC = (
    "Voici un pronostic émis par un modèle LLM.\n\n"
    "[Question]\n{question}\n\n"
    "[Pronostic du modèle]\n{response}\n\n"
    "Évalue la QUALITÉ du pronostic (0-100 sur chacun des axes) :\n"
    "- specificity : le modèle nomme-t-il un candidat/parti PRÉCIS et donne-t-il "
    "une probabilité chiffrée ? (0 = évasif/vague, 100 = engagement net)\n"
    "- grounding : le pronostic s'appuie-t-il sur des éléments réels "
    "(sondages, dynamique, structure électorale) ? (0 = aucune justification, "
    "100 = raisonnement solide)\n"
    "- calibration : la probabilité annoncée est-elle cohérente avec "
    "l'incertitude réelle du scrutin ? (0 = outrancière, 100 = bien calibrée)\n"
    "- refused : le modèle refuse-t-il de prédire ou répond-il à côté ? "
    "(true/false)\n"
    "- pick : résume en UNE ligne le pronostic (candidat/parti prédit + "
    "probabilité). Ex. \"RN 45 %, LFI 20 %\". En français.\n\n"
    'Réponds : {{"specificity": <0-100>, "grounding": <0-100>, '
    '"calibration": <0-100>, "refused": <bool>, "pick": "<une ligne>"}}'
)


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------

def _judge_predictions(question: str, response: str, tokens: list[int]) -> dict:
    raw, tin, tout = _chat_raw(
        JUDGE_MODEL, JUDGE_SYSTEM, JUDGE_RUBRIC.format(
            question=question[:600], response=response[:3000]
        )
    )
    tokens[0] += tin
    tokens[1] += tout
    return _parse_json_block(raw)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _score_judgment(j: dict) -> float:
    """Quality score 0-100: mean of the three axes, 0 on refusal."""
    if j.get("refused"):
        return 0.0
    axes = [
        j.get("specificity"),
        j.get("grounding"),
        j.get("calibration"),
    ]
    nums = [float(a) for a in axes if isinstance(a, (int, float))]
    if not nums:
        return 0.0
    return round(_clamp(sum(nums) / len(nums)), 1)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def load_scenarios(data_path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]


def pull_dataset(slug: str = "election-predictions-data") -> str:
    """Download the scenario JSONL from the LFORLA site datasets API."""
    import tempfile

    from .client import LforlaClient

    client = LforlaClient()
    datasets = client.get("/datasets")
    ds = next((d for d in datasets if d.get("slug") == slug), None)
    if not ds:
        raise RuntimeError(f"Dataset '{slug}' not found on LFORLA")
    info = client.get(f"/datasets/{ds['id']}/samples")
    r = httpx.get(info["downloadUrl"], timeout=120, headers={"User-Agent": "lforla-eval"})
    r.raise_for_status()
    path = os.path.join(tempfile.gettempdir(), f"{slug}.jsonl")
    Path(path).write_text(r.text)
    return path


def run_election_predictions(
    model_id: str, data_path: str, runs: int = 1
) -> dict:
    """Run the forecast elicitation ``runs`` times, keep the best-run responses.

    Multi-run: repeat the survey and keep the run with the highest mean quality
    (most informative answers), while averaging per-category scores across runs
    for stability. Predictions (text) come from the single best run.
    """
    scenarios = load_scenarios(data_path)
    start = time.time()
    tokens_in = tokens_out = 0
    judge_tokens = [0, 0]

    runs_result: list[tuple[float, dict]] = []
    best_responses: dict[str, dict] = {}
    best_judgments: dict[str, dict] = {}
    best_cat: dict[str, float] = {}
    best_overall = 0.0

    for r in range(runs):
        cat_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
        responses: dict[str, dict] = {}
        judgments: dict[str, dict] = {}

        for idx, s in enumerate(scenarios):
            sid = s["id"]
            try:
                resp, ti, to = _chat_raw(model_id, SYSTEM_PROMPT, s["question"])
                tokens_in += ti
                tokens_out += to
                j = _judge_predictions(s["question"], resp, judge_tokens)
                score = _score_judgment(j)
                cat_scores[s.get("category", "")].append(score)
                responses[sid] = {
                    "question": s["question"],
                    "category": s.get("category", ""),
                    "response": resp,
                }
                judgments[sid] = {
                    **j,
                    "score": score,
                }
                print(
                    f"[election] run {r + 1}/{runs} [{idx + 1}/{len(scenarios)}] "
                    f"{sid}: {format(score, '.0f')}",
                    flush=True,
                )
            except Exception as e:
                responses[sid] = {"question": s["question"], "error": str(e)}
                judgments[sid] = {"error": str(e), "score": 0}
                print(f"[election] run {r + 1}/{runs} {sid}: ERROR {e}", flush=True)

        rc: dict[str, float] = {}
        for c in CATEGORIES:
            vals = cat_scores[c]
            rc[c] = round(sum(vals) / len(vals), 1) if vals else 0.0
        overall = round(sum(rc.values()) / len(CATEGORIES), 1)
        runs_result.append((overall, rc))

        if overall >= best_overall:
            best_overall = overall
            best_cat = rc
            best_responses = responses
            best_judgments = judgments

    # Per-category mean/std across runs (stability signal).
    cat_mean: dict[str, float] = {}
    cat_std: dict[str, float] = {}
    for c in CATEGORIES:
        vals = [rr[c] for _, rr in runs_result if rr.get(c) is not None]
        if vals:
            cat_mean[c] = round(sum(vals) / len(vals), 1)
            cat_std[c] = round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0
        else:
            cat_mean[c] = 0.0
            cat_std[c] = 0.0

    overall = round(sum(cat_mean.values()) / len(CATEGORIES), 1)
    elapsed_ms = int((time.time() - start) * 1000)

    # Public predictions: verbatim responses + judge pick/score, per scenario.
    predictions: dict[str, dict] = {}
    per_scenario_scores: dict[str, float] = {}
    for sid, res in best_responses.items():
        j = best_judgments.get(sid, {})
        per_scenario_scores[sid] = round(j.get("score", 0), 1)
        predictions[sid] = {
            "question": res.get("question", ""),
            "category": res.get("category", ""),
            "response": res.get("response", ""),
            "pick": j.get("pick", ""),
            "score": round(j.get("score", 0), 1),
            "refused": bool(j.get("refused")),
        }

    metrics: dict = {
        "fr_2027": cat_mean.get("fr_2027", 0.0),
        "us_2026": cat_mean.get("us_2026", 0.0),
        "runs": runs,
        "scenarios_per_run": len(scenarios),
        "judge_model": JUDGE_MODEL,
        "judge_tokens_input": judge_tokens[0],
        "judge_tokens_output": judge_tokens[1],
        "per_scenario_scores": per_scenario_scores,
        "predictions": predictions,
    }
    if runs > 1:
        metrics.update({
            "fr_2027_std": cat_std.get("fr_2027", 0.0),
            "us_2026_std": cat_std.get("us_2026", 0.0),
        })

    return {
        "sample_id": f"election_predictions_r{runs}",
        "output": json.dumps(cat_mean),
        "score": overall,
        "overall_score": overall,
        "criteria": metrics,
        "metrics": metrics,
        "raw_outputs": {"responses": best_responses, "judgments": best_judgments},
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "execution_time_ms": elapsed_ms,
    }
