"""lforla-eval CLI — run benchmarks locally, push results to lforla.org."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from . import __version__
from .client import LforlaClient
from .model_runner import ModelRunner
from .oracle import (
    DRONE_SYSTEM_PROMPT,
    DroneOracle,
    RecruitmentOracle,
    parse_final_json,
    parse_final_team,
    run_drone_sample,
    sanitize_drone_output,
    score_team,
)

app = typer.Typer(
    name="lforla-eval",
    help="CLI for LFORLA — run LLM benchmarks locally and submit results.",
    no_args_is_help=True,
)
console = Console()

client = LforlaClient()


@app.callback()
def _main(
    api_url: Optional[str] = typer.Option(
        None, "--api-url", "-u", envvar="LFORLA_API_URL", help="API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", envvar="LFORLA_API_KEY", help="API key"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    global client
    client = LforlaClient(api_url=api_url, api_key=api_key)


@app.command()
def login(
    api_key: str = typer.Argument(..., help="Your LFORLA API key"),
):
    """Save API key to ~/.config/lforla/config.json."""
    client.save_key(api_key)
    rprint("[green]✓[/green] API key saved to ~/.config/lforla/config.json")


@app.command()
def whoami():
    """Check authentication status."""
    try:
        data = client.get("/auth/me") or {}
        user = data.get("user") or {}
        if user.get("email"):
            rprint(f"[green]Authenticated as[/green] {user['email']}")
        else:
            rprint("[yellow]Not authenticated[/yellow] — run [bold]lforla-eval login <api-key>[/bold]")
    except Exception as e:
        rprint(f"[red]Not authenticated[/red] — {e}")


@app.command()
def list_benchmarks(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
):
    """List available benchmarks on LFORLA."""
    params = {}
    if category:
        params["category"] = category
    benchmarks = client.get("/benchmarks", params=params)

    table = Table("Slug", "Name", "Category", "Score")
    for b in benchmarks:
        cats = ", ".join(b.get("categories", []))
        table.add_row(b["slug"], b["name"], cats, str(b.get("average_score", "—")))
    console.print(table)


@app.command()
def pull(
    slug: str = typer.Argument(..., help="Benchmark slug"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or yaml"),
):
    """Download benchmark metadata and samples."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    benchmark = client.get(f"/benchmarks/{slug}")
    meta_file = out / f"{slug}_meta.{format}"
    _write_file(meta_file, benchmark, format)
    rprint(f"[green]✓[/green] Benchmark metadata → {meta_file}")

    # Try dataset samples (v1 endpoints — may not be deployed yet)
    datasets = None
    try:
        datasets = client.get(f"/benchmarks/{slug}/datasets")
    except Exception:
        try:
            datasets = client.get(f"/datasets?slug={slug}")
        except Exception:
            rprint("[yellow]⚠[/yellow] Dataset endpoints not available yet on this API version")
            rprint("  Download sample files manually and use [bold]lforla-eval run[/bold]")

    if datasets:
        ds_file = out / f"{slug}_datasets.{format}"
        _write_file(ds_file, datasets, format)
        rprint(f"[green]✓[/green] Dataset info → {ds_file}")


@app.command()
def bench_push(
    file: str = typer.Argument(..., help="Benchmark YAML file (.lforla-benchmark.yaml)"),
    moderator: bool = typer.Option(False, "--moderator", help="Verify moderator/admin role before pushing"),
):
    """Push a benchmark definition to LFORLA (moderator-only)."""
    path = Path(file)
    if not path.exists():
        rprint(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    raw = path.read_text()
    data = yaml.safe_load(raw)

    required = ["name", "slug", "description", "taskType", "categories", "maxScore"]
    missing = [k for k in required if k not in data]
    if missing:
        rprint(f"[red]Error:[/red] Missing required fields: {', '.join(missing)}")
        raise typer.Exit(1)

    if moderator:
        try:
            me = client.get("/auth/me") or {}
            user = me.get("user") or {}
            role = user.get("role", "")
            is_admin = user.get("is_super_admin", False) or role == "admin"
            if not is_admin:
                rprint("[red]Error:[/red] --moderator flag requires admin/moderator role")
                rprint(f"  Your role: {role}")
                raise typer.Exit(1)
            rprint(f"[green]✓[/green] Moderator check passed (role: {role})")
        except Exception as e:
            rprint(f"[red]Error:[/red] Cannot verify role: {e}")
            rprint("  Make sure you are logged in as a moderator/admin")
            raise typer.Exit(1)

    payload = {
        "name": data["name"],
        "slug": data["slug"],
        "shortDescription": data.get("shortDescription", data.get("short_description", data["description"][:120])),
        "description": data["description"],
        "visibility": data.get("visibility", "public"),
        "modelType": "llm",
        "source": data.get("source"),
        "license": data.get("license"),
        "language": data.get("language", "en-US"),
        "pricingTier": data.get("pricingTier", "free"),
        "resultSchema": data.get("resultSchema", {}),
        "defaultConfig": data.get("evaluation", {}),
        "themeConfig": data.get("themeConfig", {}),
        "tagIds": [],
        "datasetIds": [],
    }

    # taskTypeId must be a valid UUID or omitted — the backend rejects null
    task_type_id = data.get("taskTypeId")
    if isinstance(task_type_id, str) and task_type_id:
        payload["taskTypeId"] = task_type_id

    rprint(f"\n[bold]Pushing benchmark:[/bold] {data['name']} ({data['slug']})")
    rprint(f"  Task type: {data['taskType']}")
    rprint(f"  Categories: {', '.join(data.get('categories', []))}")

    try:
        result = client.post("/benchmarks", payload)
        bid = result.get("id", "?")
        rprint(f"\n[green]✓[/green] Benchmark created: {bid}")
        rprint(f"  View at: {client.api_url}/benchmarks/{data['slug']}")

        files = data.get("files", [])
        if files:
            rprint(f"\n[bold]Files to upload:[/bold] {len(files)} file(s)")
            for f in files:
                src = f.get("source", f.get("path", ""))
                rprint(f"  • {f.get('path', src)} ({f.get('description', '')})")
            rprint("\n[yellow]Use 'lforla-eval dataset upload' to upload files[/yellow]")

        rprint("\n[yellow]── After pushing all changes ──[/yellow]")
        rprint("  To rebuild the frontend on the server:")
        rprint("    ssh hetzner_fips 'cd /home/jeremy/sites/lforla && docker compose \\")
        rprint("      -f docker-compose.yml -f docker-compose.vps.yml build --no-cache nginx \\")
        rprint("      && docker compose up -d --force-recreate nginx'")
        rprint("")

    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _score_drone_via_backend(sample_id: str, output: dict | None, oracle_calls: int) -> dict:
    """Score a drone-build output against the deterministic backend oracle."""
    body = {
        "sample_id": sample_id,
        "output": output,
        "agent_behavior": {
            "json_valid": isinstance(output, dict),
            "oracle_calls": oracle_calls,
            "distinct_informative_calls": oracle_calls,
        },
    }
    return client.post("/scoring/drone-build", json_body=body)


def run_sample_any(runner: ModelRunner, sample: dict) -> dict:
    """Run a sample through the runner.

    Dispatch order:
      1. ``bom_mode`` key present          → drone-build agentic oracle path
      2. ``oracle`` key present            → recruit-equipe agentic oracle path
      3. otherwise                         → single-turn
    """
    if "bom_mode" in sample or sample.get("benchmark_kind") == "drone-build":
        result = run_drone_sample(runner, sample)
        sanitized = sanitize_drone_output(result.get("output_parsed"))
        try:
            scored = _score_drone_via_backend(
                result["sample_id"], sanitized, result.get("oracle_calls", 0)
            )
        except Exception as e:
            # Unparseable JSON / contract violation: score 0 rather than drop
            # the sample so weak models still appear on the leaderboard.
            rprint(f"   [yellow]scoring failed ({e}) — recording 0[/yellow]")
            scored = {"overall_score": 0, "score": 0, "metrics": {}, "criteria": {}}
        result["score"] = scored.get("overall_score", scored.get("score", 0))
        result["overall_score"] = result["score"]
        result["metrics"] = scored.get("metrics", {})
        result["criteria"] = scored.get("metrics", {})
        result["gates"] = scored.get("gates")
        return result

    oracle_cfg = sample.get("oracle")
    if not oracle_cfg:
        return runner.run_sample(sample)
    sample_id = sample.get("id", sample.get("sample_id", ""))
    oracle = RecruitmentOracle(
        candidates=oracle_cfg.get("candidates", []),
        role_needs=oracle_cfg.get("role_needs", []),
        num_seats=oracle_cfg.get("num_seats", 1),
        total_budget=oracle_cfg.get("total_budget"),
        context=oracle_cfg.get("context", ""),
    )

    scenario = sample.get("scenario") or oracle_cfg.get("context") or \
        oracle_cfg.get("task_instruction", "")
    system = (
        "You are a recruitment manager assembling a team. You can query an oracle "
        "with the provided tools to inspect the hiring context and the candidate pool. "
        "Gather the information you need, then pick a team that best fits the CV criteria, "
        "respecting the number of available seats and (if any) the total budget. "
        "Respond with a JSON object that has a 'team' (array of {id}) and a 'justification' (string), "
        "and nothing else."
    )
    prompt = (
        f"Scenario: {scenario}\n"
        "Build your team and justify it. You MUST call the oracle tools to inspect "
        "candidates before finalizing. Return your final answer as a single JSON object "
        '{"team": [{"id": "..."}, ...], "justification": "..."}.'
    )

    tool_specs = oracle.get_tool_specs(provider=runner.provider)
    callers = oracle.get_tool_callers()

    def handle_call(name: str, args: dict):
        if name not in callers:
            return {"error": f"Unknown tool: {name}"}
        return callers[name](args)

    result = runner.run_oracle_sample(
        prompt,
        tool_specs,
        handle_call,
        system=system,
    )

    output = result.get("output", "")
    chosen = parse_final_team(output)
    if chosen is None:
        team = []
    else:
        team = [
            {"id": cid} for cid in chosen
            if any(str(c.get("id")) == str(cid) for c in oracle.candidates)
        ]

    scoring = score_team(team, oracle, include_debug=True)
    result["score"] = scoring["overall_score"]
    result["overall_score"] = scoring["overall_score"]
    result["team"] = team
    result["criteria"] = scoring["criteria"]
    result["sample_id"] = sample_id
    result["metrics"] = {**result.get("metrics", {}), **scoring["criteria"]}
    return result


@app.command()
def run(
    input_file: str = typer.Argument(
        ..., help="Samples file (JSON/YAML array of samples with 'input_data' field)"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Model identifier (e.g. gpt-4o, claude-3-opus)"
    ),
    provider: str = typer.Option(
        "openai", "--provider", "-p", help="Provider: openai, anthropic, ollama, generic"
    ),
    benchmark_id: Optional[str] = typer.Option(
        None, "--benchmark-id", "-b", help="Benchmark UUID (included in results)"
    ),
    output: str = typer.Option("results.json", "--output", "-o", help="Output results file"),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Only run first N samples"
    ),
):
    """Run a benchmark locally by calling an LLM API."""
    path = Path(input_file)
    if not path.exists():
        rprint(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    raw = path.read_text()
    stripped = raw.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            samples = json.loads(stripped)
            if isinstance(samples, dict):
                samples = samples.get("samples", samples.get("data", []))
                if not isinstance(samples, list):
                    samples = [samples]
        except json.JSONDecodeError:
            # Not a single JSON document → try JSONL (one JSON object per line)
            samples = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        samples = yaml.safe_load(raw)
        if isinstance(samples, dict):
            samples = samples.get("samples", samples.get("data", [samples]))
        else:
            samples = list(samples) if samples is not None else []

    if not isinstance(samples, list):
        rprint("[red]Error:[/red] Input must be a JSON/YAML array or {samples: [...]}")
        raise typer.Exit(1)

    if limit:
        samples = samples[:limit]

    runner = ModelRunner(model_id=model, provider=provider)

    results = []
    total = len(samples)
    with console.status(f"Running {total} samples...") as status:
        for i, sample in enumerate(samples):
            sample_id = sample.get("id", sample.get("sample_id", str(i)))
            status.update(f"[{i+1}/{total}] Running sample {sample_id}...")
            try:
                result = run_sample_any(runner, sample)
                result["sample_id"] = sample_id
                results.append(result)
            except Exception as e:
                rprint(f"\n[red]✗[/red] Sample {sample_id} failed: {e}")
                results.append({"sample_id": sample_id, "error": str(e)})

    output_data = {
        "benchmark_id": benchmark_id,
        "model": model,
        "provider": provider,
        "total_samples": total,
        "results": results,
    }
    out_path = Path(output)
    out_path.write_text(json.dumps(output_data, indent=2))
    successes = sum(1 for r in results if "error" not in r)
    rprint(f"\n[green]✓[/green] {successes}/{total} samples completed → {out_path}")


@app.command()
def dataset_upload(
    file: str = typer.Argument(..., help="Local file path to upload"),
    dataset_slug: str = typer.Argument(..., help="Dataset slug on LFORLA"),
    name: str = typer.Option(None, "--name", help="Dataset name (created if not exists)"),
    description: str = typer.Option(None, "--description", help="Dataset description"),
    version: str = typer.Option("1.0", "--version", help="Dataset version"),
    benchmark_slug: str = typer.Option(None, "--benchmark", "-b", help="Link to benchmark after upload"),
):
    """Upload a file to MinIO and create a dataset version."""
    path = Path(file)
    if not path.exists():
        rprint(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    file_size = path.stat().st_size
    rprint(f"Uploading {path.name} ({file_size / 1024 / 1024:.1f} MB)")

    # 1. Find or create dataset
    dataset = None
    try:
        dataset = client.get(f"/datasets/{dataset_slug}")
        rprint(f"[green]✓[/green] Dataset found: {dataset['id']}")
    except Exception:
        if not name:
            rprint("[red]Error:[/red] Dataset not found. Use --name to create it.")
            raise typer.Exit(1)
        dataset = client.post("/datasets", {
            "name": name,
            "slug": dataset_slug,
            "description": description or "",
        })
        rprint(f"[green]✓[/green] Dataset created: {dataset['id']}")

    # 2. Generate presigned upload URL
    file_key = None
    upload_url = None
    try:
        upload_info = client.post(
            f"/datasets/{dataset['id']}/presign-upload", {"fileName": path.name}
        ) or {}
        upload_url = upload_info.get("uploadUrl", "")
        file_key = upload_info.get("fileKey")
    except Exception as e:
        rprint(f"[yellow]⚠[/yellow] presign-upload failed: {e}")

    if not upload_url:
        rprint(
            "[red]✗[/red] No presigned upload URL — file was NOT uploaded. "
            "Aborting (no version created)."
        )
        rprint("  Check that you own the dataset (403 = forbidden) and that the API is reachable.")
        raise typer.Exit(1)

    rprint("  Uploading to MinIO...")
    import httpx
    with open(path, "rb") as f:
        resp = httpx.put(upload_url, content=f.read())
        resp.raise_for_status()
    rprint(f"[green]✓[/green] File uploaded to MinIO")

    # 3. Create dataset version
    import datetime
    sample_count = 0
    try:
        sample_count = sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))
    except OSError:
        pass
    version_data = {
        "version": version,
        "description": description or f"Upload of {path.name}",
        "releaseDate": datetime.date.today().isoformat(),
        "fileKey": file_key,
    }
    if sample_count:
        version_data["sampleCount"] = sample_count
    ds_version = client.post(f"/datasets/{dataset['id']}/versions", version_data)
    rprint(f"[green]✓[/green] Version created: {ds_version.get('id', '?')}")

    # 4. Link to benchmark
    if benchmark_slug:
        link = client.post(f"/benchmarks/{benchmark_slug}/datasets", {
            "datasetId": dataset["id"],
            "version": version,
        })
        rprint(f"[green]✓[/green] Linked to benchmark: {benchmark_slug}")

    rprint(f"\n[green]✓[/green] Dataset ready: {dataset_slug} (version {version})")


@app.command()
def report(
    results_file: str = typer.Argument(..., help="Results JSON from 'run' command"),
    output: str = typer.Option("report.md", "--output", "-o", help="Output report file (.md or .html)"),
    benchmark_slug: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Benchmark slug (fetches metadata for context)"
    ),
):
    """Generate an evaluation report (markdown or HTML) from results."""
    path = Path(results_file)
    if not path.exists():
        rprint(f"[red]Error:[/red] File not found: {results_file}")
        raise typer.Exit(1)

    data = json.loads(path.read_text())
    results = data.get("results", [data])
    if isinstance(results, dict):
        results = [results]

    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    meta = None
    if benchmark_slug:
        try:
            meta = client.get(f"/benchmarks/{benchmark_slug}")
        except Exception:
            meta = None

    scores = [_extract_score(s) for s in successes]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    total_tokens = sum(s.get("tokens_input", 0) + s.get("tokens_output", 0) for s in successes)
    total_ms = sum(s.get("execution_time_ms", 0) for s in successes)
    avg_latency = total_ms / len(successes) if successes else 0.0

    title = meta.get("name", "Evaluation report") if meta else "Evaluation report"
    slug = (meta.get("slug") or benchmark_slug) if meta else benchmark_slug

    lines = [
        f"# {title}",
        "",
        f"- **Benchmark**: `{slug or 'unknown'}`"
        f"  · **Model**: `{data.get('model', 'unknown')}`"
        f"  · **Provider**: `{data.get('provider', 'unknown')}`",
        f"- **Samples**: {len(successes)}/{data.get('total_samples', len(results))} succeeded"
        f"  · **Failures**: {len(failures)}",
        f"- **Average score**: **{avg_score:.2f}**"
        f"  · **Avg latency**: {avg_latency:.0f} ms"
        f"  · **Total tokens**: {total_tokens}",
        "",
        "## Per-sample results",
        "",
        "| # | Sample | Score | Latency (ms) | Tokens |",
        "|---|--------|-------|--------------|--------|",
    ]

    for i, s in enumerate(successes, 1):
        sid = s.get("sample_id", str(i))
        lines.append(
            f"| {i} | {sid} | {_extract_score(s)} | "
            f"{s.get('execution_time_ms', 0)} | "
            f"{s.get('tokens_input', 0) + s.get('tokens_output', 0)} |"
        )
    if not successes:
        lines.append("| — | — | — | — | — |")

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for f in failures:
            lines.append(f"- **{f.get('sample_id', '?')}**: {f.get('error', 'unknown error')}")

    if meta and (meta.get("resultSchema") or meta.get("result_schema")):
        result_schema = meta.get("resultSchema") or meta.get("result_schema")
        lines.append("")
        lines.append("## Result schema")
        lines.append("")
        lines.append("| Metric | Type | Min | Max | Higher is better |")
        lines.append("|--------|------|-----|-----|------------------|")
        for name, spec in result_schema.items():
            lines.append(
                f"| {name} | {spec.get('type', '')} | {spec.get('min', '')} | "
                f"{spec.get('max', '')} | {spec.get('higher_is_better', '')} |"
            )

    body = "\n".join(lines) + "\n"
    out_path = Path(output)
    if out_path.suffix.lower() == ".html":
        body = _to_html(title, body)
    out_path.write_text(body)
    rprint(f"[green]✓[/green] Report written → {out_path}")


def _to_html(title: str, markdown_body: str) -> str:
    import html as _html

    escaped = _html.escape(markdown_body)
    rows = []
    for line in escaped.splitlines():
        if line.startswith("# "):
            rows.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            rows.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("| ") and "---" not in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        elif line.startswith("|") and "---" in line:
            continue
        elif line.startswith("- "):
            rows.append(f"<li>{line[2:]}</li>")
        elif line == "":
            rows.append("</table>" if rows and rows[-1].startswith("<tr>") else "<br>")
        else:
            rows.append(f"<p>{line}</p>")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;"
        "padding:0 20px;color:#1f2937}h1{border-bottom:2px solid #e5e7eb;padding-bottom:8px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "td,th{border:1px solid #e5e7eb;padding:6px 10px;text-align:left}"
        "tr:nth-child(even){background:#f9fafb}li{margin:4px 0}</style></head>"
        f"<body>{''.join(rows)}</body></html>\n"
    )


@app.command()
def push(
    results_file: str = typer.Argument(..., help="Results JSON from 'run' command"),
    benchmark_id: Optional[str] = typer.Option(
        None, "--benchmark-id", "-b", help="Benchmark UUID (overrides value in results file)"
    ),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", help="Model UUID on LFORLA (if known)"
    ),
    visibility: str = typer.Option("public", "--visibility", help="public, private, org"),
):
    """Push evaluation results to LFORLA."""
    path = Path(results_file)
    if not path.exists():
        rprint(f"[red]Error:[/red] File not found: {results_file}")
        raise typer.Exit(1)

    data = json.loads(path.read_text())

    bid = benchmark_id or data.get("benchmark_id")
    if not bid:
        rprint("[red]Error:[/red] benchmark_id required (use --benchmark-id or include in results)")
        raise typer.Exit(1)

    results = data.get("results", [data])
    if isinstance(results, dict):
        results = [results]

    scores = [r for r in results if "error" not in r]
    if not scores:
        rprint("[red]Error:[/red] No successful results to push")
        raise typer.Exit(1)

    avg_score = sum(_extract_score(s) for s in scores) / len(scores)
    total_tokens = sum(s.get("tokens_input", 0) + s.get("tokens_output", 0) for s in scores)
    total_cost_usd = sum(float(s.get("total_cost_usd") or 0) for s in scores)
    total_ms = sum(s.get("execution_time_ms", 0) for s in scores)
    avg_latency = total_ms / len(scores) if scores else 0

    metric_names = [
        "role_coverage", "skill_match", "budget", "seat_limit",
        "seniority_balance", "meets_min_seniority",
        # drone-build
        "bom_completeness", "compatibility_score", "performance_score",
        "mass_accuracy", "thrust_accuracy", "endurance_accuracy",
        "internal_consistency", "cad_validity_score", "cad_consistency_score",
        "printability_score",
    ]
    oracle_metrics = {}
    for name in metric_names:
        vals = [s.get("criteria", {}).get(name) for s in scores
                if isinstance(s.get("criteria"), dict) and s.get("criteria", {}).get(name) is not None]
        if vals:
            oracle_metrics[name] = round(sum(vals) / len(vals), 3)

    payload = {
        "benchmark_id": bid,
        "model_id": model_id or data.get("model_id", ""),
        "overall_score": round(avg_score, 2),
        "total_tokens": total_tokens or None,
        "total_cost_usd": round(total_cost_usd, 6) if total_cost_usd > 0 else None,
        "metrics": {
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": total_tokens,
            "sample_count": len(scores),
            **oracle_metrics,
        },
        "visibility": visibility,
    }

    resp = client.post("/evaluations", payload)
    eval_id = resp.get("id", resp.get("evaluation_id", "unknown"))
    rprint(f"[green]✓[/green] Results pushed! Evaluation ID: {eval_id}")
    rprint(f"   Score: {payload['overall_score']}")
    rprint(f"   Samples: {len(scores)}")
    return resp


def _extract_score(result: dict) -> float:
    score = result.get("score") or result.get("overall_score") or 0
    return float(score)


def _write_file(path: Path, data: dict | list, format: str) -> None:
    if format == "yaml":
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    app()


if __name__ == "__main__":
    main()
