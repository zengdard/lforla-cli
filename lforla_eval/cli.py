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
    samples = json.loads(raw) if raw.strip().startswith("{") else yaml.safe_load(raw)

    if isinstance(samples, dict):
        samples = samples.get("samples", samples.get("data", [samples]))
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
                result = runner.run_sample(sample)
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
    total_ms = sum(s.get("execution_time_ms", 0) for s in scores)
    avg_latency = total_ms / len(scores) if scores else 0

    payload = {
        "benchmark_id": bid,
        "model_id": model_id or data.get("model_id", ""),
        "overall_score": round(avg_score, 2),
        "result_type": "generic",
        "metrics": {
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": total_tokens,
            "sample_count": len(scores),
        },
        "raw_outputs": {s["sample_id"]: s.get("output", "") for s in scores if "sample_id" in s},
        "visibility": visibility,
    }

    if data.get("model"):
        payload["model_name"] = data["model"]
    if data.get("provider"):
        payload["provider"] = data["provider"]

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
