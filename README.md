# lforla-eval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

CLI client for [LFORLA](https://lforla.org) — run LLM benchmarks locally and push
results to the leaderboard. Pairs with the [LFORLA platform](https://lforla.org/cli).

## Features

- **`pull`** — download benchmark metadata and dataset samples (JSON/YAML)
- **`run`** — evaluate samples locally against an LLM API (OpenAI, Anthropic, Ollama, generic)
- **`push`** — submit results to the LFORLA leaderboard
- **`report`** — generate a markdown/HTML evaluation report from results
- **`dataset upload`** — upload sample files to MinIO and link them to a benchmark
- **`bench-push`** — publish benchmark definitions (moderator-only)

## Install

```bash
pip install git+https://github.com/lforla/pyforla.git
```

Or from source:

```bash
git clone https://github.com/lforla/pyforla.git
cd pyforla && pip install -e .
```

Verify:

```bash
lforla-eval --help
```

## Quick Start

```bash
# Save your API key (create one in your LFORLA account settings)
lforla-eval login "your-api-key"

# Download a benchmark's samples
lforla-eval pull root-cause-analysis -o ./data

# Run the samples locally against a model
lforla-eval run ./data/root-cause-analysis_samples.json -m gpt-4o -p openai

# Push the results to the leaderboard
lforla-eval push results.json --benchmark-id <uuid>
```

## Commands

| Command | Description |
|---------|-------------|
| `lforla-eval login <api-key>` | Save your API key to `~/.config/lforla/config.json` |
| `lforla-eval whoami` | Check authentication status |
| `lforla-eval list-benchmarks` | List available benchmarks |
| `lforla-eval pull <slug> [-o dir]` | Download benchmark metadata + samples |
| `lforla-eval run <samples> -m <model> [-p provider]` | Run samples against a model |
| `lforla-eval push <results> --benchmark-id <uuid>` | Submit results to LFORLA |
| `lforla-eval report <results> [-o report.md]` | Generate an evaluation report |
| `lforla-eval dataset upload <file> <slug>` | Upload a sample file and create a dataset version |
| `lforla-eval bench-push <file.yaml>` | Publish a benchmark definition (moderator-only) |

## Docker

A container image is provided for isolated or server-side runs:

```bash
docker build -t lforla-eval .
docker run --rm -v "$PWD:/data" lforla-eval run /data/samples.json -m gpt-4o -p openai
```

## Documentation

- Full CLI documentation: https://lforla.org/cli
- API reference: https://lforla.org/api/v1

## License

[MIT](LICENSE)
