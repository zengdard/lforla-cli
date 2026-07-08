# lforla-eval

CLI client for [LFORLA](https://lforla.org) — run LLM benchmarks locally and push results.

```bash
pip install lforla-eval

lforla-eval login "your-api-key"
lforla-eval list-benchmarks
lforla-eval pull drop -o ./data
lforla-eval run ./data/drop_samples.json -m gpt-4o -p openai
lforla-eval push results.json --benchmark-id <uuid>
```
