# Data

This directory contains the public LitReviewBench data snapshot released with the ICML 2026 paper.

## Files

| File | Records | Description |
| --- | ---: | --- |
| `battles.jsonl` | 2,754 | AI-domain paired draft comparisons. |
| `expert_outcomes.jsonl` | 2,754 | Dimension-wise expert outcomes keyed by `battle_id`. |
| `topics.jsonl` | 925 | Normalized topic queries with source-paper and taxonomy metadata. |
| `evaluator/litjudge_context_cache.jsonl` | cache | Cached LitJudge retrieval context. |
| `evaluator/litjudge_context_cache_mmr.jsonl` | cache | Cached diversity-aware LitJudge retrieval context. |

## Schemas

See [../docs/data_format.md](../docs/data_format.md) for the full schema of battle records, expert outcomes, and topics.

## Loading

```python
import json
from pathlib import Path

def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

battles = load_jsonl("data/battles.jsonl")
outcomes = load_jsonl("data/expert_outcomes.jsonl")
topics = load_jsonl("data/topics.jsonl")
```

## Notes

- Labels use the four-way outcome set `A`, `B`, `Tie`, and `BothBad`.
- Annotator identifiers are pseudonymized.
- The AI-domain files are the public benchmark release in this repository.
- Generated evaluator outputs should be written to ignored output paths rather than committed to the data snapshot.
