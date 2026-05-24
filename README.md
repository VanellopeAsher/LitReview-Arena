# LitReview Arena

[![ICML 2026](https://img.shields.io/badge/ICML-2026-8A2D1C)](#)
[![Benchmark](https://img.shields.io/badge/benchmark-LitReviewBench-blue)](#litreviewbench)
[![Evaluator](https://img.shields.io/badge/evaluator-LitJudge-green)](#litjudge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for **LitReview Arena: Evaluating Literature Review Agents with Battle-Style Peer Review Platform**, accepted to **ICML 2026**.

LitReview Arena studies how to evaluate AI systems that generate scientific literature reviews. Instead of relying on reference overlap or generic LLM judges, the project collects blind pairwise expert preferences over literature-review drafts and converts them into a reproducible offline benchmark, **LitReviewBench**. The repository also includes **LitJudge**, an expert-aligned evaluator calibrated from benchmark preferences.

## Highlights

- **ICML 2026 accepted paper** on expert preference evaluation for literature review agents.
- **LitReviewBench**, a frozen arena-grounded benchmark with dimension-wise expert outcomes.
- **Five review-specific dimensions**: Literature Coverage, Claim Support, Paper Structure, Research Suggestions, and Overall Utility.
- **LitJudge**, an expert-aligned evaluator using structure-matched examples, content-matched examples, and diversity-aware expert gap anchors.
- **Reproducible scripts** for loading battle records, computing leaderboards, and running calibrated or naive evaluator variants.

## Repository Layout

```text
LitReview-Arena/
├── README.md
├── requirements.txt
├── data/
│   ├── battles.jsonl                    # AI-domain paired comparison records
│   ├── expert_outcomes.jsonl            # Dimension-wise expert outcomes
│   ├── topics.jsonl                     # Normalized topic queries and taxonomy tags
│   └── evaluator/
│       ├── litjudge_context_cache.jsonl
│       └── litjudge_context_cache_mmr.jsonl
├── evaluator/
│   ├── main.py                          # LitJudge / naive judge entry point
│   ├── context_builder.py               # Structure/content/gap context construction
│   ├── naive_judge.py                   # Uncalibrated judge baseline
│   ├── judge.py                         # LLM judge wrapper
│   ├── aggregator.py                    # Agreement and leaderboard aggregation
│   └── bt_model.py                      # Pairwise preference scoring utilities
├── scripts/
│   └── compute_leaderboard.py           # Compute dimension-wise leaderboards
├── examples/
│   └── simple_agent.py                  # Minimal literature review agent
└── docs/
    ├── data_format.md
    ├── evaluation_protocol.md
    └── calibration_guide.md
```

## LitReviewBench

The current public snapshot contains:

| File | Records | Description |
| --- | ---: | --- |
| `data/battles.jsonl` | 2,754 | AI-domain paired draft comparisons |
| `data/expert_outcomes.jsonl` | 2,754 | One dimension-wise expert judgment per battle |
| `data/topics.jsonl` | 925 | Normalized topic queries with field/subfield tags |

Each expert outcome uses a four-way label for each dimension:

- `A`: Draft A is preferred.
- `B`: Draft B is preferred.
- `Tie`: both drafts are comparably good.
- `BothBad`: neither draft is acceptable on that dimension.

The five dimensions follow the paper protocol:

| Dimension | Name | Question |
| --- | --- | --- |
| D1 | Literature Coverage | Which draft cites a more complete and appropriate set of relevant papers? |
| D2 | Claim Support | Which draft better grounds key claims in the cited literature? |
| D3 | Paper Structure | Which draft better organizes prior work into meaningful categories or comparisons? |
| D4 | Research Suggestions | Which draft provides more important, non-obvious, and useful future directions? |
| D5 | Overall Utility | Which draft would a researcher prefer as a starting point for a literature review? |

Detailed schemas are in [docs/data_format.md](docs/data_format.md), and the annotation protocol is in [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Load the benchmark:

```python
import json
from pathlib import Path

def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

battles = load_jsonl("data/battles.jsonl")
outcomes = load_jsonl("data/expert_outcomes.jsonl")
topics = load_jsonl("data/topics.jsonl")

print(len(battles), len(outcomes), len(topics))
```

Compute a D5 leaderboard from expert outcomes:

```bash
python scripts/compute_leaderboard.py \
  --outcomes data/expert_outcomes.jsonl \
  --battles data/battles.jsonl \
  --method elo \
  --dimension D5
```

Run the naive judge baseline on a selected subset:

```bash
python -m evaluator.main \
  --battles-file data/battles.jsonl \
  --expert-outcomes-file data/expert_outcomes.jsonl \
  --naive \
  --max-workers 4
```

Run LitJudge with diversity-aware retrieval:

```bash
python -m evaluator.main \
  --battles-file data/battles.jsonl \
  --expert-outcomes-file data/expert_outcomes.jsonl \
  --diverse-retrieval \
  --context-cache-file data/evaluator/litjudge_context_cache_mmr.jsonl \
  --max-workers 4
```

Set an API key before running evaluator calls:

```bash
export OPENAI_API_KEY=...
# or
export OPENROUTER_API_KEY=...
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

## LitJudge

LitJudge is the calibrated evaluator introduced in the paper. It builds a task-specific context for each pairwise comparison:

- **Group S, structure-similar cases**: examples retrieved by structural similarity for Paper Structure (D3).
- **Group C, content-similar cases**: topic-neighbor examples for Literature Coverage (D1) and Claim Support (D2).
- **Group G, gap anchors**: expert-written research-gap exemplars for Research Suggestions (D4), selected with diversity-aware retrieval.

The paper reports that calibration improves alignment with expert-induced leaderboards, especially on synthesis-heavy dimensions such as Paper Structure and Research Suggestions. Cross-base-model and random few-shot controls are included in the appendix and supported by the evaluator scripts.


## Citation

BibTeX will be updated after the official ICML 2026/PMLR metadata is available.

```bibtex
@inproceedings{zhao2026litreviewarena,
  title     = {LitReview Arena: Evaluating Literature Review Agents with Battle-Style Peer Review Platform},
  author    = {Zhao, Ruotong and Chen, Zhiyu and Liu, Xurui and Xue, Haidong and Liang, Dong and Fu, Jigao and Wu, Yanbiao and Zhen, Yuanyi and Xu, Fengli and Li, Yong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## Contact

For questions, open a GitHub issue or contact the corresponding author listed in the paper.

## License

This repository is released under the [MIT License](LICENSE).
