# LitReview Arena

<p align="center">
  <strong>LitReview Arena: Evaluating Literature Review Agents with Battle-Style Peer Review Platform</strong>
</p>

<p align="center">
  <a href="https://icml.cc/Conferences/2026"><img src="https://img.shields.io/badge/ICML-2026-8A2D1C" alt="ICML 2026"></a>
  <a href="#litreviewbench"><img src="https://img.shields.io/badge/Benchmark-LitReviewBench-2563eb" alt="LitReviewBench"></a>
  <a href="#litjudge"><img src="https://img.shields.io/badge/Evaluator-LitJudge-15803d" alt="LitJudge"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

<p align="center">
  <strong>Accepted to ICML 2026</strong>
</p>

<p align="center">
  Ruotong Zhao<sup>1</sup> · Zhiyu Chen<sup>2</sup> · Xurui Liu<sup>1</sup> · Haidong Xue<sup>3</sup> · Dong Liang<sup>4</sup> · Jigao Fu<sup>3</sup> · Yanbiao Wu<sup>5</sup> · Yuanyi Zhen<sup>2</sup> · Fengli Xu<sup>1</sup> · Yong Li<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Tsinghua University &nbsp;
  <sup>2</sup>Zhongguancun Academy &nbsp;
  <sup>3</sup>Zhongguancun Institute of AI &nbsp;
  <sup>4</sup>Huazhong University of Science and Technology &nbsp;
  <sup>5</sup>Shanghai Institute of Microsystem and Information Technology
</p>

---

LitReview Arena studies how to evaluate AI systems that generate scientific literature reviews. Instead of relying on reference overlap or generic LLM judges, we collect blind pairwise expert preferences over literature-review drafts and convert them into a reproducible offline benchmark, **LitReviewBench**. The repository also includes **LitJudge**, an expert-aligned evaluator calibrated from benchmark preferences.

## News

- **ICML 2026**: LitReview Arena was accepted to the 43rd International Conference on Machine Learning.
- **Public artifact**: This repository releases the benchmark data, evaluation protocol, leaderboard scripts, and LitJudge implementation.

## Overview

| Component           | Description                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| **LitReview Arena** | Battle-style expert evaluation platform for literature review agents.                                           |
| **LitReviewBench**  | Frozen benchmark distilled from arena logs with dimension-wise expert outcomes.                                 |
| **LitJudge**        | Expert-aligned evaluator using structure-matched cases, content-matched cases, and diversity-aware gap anchors. |

The evaluation protocol uses five literature-review-specific dimensions:

| Dimension | Name                 | Question                                                                           |
| --------- | -------------------- | ---------------------------------------------------------------------------------- |
| D1        | Literature Coverage  | Which draft cites a more complete and appropriate set of relevant papers?          |
| D2        | Claim Support        | Which draft better grounds key claims in the cited literature?                     |
| D3        | Paper Structure      | Which draft better organizes prior work into meaningful categories or comparisons? |
| D4        | Research Suggestions | Which draft provides more important, non-obvious, and useful future directions?    |
| D5        | Overall Utility      | Which draft would a researcher prefer as a starting point for a literature review? |

## LitReviewBench

The current public snapshot contains:

| File                         | Records | Description                                       |
| ---------------------------- | -------:| ------------------------------------------------- |
| `data/battles.jsonl`         | 2,754   | AI-domain paired draft comparisons                |
| `data/expert_outcomes.jsonl` | 2,754   | One dimension-wise expert judgment per battle     |
| `data/topics.jsonl`          | 925     | Normalized topic queries with field/subfield tags |

Each expert outcome uses a four-way label for each dimension:

- `A`: Draft A is preferred.
- `B`: Draft B is preferred.
- `Tie`: both drafts are comparably good.
- `BothBad`: neither draft is acceptable on that dimension.

Detailed schemas are in [docs/data_format.md](docs/data_format.md), and the annotation protocol is in [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Repository Layout

```text
LitReview-Arena/
├── README.md
├── requirements.txt
├── data/
│   ├── battles.jsonl
│   ├── expert_outcomes.jsonl
│   ├── topics.jsonl
│   └── evaluator/
│       ├── litjudge_context_cache.jsonl
│       └── litjudge_context_cache_mmr.jsonl
├── evaluator/
│   ├── main.py
│   ├── context_builder.py
│   ├── naive_judge.py
│   ├── judge.py
│   ├── aggregator.py
│   └── bt_model.py
├── scripts/
│   └── compute_leaderboard.py
├── examples/
│   └── simple_agent.py
└── docs/
    ├── data_format.md
    ├── evaluation_protocol.md
    └── calibration_guide.md
```

## Citation

```bibtex
@inproceedings{zhao2026litreviewarena,
  title     = {LitReview Arena: Evaluating Literature Review Agents with Battle-Style Peer Review Platform},
  author    = {Zhao, Ruotong and Chen, Zhiyu and Liu, Xurui and Xue, Haidong and Liang, Dong and Fu, Jigao and Wu, Yanbiao and Zhen, Yuanyi and Xu, Fengli and Li, Yong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## Contact

For questions, please open a GitHub issue or contact the corresponding author listed in the paper.

## License

This repository is released under the [MIT License](LICENSE).
