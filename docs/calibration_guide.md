# LitJudge Calibration Guide (Paper Section 6)

## Overview

LitJudge improves alignment with expert preferences by using structure- and topic-matched in-context examples plus human-written gap anchors.

## Architecture

LitJudge uses three groups of demonstrations (paper Section 6.1, Figure 2a):

### Group S (Structure-similar; for D3)

- Extracts skeleton text (headers and lead sentences)
- Derives paragraph relationship network capturing discourse transitions
- Retrieves battles with most similar networks based on normalized graph similarity

### Group C (Content-similar; for D1/D2)

- Paper uses LLM-based embedding matching. This implementation uses Jaccard similarity on topic_query as a lightweight alternative.
- Provides local standards for sufficient coverage and plausible citation–claim support

### Group G (Gap anchors; for D4)

- Extracts gap anchors exclusively from expert-written reviews (lit-review_human)
- Bullet points serve as grounding exemplars for high-quality, non-hallucinated directions

## Usage

```bash
python -m evaluator.main \
  --battles-file data/battles.jsonl \
  --expert-outcomes-file data/expert_outcomes.jsonl \
  --limit 5
```

## Performance (Paper Figure 2b)

- **D1 (Literature Coverage)**: ρ ≈ 0.58 (up from ≈ 0.55)
- **D2 (Claim Support)**: ρ ≈ 0.67 (up from ≈ 0.44)
- **D3 (Paper Structure)**: ρ ≈ 0.65 (up from ≈ 0.47)
- **D4 (Research Suggestions)**: ρ ≈ 0.84 (up from ≈ 0.43)
- **D5 (Overall Utility)**: ρ ≈ 0.79 (up from ≈ 0.47)

## Generalization (Paper Figure 2c)

Held-out subfield evaluation (20%): ρ = 0.72 (D5), 0.68 (D3), 0.66 (D4).
