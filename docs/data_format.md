# Data Format Specification

## Battle Record Format

Each battle record in `battles.jsonl` follows this structure.

```json
{
  "battle_id": "unique_battle_identifier",
  "topic_id": "topic_identifier",
  "topic_query": "Conduct a literature review on {Topic}",
  "draft_a": {
    "system_id": "system_identifier_a",
    "system_name": "Human-readable system name",
    "content": "Full text of draft A"
  },
  "draft_b": {
    "system_id": "system_identifier_b",
    "system_name": "Human-readable system name",
    "content": "Full text of draft B"
  },
  "metadata": {
    "field": "AI field tag",
    "subfield": "AI subfield tag",
    "created_at": "ISO 8601 timestamp",
    "order_randomized": true
  }
}
```

## Expert Outcome Format

Each expert outcome in `expert_outcomes.jsonl` follows this structure. Each battle has one expert judgment with five dimension-wise outcomes.

```json
{
  "battle_id": "battle_identifier",
  "annotator_id": "pseudonymized_annotator_id",
  "outcomes": {
    "D1": "A|B|Tie|BothBad",
    "D2": "A|B|Tie|BothBad",
    "D3": "A|B|Tie|BothBad",
    "D4": "A|B|Tie|BothBad",
    "D5": "A|B|Tie|BothBad"
  },
  "metadata": {
    "field_match": true,
    "subfield_match": true,
    "timestamp": "ISO 8601 timestamp"
  }
}
```

## Topic Format

Each topic in `topics.jsonl` follows this structure.

```json
{
  "topic_id": "unique_topic_identifier",
  "topic_query": "Conduct a literature review on {Topic}",
  "topic_phrase": "Extracted topic phrase",
  "source_paper": {
    "openalex_id": "OpenAlex work ID",
    "title": "Source survey paper title",
    "year": 2023,
    "citation_count": 150
  },
  "taxonomy": {
    "field": "AI field tag",
    "subfield": "AI subfield tag"
  },
  "normalization": {
    "original_topic": "Original extracted topic",
    "normalized_topic": "Normalized topic phrase",
    "normalization_notes": "Notes on normalization process"
  }
}
```

## Outcome Values

- **A**: Draft A is preferred
- **B**: Draft B is preferred  
- **Tie**: Both drafts are equally good
- **Both Bad**: Neither draft meets minimum quality standards

## Aggregated Scores

The benchmark provides aggregated scores (paper Section 4.1):

1. **Bradley-Terry Model**: Produces preference probabilities. Tie and BothBad are excluded; only decisive outcomes (A/B) contribute.
2. **Elo Rating**: Produces numerical ratings (init=1500, K=32). Tie and BothBad are treated as 0.5 wins each.

Both methods are applied per dimension (D1-D5) to produce dimension-specific leaderboards.
