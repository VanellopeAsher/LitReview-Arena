"""
Compute leaderboard from expert outcomes using Bradley-Terry or Elo methods.
"""

import json
import argparse
from collections import defaultdict
from typing import Optional


def load_outcomes(outcomes_file: str) -> list:
    """Load expert outcomes from JSONL file."""
    outcomes = []
    with open(outcomes_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                outcomes.append(json.loads(line))
    return outcomes


def load_battles(battles_file: str) -> list:
    """Load battle records from JSONL file."""
    battles = []
    with open(battles_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                battles.append(json.loads(line))
    return battles


def _battle_index(battles: list) -> dict:
    return {battle["battle_id"]: battle for battle in battles}


def _systems_for_battle(battle: dict) -> tuple[str, str]:
    draft_a = battle.get("draft_a") or {}
    draft_b = battle.get("draft_b") or {}
    return draft_a["system_id"], draft_b["system_id"]


def _compute_win_rate_scores(outcomes: list, battles: list, dimension: str) -> dict:
    battles_by_id = _battle_index(battles)
    wins = defaultdict(float)
    losses = defaultdict(float)
    seen = set()

    for outcome in outcomes:
        battle = battles_by_id.get(outcome["battle_id"])
        if not battle:
            continue
        system_a, system_b = _systems_for_battle(battle)
        seen.update([system_a, system_b])
        vote = outcome.get("outcomes", {}).get(dimension)
        if vote == "A":
            wins[system_a] += 1.0
            losses[system_b] += 1.0
        elif vote == "B":
            wins[system_b] += 1.0
            losses[system_a] += 1.0

    return {
        system_id: (wins[system_id] + 0.5) / (wins[system_id] + losses[system_id] + 1.0)
        for system_id in seen
    }


def _compute_elo_scores(outcomes: list, battles: list, dimension: str) -> dict:
    battles_by_id = _battle_index(battles)
    ratings = defaultdict(lambda: 1500.0)
    k_factor = 32.0

    for outcome in outcomes:
        battle = battles_by_id.get(outcome["battle_id"])
        if not battle:
            continue
        system_a, system_b = _systems_for_battle(battle)
        vote = outcome.get("outcomes", {}).get(dimension)
        if vote == "A":
            score_a = 1.0
        elif vote == "B":
            score_a = 0.0
        elif vote == "Tie":
            score_a = 0.5
        else:
            continue

        rating_a = ratings[system_a]
        rating_b = ratings[system_b]
        expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
        expected_b = 1.0 - expected_a
        ratings[system_a] = rating_a + k_factor * (score_a - expected_a)
        ratings[system_b] = rating_b + k_factor * ((1.0 - score_a) - expected_b)

    return dict(ratings)


def compute_leaderboard(
    outcomes_file: str,
    battles_file: str,
    method: str = "bradley_terry",
    dimension: str = "D5",
    output_file: Optional[str] = None
):
    """
    Compute leaderboard from expert outcomes.

    Args:
        outcomes_file: Path to expert outcomes JSONL file
        battles_file: Path to battles JSONL file (for system IDs)
        method: Aggregation method ("bradley_terry" or "elo")
        dimension: Evaluation dimension (D1-D5)
        output_file: Optional output file path
    """
    print(f"Loading outcomes from {outcomes_file}...")
    outcomes = load_outcomes(outcomes_file)
    print(f"Loaded {len(outcomes)} outcome records")

    print(f"Loading battles from {battles_file}...")
    battles = load_battles(battles_file)
    print(f"Loaded {len(battles)} battle records")

    if method == "bradley_terry":
        print(f"Computing Bradley-Terry scores for {dimension}...")
        scores = _compute_win_rate_scores(outcomes, battles, dimension)
    elif method == "elo":
        print(f"Computing Elo ratings for {dimension}...")
        scores = _compute_elo_scores(outcomes, battles, dimension)
    else:
        raise ValueError(f"Unknown method: {method}")

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{'='*60}")
    print(f"Leaderboard for {dimension} ({method})")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'System':<40} {'Score':<15}")
    print(f"{'-'*60}")

    for rank, (system_id, score) in enumerate(sorted_scores, 1):
        print(f"{rank:<6} {system_id:<40} {score:<15.2f}")

    if output_file:
        leaderboard = [
            {
                "rank": rank,
                "system_id": system_id,
                "score": score,
                "dimension": dimension,
                "method": method
            }
            for rank, (system_id, score) in enumerate(sorted_scores, 1)
        ]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard, f, indent=2, ensure_ascii=False)
        print(f"\nLeaderboard saved to {output_file}")

    return sorted_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute leaderboard from expert outcomes"
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        default="data/expert_outcomes.jsonl",
        help="Path to expert outcomes JSONL file"
    )
    parser.add_argument(
        "--battles",
        type=str,
        default="data/battles.jsonl",
        help="Path to battles JSONL file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bradley_terry", "elo"],
        default="bradley_terry",
        help="Aggregation method"
    )
    parser.add_argument(
        "--dimension",
        type=str,
        choices=["D1", "D2", "D3", "D4", "D5"],
        default="D5",
        help="Evaluation dimension"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional)"
    )

    args = parser.parse_args()

    compute_leaderboard(
        outcomes_file=args.outcomes,
        battles_file=args.battles,
        method=args.method,
        dimension=args.dimension,
        output_file=args.output
    )
