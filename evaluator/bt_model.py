"""
Bradley–Terry–style scores for leaderboard correlation in aggregator.

`fit` accepts comparisons as (winner_id, loser_id, "A", weight); Tie is skipped.
Uses win-rate scores (rank-preserving enough for Spearman vs. expert/judge).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np


class BradleyTerryModel:
    def __init__(self) -> None:
        self.beta = None
        self.model_ids = None

    def fit(self, comparisons: List[Tuple[int, int, str, float]]) -> None:
        wins = defaultdict(float)
        losses = defaultdict(float)
        seen = set()

        for w, ell, tag, wt in comparisons:
            seen.add(w)
            seen.add(ell)
            if tag == "Tie":
                continue
            wins[w] += float(wt)
            losses[ell] += float(wt)

        ids = sorted(seen)
        if not ids:
            self.beta = np.array([])
            self.model_ids = np.array([])
            return

        self.model_ids = np.array(ids, dtype=int)
        # Score = wins / (wins + losses) with Laplace smoothing
        self.beta = np.array(
            [
                (wins[i] + 0.5) / max(wins[i] + losses[i] + 1.0, 1e-9)
                for i in ids
            ],
            dtype=float,
        )
