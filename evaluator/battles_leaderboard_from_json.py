"""
从 battles JSON 数组（如 battles_300_litjudge.json）构建 Elo 与 Bradley–Terry 风格排行榜。

与仓库 docs/data_format.md / evaluator 聚合逻辑一致：
- **Bradley–Terry（BLO）**：仅 decisive A/B；Tie/BothBad 不参与（与 bt_model 一致）。
- **Elo（ET）**：init=1500, K=32；Tie/BothBad 各计 0.5 胜（与 data_format 描述一致）。

默认读取 `metadata.judge_outcomes` 的 D1–D5（LitJudge 输出）。
若某维缺失且存在 `metadata.arena_votes`，则用 arena 键补齐：**d0_overall_utility → D5**，d1–d4 → D1–D4（与 `arena_litjudge_fill.LITJUDGE_TO_ARENA_VOTES` 一致）；同维以 `judge_outcomes` 为准。

用法：
  python evaluator/battles_leaderboard_from_json.py --input battles_300_litjudge.json --output battles_300_leaderboard.json
  python evaluator/battles_leaderboard_from_json.py --input data/battles.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.append(str(_EVAL))

try:
    from .config import DIMENSIONS, NEUTRAL_OUTCOMES, OUTCOMES
    from .bt_model import BradleyTerryModel
except ImportError:
    from evaluator.config import DIMENSIONS, NEUTRAL_OUTCOMES, OUTCOMES
    from evaluator.bt_model import BradleyTerryModel

ELO_INIT = 1500.0
ELO_K = 32
ELO_ROUNDS = 30

_VALID_OUTCOMES = frozenset(OUTCOMES)

# Arena export vote keys -> LitJudge dimension（d0 = Overall 对应 D5）
ARENA_VOTE_KEY_TO_LITJUDGE_DIM: Dict[str, str] = {
    "d0_overall_utility": "D5",
    "d1_literature_coverage": "D1",
    "d2_claim_support": "D2",
    "d3_paper_structure": "D3",
    "d4_research_suggestions": "D4",
}


def get_judge_outcomes(battle: Dict[str, Any]) -> Optional[Dict[str, str]]:
    m = battle.get("metadata")
    if not isinstance(m, dict):
        return None
    jo = m.get("judge_outcomes")
    return jo if isinstance(jo, dict) else None


def _outcomes_from_arena_votes(metadata: Dict[str, Any]) -> Dict[str, str]:
    av = metadata.get("arena_votes")
    if not isinstance(av, dict):
        return {}
    out: Dict[str, str] = {}
    for av_key, dim in ARENA_VOTE_KEY_TO_LITJUDGE_DIM.items():
        v = av.get(av_key)
        if isinstance(v, str) and v in _VALID_OUTCOMES:
            out[dim] = v
    return out


def resolve_judge_outcomes(battle: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    合并 judge_outcomes 与 arena_votes：先填 arena（含 d0→D5），再用 judge_outcomes 覆盖同维。
    """
    m = battle.get("metadata")
    if not isinstance(m, dict):
        m = {}
    jo = m.get("judge_outcomes")
    merged: Dict[str, str] = {}
    merged.update(_outcomes_from_arena_votes(m))
    if isinstance(jo, dict):
        merged.update({k: v for k, v in jo.items() if isinstance(v, str)})
    return merged if merged else None


def agent_keys(battle: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    da = battle.get("draft_a") or {}
    db = battle.get("draft_b") or {}
    return da.get("system_id"), db.get("system_id")


def build_bt_comparisons(
    battles: List[Dict],
    dimension: str,
) -> List[Tuple[int, int, str, float]]:
    """与 aggregator.compute_leaderboard_correlation 中 judge 分支一致。"""
    keys_sorted = sorted(
        {k for b in battles for k in agent_keys(b) if k and isinstance(k, str)}
    )
    key_to_id = {k: i for i, k in enumerate(keys_sorted)}
    out: List[Tuple[int, int, str, float]] = []

    for battle in battles:
        jo = resolve_judge_outcomes(battle)
        if not jo:
            continue
        o = jo.get(dimension)
        if not o:
            continue
        a_key, b_key = agent_keys(battle)
        if not a_key or not b_key:
            continue
        if a_key not in key_to_id or b_key not in key_to_id:
            continue
        a_id = key_to_id[a_key]
        b_id = key_to_id[b_key]

        if o == "A":
            out.append((a_id, b_id, "A", 1.0))
        elif o == "B":
            out.append((b_id, a_id, "A", 1.0))
        elif o in NEUTRAL_OUTCOMES:
            out.append((a_id, b_id, "Tie", 1.0))
    return out


def _expected(r_a: float, r_b: float) -> Tuple[float, float]:
    ea = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    return ea, 1.0 - ea


def compute_elo(
    battles: List[Dict],
    dimension: str,
    init: float = ELO_INIT,
    k: float = ELO_K,
    rounds: int = ELO_ROUNDS,
) -> Dict[str, float]:
    """多轮遍历 battles，对每场更新 Elo；Tie/BothBad 为 0.5 分。"""
    models = sorted(
        {x for b in battles for x in agent_keys(b) if x and isinstance(x, str)}
    )
    r: Dict[str, float] = {m: float(init) for m in models}

    # 稳定顺序：按 battle_id
    ordered = sorted(
        [b for b in battles if b.get("battle_id")],
        key=lambda x: str(x.get("battle_id")),
    )

    for _ in range(rounds):
        for battle in ordered:
            jo = resolve_judge_outcomes(battle)
            if not jo:
                continue
            o = jo.get(dimension)
            if not o:
                continue
            a_key, b_key = agent_keys(battle)
            if not a_key or not b_key or a_key not in r or b_key not in r:
                continue

            ra, rb = r[a_key], r[b_key]
            ea, eb = _expected(ra, rb)

            if o == "A":
                sa, sb = 1.0, 0.0
            elif o == "B":
                sa, sb = 0.0, 1.0
            elif o in NEUTRAL_OUTCOMES:
                sa, sb = 0.5, 0.5
            else:
                continue

            r[a_key] = ra + k * (sa - ea)
            r[b_key] = rb + k * (sb - eb)

    return r


def bt_scores_to_leaderboard(
    model: BradleyTerryModel,
    id_to_key: Dict[int, str],
) -> List[Dict[str, Any]]:
    if model.beta is None or model.model_ids is None or len(model.beta) == 0:
        return []
    rows = []
    for idx, mid in enumerate(model.model_ids):
        key = id_to_key.get(int(mid))
        if key is None:
            continue
        rows.append({"system_id": key, "score": float(model.beta[idx])})
    rows.sort(key=lambda x: (-x["score"], x["system_id"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def elo_to_leaderboard(ratings: Dict[str, float]) -> List[Dict[str, Any]]:
    items = sorted(ratings.items(), key=lambda x: (-x[1], x[0]))
    return [
        {"rank": i + 1, "system_id": sid, "rating": round(r, 4)}
        for i, (sid, r) in enumerate(items)
    ]


def run_report(
    battles: List[Dict],
) -> Dict[str, Any]:
    keys_sorted = sorted(
        {k for b in battles for k in agent_keys(b) if k and isinstance(k, str)}
    )
    key_to_id = {k: i for i, k in enumerate(keys_sorted)}
    id_to_key = {i: k for k, i in key_to_id.items()}

    by_dim: Dict[str, Any] = {}
    for dim in DIMENSIONS.keys():
        comps = build_bt_comparisons(battles, dim)
        bt = BradleyTerryModel()
        bt.fit(comps)
        # 仅 id 在 bt.model_ids 中的映射
        blo = bt_scores_to_leaderboard(bt, id_to_key)

        elo = compute_elo(battles, dim)
        et = elo_to_leaderboard(elo)

        by_dim[dim] = {
            "bradley_terry_leaderboard_blo": blo,
            "elo_leaderboard_et": et,
            "n_pairwise_comparisons_bt": sum(
                1
                for c in comps
                if c[2] != "Tie"
            ),
            "n_battles_with_outcome": sum(
                1 for b in battles if (resolve_judge_outcomes(b) or {}).get(dim)
            ),
        }

    return {
        "by_dimension": by_dim,
        "n_battles": len(battles),
        "n_models": len(keys_sorted),
        "outcome_resolution": (
            "metadata.judge_outcomes merged with metadata.arena_votes "
            "(d0_overall_utility→D5, d1–d4→D1–D4); judge_outcomes overrides per key"
        ),
        "methods": {
            "bradley_terry_blo": "Win-rate BT proxy (Laplace smoothing) with ties excluded from wins; see evaluator/bt_model.py",
            "elo_et": f"init={ELO_INIT}, K={ELO_K}, rounds={ELO_ROUNDS}; Tie/BothBad = 0.5 win each",
        },
    }


def load_battles_input(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON input must be an array of battles")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Elo + BLO leaderboards from battles JSON array")
    p.add_argument("--input", type=Path, required=True, help="battles JSON array or .jsonl")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    data = load_battles_input(args.input)

    report = run_report(data)
    report["source_file"] = str(args.input.resolve())

    out = args.output or args.input.with_name(args.input.stem + "_leaderboard.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
