"""
成对报告：在同一候选池与同一相关分下，比较 top-k vs MMR 的校准示例冗余度（Group S / C / G）。

用于论文/实验：证明 diversity-constrained retrieval **直接**降低示例间相似度（提高 calibration diversity），
与 downstream suggestion entropy 解耦。

用法（仓库根目录）：
  python -m evaluator.calibration_diversity_report --battles-file data/battles.jsonl --results data/evaluator/outputs/evaluator_results_*.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger

from evaluator.config import (
    DEFAULT_MODEL,
    DEFAULT_PLATFORM,
    LOCAL_EMBEDDING_MODEL,
    MMR_LAMBDA,
    DIVERSITY_POOL_MULT_S,
    DIVERSITY_POOL_MULT_G,
    PAIR_SIM_MODE_DEFAULT,
)
from evaluator.context_builder import ContextBuilder
from evaluator.data_loader import load_battles


def _mean(xs: List[float]) -> Optional[float]:
    return float(statistics.mean(xs)) if xs else None


def run_report(
    battles: List[Dict],
    battle_ids: List[str],
    model_name: str = DEFAULT_MODEL,
    platform: str = DEFAULT_PLATFORM,
    mmr_lambda: float = MMR_LAMBDA,
    pair_sim_mode: str = PAIR_SIM_MODE_DEFAULT,
    pool_mult_s: int = DIVERSITY_POOL_MULT_S,
    pool_mult_g: int = DIVERSITY_POOL_MULT_G,
    embedding_model_name: str = LOCAL_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    by_id = {str(b.get("battle_id")): b for b in battles if b.get("battle_id")}
    cb = ContextBuilder(
        all_battles=battles,
        model_name=model_name,
        platform=platform,
        diversity_retrieval=False,
        mmr_lambda=mmr_lambda,
        pair_sim_mode=pair_sim_mode,
        pool_mult_s=pool_mult_s,
        pool_mult_g=pool_mult_g,
        embedding_model_name=embedding_model_name,
    )

    rows: List[Dict[str, Any]] = []
    s_drop: List[float] = []
    c_drop: List[float] = []
    g_drop: List[float] = []
    s_topk: List[float] = []
    s_mmr: List[float] = []
    c_topk: List[float] = []
    c_mmr: List[float] = []
    g_topk: List[float] = []
    g_mmr: List[float] = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    it = tqdm(battle_ids, desc="Paired top-k vs MMR") if tqdm else battle_ids
    for bid in it:
        b = by_id.get(str(bid))
        if not b:
            continue
        m = cb.paired_calibration_redundancy(b)
        m["battle_id"] = str(bid)
        rows.append(m)
        if m.get("calibration_redundancy_drop_S") is not None:
            s_drop.append(m["calibration_redundancy_drop_S"])
        if m.get("calibration_redundancy_drop_C") is not None:
            c_drop.append(m["calibration_redundancy_drop_C"])
        if m.get("calibration_redundancy_drop_G") is not None:
            g_drop.append(m["calibration_redundancy_drop_G"])
        if m.get("mean_pairwise_S_topk") is not None:
            s_topk.append(float(m["mean_pairwise_S_topk"]))
        if m.get("mean_pairwise_S_mmr") is not None:
            s_mmr.append(float(m["mean_pairwise_S_mmr"]))
        if m.get("mean_pairwise_C_topk") is not None:
            c_topk.append(float(m["mean_pairwise_C_topk"]))
        if m.get("mean_pairwise_C_mmr") is not None:
            c_mmr.append(float(m["mean_pairwise_C_mmr"]))
        if m.get("mean_pairwise_G_topk") is not None:
            g_topk.append(float(m["mean_pairwise_G_topk"]))
        if m.get("mean_pairwise_G_mmr") is not None:
            g_mmr.append(float(m["mean_pairwise_G_mmr"]))

    summary = {
        "n_battles": len(rows),
        "mean_mean_pairwise_S_topk": _mean(s_topk),
        "mean_mean_pairwise_S_mmr": _mean(s_mmr),
        "mean_mean_pairwise_C_topk": _mean(c_topk),
        "mean_mean_pairwise_C_mmr": _mean(c_mmr),
        "mean_mean_pairwise_G_topk": _mean(g_topk),
        "mean_mean_pairwise_G_mmr": _mean(g_mmr),
        "mean_calibration_redundancy_drop_S": _mean(s_drop),
        "mean_calibration_redundancy_drop_C": _mean(c_drop),
        "mean_calibration_redundancy_drop_G": _mean(g_drop),
        "fraction_battles_mmr_more_diverse_S": (
            sum(1 for x in s_drop if x > 1e-9) / len(s_drop) if s_drop else None
        ),
        "fraction_battles_mmr_more_diverse_C": (
            sum(1 for x in c_drop if x > 1e-9) / len(c_drop) if c_drop else None
        ),
        "fraction_battles_mmr_more_diverse_G": (
            sum(1 for x in g_drop if x > 1e-9) / len(g_drop) if g_drop else None
        ),
        "interpretation": (
            "calibration_redundancy_drop_* = mean_pairwise_topk - mean_pairwise_mmr; "
            "positive values mean MMR selected less redundant (more diverse) example sets."
        ),
        "mmr_lambda": mmr_lambda,
        "pool_mult_s": pool_mult_s,
        "pool_mult_g": pool_mult_g,
        "pair_sim_mode": pair_sim_mode,
        "embedding_model_for_pair_sim_c": embedding_model_name,
        "model_name": model_name,
        "platform": platform,
    }

    return {"summary": summary, "per_battle": rows}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Paired top-k vs MMR calibration redundancy (same relevance, fair comparison)."
    )
    p.add_argument("--battles-file", type=Path, required=True)
    p.add_argument(
        "--results",
        type=Path,
        help="evaluator_results JSON; battle IDs = judge_outcomes keys (recommended)",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Write JSON report (default: stdout path sibling)",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--platform", type=str, default=DEFAULT_PLATFORM)
    p.add_argument("--mmr-lambda", type=float, default=MMR_LAMBDA)
    p.add_argument("--pair-sim-mode", type=str, default=PAIR_SIM_MODE_DEFAULT, choices=["embedding", "jaccard"])
    p.add_argument("--pool-mult-s", type=int, default=DIVERSITY_POOL_MULT_S)
    p.add_argument("--pool-mult-g", type=int, default=DIVERSITY_POOL_MULT_G)
    p.add_argument("--embedding-model", type=str, default=LOCAL_EMBEDDING_MODEL)
    args = p.parse_args()

    battles = load_battles(args.battles_file)
    if args.results and args.results.exists():
        with open(args.results, "r", encoding="utf-8") as f:
            data = json.load(f)
        jo = data.get("judge_outcomes") or {}
        battle_ids = list(jo.keys())
        logger.info(f"Using {len(battle_ids)} battle IDs from results file")
    else:
        battle_ids = [str(b.get("battle_id")) for b in battles if b.get("battle_id")]
        logger.info(f"Using all {len(battle_ids)} battle IDs from battles file")

    out = run_report(
        battles,
        battle_ids,
        model_name=args.model,
        platform=args.platform,
        mmr_lambda=args.mmr_lambda,
        pair_sim_mode=args.pair_sim_mode,
        pool_mult_s=args.pool_mult_s,
        pool_mult_g=args.pool_mult_g,
        embedding_model_name=args.embedding_model,
    )

    out_path = args.output
    if not out_path:
        out_path = (
            args.results.with_name(args.results.stem + "_calibration_paired.json")
            if args.results
            else Path("calibration_paired_report.json")
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {out_path}")
    s = out["summary"]
    logger.info(
        f"Mean redundancy drop S={s.get('mean_calibration_redundancy_drop_S')}, "
        f"C={s.get('mean_calibration_redundancy_drop_C')}, "
        f"G={s.get('mean_calibration_redundancy_drop_G')} "
        f"(positive => MMR more diverse than top-k)"
    )


if __name__ == "__main__":
    main()
