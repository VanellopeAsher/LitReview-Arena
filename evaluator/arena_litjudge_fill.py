"""
用 LitJudge（或 Naive）对 xinyang arena 导出 JSON 逐条判别，仅更新每条结果的 votes（不增删其它字段）。

映射：D5 -> d0_overall_utility，D1–D4 -> d1–d4（与 LitJudge 维度一致）。

用法（仓库根目录）：
  python evaluator/arena_litjudge_fill.py --glob-dir .                    # 处理目录下所有 *_arena_annotations.json
  python evaluator/arena_litjudge_fill.py --input xinyang_arena_annotations.json
  python evaluator/arena_litjudge_fill.py --input xinyang_arena_annotations.json --naive

默认输出：foo_arena_annotations.json -> foo_arena_litjudge.json（同目录）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.append(str(_EVAL))

from loguru import logger

# LitJudge D1–D5 -> arena vote keys（d0 对应 Overall = D5）
LITJUDGE_TO_ARENA_VOTES: Dict[str, str] = {
    "D5": "d0_overall_utility",
    "D1": "d1_literature_coverage",
    "D2": "d2_claim_support",
    "D3": "d3_paper_structure",
    "D4": "d4_research_suggestions",
}


def default_litjudge_output_path(input_path: Path) -> Path:
    """foo_arena_annotations.json -> foo_arena_litjudge.json"""
    name = input_path.name
    if name.endswith("_arena_annotations.json"):
        return input_path.with_name(
            name.replace("_arena_annotations.json", "_arena_litjudge.json", 1)
        )
    return input_path.with_name(input_path.stem + "_litjudge.json")


def arena_result_to_battle(row: Dict[str, Any]) -> Dict[str, Any]:
    sid = row.get("sample_id") or row.get("id")
    if not sid:
        raise ValueError("result 缺少 sample_id")
    prompt = (row.get("prompt") or "").strip()
    ra = row.get("response_a") or ""
    rb = row.get("response_b") or ""
    return {
        "battle_id": str(sid),
        "topic_query": prompt,
        "draft_a": {
            "system_id": "arena_response_a",
            "system_name": row.get("left_label") or "Response A",
            "content": ra,
        },
        "draft_b": {
            "system_id": "arena_response_b",
            "system_name": row.get("right_label") or "Response B",
            "content": rb,
        },
        "metadata": {"subfield": "unknown"},
    }


def load_base_battles(battles_file: Path) -> List[Dict]:
    try:
        from .data_loader import load_battles
    except ImportError:
        from evaluator.data_loader import load_battles
    return load_battles(battles_file)


def build_evaluator(
    all_battles: List[Dict],
    naive: bool,
    model_name: str,
    platform: str,
    diverse_retrieval: bool,
    mmr_lambda: float,
    pair_sim_mode: str,
    pool_mult_s: int,
    pool_mult_g: int,
    embedding_model: str,
    expert_outcomes: Optional[Dict[str, Dict[str, str]]],
):
    try:
        from .config import (
            MMR_LAMBDA,
            DIVERSITY_POOL_MULT_S,
            DIVERSITY_POOL_MULT_G,
            PAIR_SIM_MODE_DEFAULT,
            LOCAL_EMBEDDING_MODEL,
        )
        from .context_builder import ContextBuilder
        from .naive_judge import NaiveContextBuilder
        from .judge import Evaluator
    except ImportError:
        from evaluator.config import (
            MMR_LAMBDA,
            DIVERSITY_POOL_MULT_S,
            DIVERSITY_POOL_MULT_G,
            PAIR_SIM_MODE_DEFAULT,
            LOCAL_EMBEDDING_MODEL,
        )
        from evaluator.context_builder import ContextBuilder
        from evaluator.naive_judge import NaiveContextBuilder
        from evaluator.judge import Evaluator

    if naive:
        cb = NaiveContextBuilder(all_battles=all_battles, model_name=model_name, platform=platform)
    else:
        cb = ContextBuilder(
            all_battles=all_battles,
            model_name=model_name,
            platform=platform,
            diversity_retrieval=diverse_retrieval,
            mmr_lambda=mmr_lambda or MMR_LAMBDA,
            pair_sim_mode=pair_sim_mode or PAIR_SIM_MODE_DEFAULT,
            pool_mult_s=pool_mult_s or DIVERSITY_POOL_MULT_S,
            pool_mult_g=pool_mult_g or DIVERSITY_POOL_MULT_G,
            embedding_model_name=embedding_model or LOCAL_EMBEDDING_MODEL,
        )
    return Evaluator(
        context_builder=cb,
        model_name=model_name,
        platform=platform,
        expert_outcomes_map=expert_outcomes,
    )


def outcome_to_votes(out: Dict[str, str]) -> Dict[str, str]:
    votes = {
        "d0_overall_utility": "",
        "d1_literature_coverage": "",
        "d2_claim_support": "",
        "d3_paper_structure": "",
        "d4_research_suggestions": "",
    }
    for d_lit, k_arena in LITJUDGE_TO_ARENA_VOTES.items():
        v = out.get(d_lit)
        if v is not None:
            votes[k_arena] = str(v)
    return votes


def run(
    input_path: Path,
    output_path: Path,
    battles_file: Path,
    naive: bool,
    model_name: str,
    platform: str,
    diverse_retrieval: bool,
    mmr_lambda: float,
    pair_sim_mode: str,
    pool_mult_s: int,
    pool_mult_g: int,
    embedding_model: str,
    expert_outcomes_file: Optional[Path],
) -> None:
    try:
        from .data_loader import load_expert_outcomes_jsonl
    except ImportError:
        from evaluator.data_loader import load_expert_outcomes_jsonl

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("输入 JSON 需包含 results 数组")

    base = load_base_battles(battles_file)
    arena_battles = []
    for row in results:
        try:
            arena_battles.append(arena_result_to_battle(row))
        except Exception as e:
            logger.warning(f"跳过无效条目: {e}")

    if not arena_battles:
        raise ValueError("没有可转换的 arena 样本")

    all_battles = base + arena_battles
    logger.info(f"基础 battles: {len(base)}，arena 条数: {len(arena_battles)}，合并后: {len(all_battles)}")

    expert_path = expert_outcomes_file
    eo = load_expert_outcomes_jsonl(Path(expert_path)) if expert_path else load_expert_outcomes_jsonl()

    ev = build_evaluator(
        all_battles,
        naive=naive,
        model_name=model_name,
        platform=platform,
        diverse_retrieval=diverse_retrieval,
        mmr_lambda=mmr_lambda,
        pair_sim_mode=pair_sim_mode,
        pool_mult_s=pool_mult_s,
        pool_mult_g=pool_mult_g,
        embedding_model=embedding_model,
        expert_outcomes=eo,
    )

    by_id = {b["battle_id"]: b for b in arena_battles}

    for i, row in enumerate(results):
        sid = row.get("sample_id")
        if not sid or str(sid) not in by_id:
            continue
        battle = by_id[str(sid)]
        logger.info(f"[{i + 1}/{len(results)}] LitJudge: {sid}")
        rec, _stats = ev.evaluate_battle(battle)
        if not rec:
            logger.warning(f"判别失败，跳过写入 votes: {sid}")
            continue
        out = {k: rec[k] for k in ("D1", "D2", "D3", "D4", "D5") if k in rec}
        # 仅更新 votes，不增删该条目的其它字段
        row["votes"] = outcome_to_votes(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"已写入: {output_path}")


def main() -> None:
    try:
        from .config import DEFAULT_MODEL, DEFAULT_PLATFORM, BATTLES_FILE
    except ImportError:
        from evaluator.config import DEFAULT_MODEL, DEFAULT_PLATFORM, BATTLES_FILE

    p = argparse.ArgumentParser(description="LitJudge 填充 arena JSON 的 votes")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="单个 *_arena_annotations.json；省略则处理 --glob-dir 下所有匹配文件",
    )
    p.add_argument(
        "--glob-dir",
        type=Path,
        default=Path("."),
        help="未指定 --input 时，在此目录下匹配 *_arena_annotations.json（默认当前目录）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="仅在与 --input 同用且只处理一个文件时生效；否则为每个输入生成 *_arena_litjudge.json",
    )
    p.add_argument("--battles-file", type=Path, default=BATTLES_FILE)
    p.add_argument("--naive", action="store_true", help="不做检索，仅 rubric + 当前对局")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--platform", type=str, default=DEFAULT_PLATFORM)
    p.add_argument("--diverse-retrieval", action="store_true")
    p.add_argument("--mmr-lambda", type=float, default=0.7)
    p.add_argument("--pair-sim-mode", type=str, default="embedding")
    p.add_argument("--pool-mult-s", type=int, default=5)
    p.add_argument("--pool-mult-g", type=int, default=5)
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--expert-outcomes-file", type=Path, default=None)
    args = p.parse_args()

    if args.input is not None:
        input_paths = [args.input.resolve()]
    else:
        gdir = args.glob_dir.resolve()
        input_paths = sorted(gdir.glob("*_arena_annotations.json"))
        if not input_paths:
            logger.error(f"未找到匹配文件: {gdir / '*_arena_annotations.json'}")
            sys.exit(1)
        logger.info(f"将处理 {len(input_paths)} 个文件: {[p.name for p in input_paths]}")

    for idx, inp in enumerate(input_paths):
        if len(input_paths) == 1 and args.output is not None:
            out = args.output.resolve()
        else:
            if args.output is not None and idx == 0:
                logger.warning(
                    "--output 在批量处理多个文件时无效，已忽略；各文件写入与输入同目录的 *_arena_litjudge.json"
                )
            out = default_litjudge_output_path(inp)

        run(
            input_path=inp,
            output_path=out,
            battles_file=args.battles_file,
            naive=args.naive,
            model_name=args.model,
            platform=args.platform,
            diverse_retrieval=args.diverse_retrieval,
            mmr_lambda=args.mmr_lambda,
            pair_sim_mode=args.pair_sim_mode,
            pool_mult_s=args.pool_mult_s,
            pool_mult_g=args.pool_mult_g,
            embedding_model=args.embedding_model or "",
            expert_outcomes_file=args.expert_outcomes_file,
        )


if __name__ == "__main__":
    main()
