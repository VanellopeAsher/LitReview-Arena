"""
对 JSON 数组形式的 battles 文件（如 battles_300.json）中「尚无完整 LitJudge 五维结果」的条目跑 LitJudge，
仅写入 metadata.judge_outcomes（D1–D5），不增删其它字段。

判定「已有结果」：metadata.judge_outcomes 含 D1–D5 且取值属于 A/B/Tie/BothBad。

用法（仓库根目录）：
  python evaluator/battles_json_litjudge_fill.py --input battles_300.json --output battles_300_filled.json
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

DIMS = ("D1", "D2", "D3", "D4", "D5")
VALID = frozenset({"A", "B", "Tie", "BothBad"})


def get_judge_outcomes(battle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    m = battle.get("metadata")
    if not isinstance(m, dict):
        return None
    jo = m.get("judge_outcomes")
    return jo if isinstance(jo, dict) else None


def has_complete_litjudge(battle: Dict[str, Any]) -> bool:
    jo = get_judge_outcomes(battle)
    if not jo:
        return False
    return all(jo.get(d) in VALID for d in DIMS)


def set_judge_outcomes(battle: Dict[str, Any], rec: Dict[str, Any]) -> None:
    if not all(rec.get(d) in VALID for d in DIMS):
        return
    battle.setdefault("metadata", {})["judge_outcomes"] = {d: str(rec[d]) for d in DIMS}


def merge_corpus_for_retrieval(
    user_battles: List[Dict],
    corpus_file: Path,
) -> List[Dict]:
    """全库 + 用户条目中同 battle_id 以用户对象为准（保证正文与 battles_300 一致）。"""
    try:
        from .data_loader import load_battles
    except ImportError:
        from evaluator.data_loader import load_battles

    base = load_battles(corpus_file)
    by_id: Dict[str, Dict] = {str(b.get("battle_id")): b for b in base if b.get("battle_id")}
    for b in user_battles:
        bid = b.get("battle_id")
        if bid:
            by_id[str(bid)] = b
    return list(by_id.values())


def run(
    input_path: Path,
    output_path: Path,
    corpus_file: Path,
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
    max_workers: int,
    no_progress: bool,
    checkpoint_file: Optional[Path],
) -> None:
    try:
        from .config import (
            BATTLES_FILE,
            MMR_LAMBDA,
            DIVERSITY_POOL_MULT_S,
            DIVERSITY_POOL_MULT_G,
            PAIR_SIM_MODE_DEFAULT,
            LOCAL_EMBEDDING_MODEL,
            CHECKPOINT_INTERVAL,
        )
        from .context_builder import ContextBuilder
        from .naive_judge import NaiveContextBuilder
        from .judge import Evaluator
        from .data_loader import load_expert_outcomes_jsonl
    except ImportError:
        from evaluator.config import (
            BATTLES_FILE,
            MMR_LAMBDA,
            DIVERSITY_POOL_MULT_S,
            DIVERSITY_POOL_MULT_G,
            PAIR_SIM_MODE_DEFAULT,
            LOCAL_EMBEDDING_MODEL,
            CHECKPOINT_INTERVAL,
        )
        from evaluator.context_builder import ContextBuilder
        from evaluator.naive_judge import NaiveContextBuilder
        from evaluator.judge import Evaluator
        from evaluator.data_loader import load_expert_outcomes_jsonl

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入须为 JSON 数组（battles 列表）")

    missing = [b for b in data if not has_complete_litjudge(b)]
    done_ct = len(data) - len(missing)
    logger.info(f"共 {len(data)} 条：已有完整 judge_outcomes {done_ct}，待评测 {len(missing)}")
    if not missing:
        logger.info("无需评测，直接复制输入到输出")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    all_battles = merge_corpus_for_retrieval(data, corpus_file)
    logger.info(f"检索语料合并后共 {len(all_battles)} 条 battle")

    eo = (
        load_expert_outcomes_jsonl(Path(expert_outcomes_file))
        if expert_outcomes_file
        else load_expert_outcomes_jsonl()
    )

    if naive:
        logger.info("初始化 NaiveContextBuilder（无检索）…")
        cb = NaiveContextBuilder(all_battles=all_battles, model_name=model_name, platform=platform)
    else:
        logger.info(
            "初始化 LitJudge ContextBuilder：加载 embedding、预计算结构图与 gap anchors（可能耗时数分钟，"
            "期间 PyTorch 可能打印 Redirects 警告，可忽略）…"
        )
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
        logger.info("ContextBuilder 就绪。")

    logger.info("初始化 Evaluator（LLM judge）…")
    evaluator = Evaluator(
        context_builder=cb,
        model_name=model_name,
        platform=platform,
        expert_outcomes_map=eo,
    )
    logger.info(f"开始评测 {len(missing)} 条 battle（max_workers={max_workers}）…")

    cp = str(checkpoint_file) if checkpoint_file else None
    judge_outcomes: Dict[str, Dict[str, Any]] = {}
    if cp and Path(cp).exists():
        try:
            with open(cp, "r", encoding="utf-8") as f:
                ck = json.load(f)
            judge_outcomes = ck.get("judge_outcomes") or {}
            logger.info(f"从 checkpoint 恢复 {len(judge_outcomes)} 条结果: {cp}")
        except Exception as e:
            logger.warning(f"读取 checkpoint 失败，将从头评测: {e}")

    results, _stats = evaluator.evaluate_battles(
        missing,
        existing_results=judge_outcomes,
        show_progress=not no_progress,
        max_workers=max_workers,
        checkpoint_file=cp if cp and CHECKPOINT_INTERVAL > 0 else None,
    )

    by_id: Dict[str, Dict] = {str(b.get("battle_id")): b for b in data if b.get("battle_id")}
    for bid, rec in results.items():
        if not isinstance(rec, dict):
            continue
        b = by_id.get(str(bid))
        if b:
            set_judge_outcomes(b, rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"已写入: {output_path}")


def main() -> None:
    try:
        from .config import (
            DEFAULT_MODEL,
            DEFAULT_PLATFORM,
            BATTLES_FILE,
            MMR_LAMBDA,
        )
    except ImportError:
        from evaluator.config import DEFAULT_MODEL, DEFAULT_PLATFORM, BATTLES_FILE, MMR_LAMBDA

    p = argparse.ArgumentParser(description="对 battles JSON 数组中缺 judge_outcomes 的条目跑 LitJudge")
    p.add_argument("--input", type=Path, required=True, help="battles_300.json（JSON 数组）")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认: <input 名>_litjudge.json",
    )
    p.add_argument(
        "--corpus-battles-file",
        type=Path,
        default=BATTLES_FILE,
        help="检索用全库（默认 data/battles.jsonl），与用户文件按 battle_id 合并",
    )
    p.add_argument("--naive", action="store_true")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--platform", type=str, default=DEFAULT_PLATFORM)
    p.add_argument("--diverse-retrieval", action="store_true")
    p.add_argument("--mmr-lambda", type=float, default=MMR_LAMBDA)
    p.add_argument("--pair-sim-mode", type=str, default="embedding")
    p.add_argument("--pool-mult-s", type=int, default=5)
    p.add_argument("--pool-mult-g", type=int, default=5)
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--expert-outcomes-file", type=Path, default=None)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="断点 JSON：含 judge_outcomes 键，与 main 的 checkpoint 格式兼容",
    )
    args = p.parse_args()

    out = args.output or args.input.with_name(args.input.stem + "_litjudge.json")

    run(
        input_path=args.input.resolve(),
        output_path=out.resolve(),
        corpus_file=args.corpus_battles_file.resolve(),
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
        max_workers=args.max_workers,
        no_progress=args.no_progress,
        checkpoint_file=args.checkpoint_file,
    )


if __name__ == "__main__":
    main()
