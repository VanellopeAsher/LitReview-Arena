#!/usr/bin/env python3
"""
Main entry point for the Evaluator Agent

Implements Section 6 of the ICML 2026 paper: Expert-aligned Evaluator
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import statistics
from loguru import logger
from tqdm import tqdm

# 仓库根须在 path 中，以便 `import evaluator.*`；evaluator/ 子目录在末尾供 `from utils` 等。
# （原 evaluator.py 已改名为 judge.py，避免与包名 evaluator 冲突。）
_ROOT = Path(__file__).resolve().parent.parent
_EVAL = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EVAL) not in sys.path:
    sys.path.append(str(_EVAL))

try:
    from siliconflow import ExperimentTracker, calculate_cost
except ImportError:
    from evaluator.siliconflow import ExperimentTracker, calculate_cost

# 使用相对导入（当作为模块运行时）或绝对导入（当作为脚本运行时）
try:
    from .config import (
        DEFAULT_MODEL,
        DEFAULT_PLATFORM,
        OUTPUT_DIR,
        BATTLES_FILE,
        EXPERT_OUTCOMES_FILE,
        DEFAULT_AGGREGATED_SUBSET_FILE,
        MMR_LAMBDA,
        PAIR_SIM_MODE_DEFAULT,
        LOCAL_EMBEDDING_MODEL,
        DIVERSITY_POOL_MULT_S,
        DIVERSITY_POOL_MULT_G,
        SUGGESTION_CLUSTER_COSINE_THRESHOLD,
    )
    from .data_loader import load_all_data, load_expert_outcomes_jsonl
    from .context_builder import ContextBuilder
    from .naive_judge import NaiveContextBuilder
    from .judge import Evaluator
    from .aggregator import aggregate_results, load_expert_outcomes_from_battles
except ImportError:
    from evaluator.config import (
        DEFAULT_MODEL,
        DEFAULT_PLATFORM,
        OUTPUT_DIR,
        BATTLES_FILE,
        EXPERT_OUTCOMES_FILE,
        DEFAULT_AGGREGATED_SUBSET_FILE,
        MMR_LAMBDA,
        PAIR_SIM_MODE_DEFAULT,
        LOCAL_EMBEDDING_MODEL,
        DIVERSITY_POOL_MULT_S,
        DIVERSITY_POOL_MULT_G,
        SUGGESTION_CLUSTER_COSINE_THRESHOLD,
    )
    from evaluator.data_loader import load_all_data, load_expert_outcomes_jsonl
    from evaluator.context_builder import ContextBuilder
    from evaluator.naive_judge import NaiveContextBuilder
    from evaluator.judge import Evaluator
    from evaluator.aggregator import aggregate_results, load_expert_outcomes_from_battles


def _compute_run_metadata(
    args,
    judge_outcomes: Dict[str, Any],
    battles: List[Dict],
) -> tuple:
    """LitJudge 检索汇总 + 可选 research-suggestion diversity（与 eval 是否跳过无关）。"""
    litjudge_retrieval = (
        "naive"
        if args.naive
        else ("diversity_mmr" if args.diverse_retrieval else "topk")
    )
    retrieval_diagnostics = (
        None if args.naive
        else _summarize_retrieval_diagnostics(judge_outcomes)
    )
    suggestion_div = None
    if args.analyze_suggestion_diversity and not args.naive:
        try:
            from .suggestion_diversity_analysis import (
                compute_run_level_research_suggestion_diversity,
            )
        except ImportError:
            from evaluator.suggestion_diversity_analysis import (
                compute_run_level_research_suggestion_diversity,
            )
        try:
            suggestion_div = compute_run_level_research_suggestion_diversity(
                judge_outcomes,
                battles,
                embedding_model_name=args.embedding_model,
                distance_threshold=args.suggestion_cluster_threshold,
                text_source=args.suggestion_diversity_text_source,
                extraction_model=(args.extraction_model or args.model),
                extraction_platform=(args.extraction_platform or args.platform),
                extraction_max_workers=args.extraction_workers,
            )
        except Exception as e:
            logger.warning(f"Research-suggestion diversity analysis failed: {e}")
    return litjudge_retrieval, retrieval_diagnostics, suggestion_div


def _summarize_retrieval_diagnostics(judge_outcomes: Dict[str, Any]) -> Dict[str, Any]:
    """Run-level means of per-battle example pairwise similarity (mechanism metrics)."""
    s_vals: List[float] = []
    c_vals: List[float] = []
    g_vals: List[float] = []
    for rec in judge_outcomes.values():
        if not isinstance(rec, dict):
            continue
        rm = rec.get("retrieval_metrics")
        if not isinstance(rm, dict):
            continue
        if rm.get("avg_pairwise_example_similarity_S") is not None:
            s_vals.append(rm["avg_pairwise_example_similarity_S"])
        if rm.get("avg_pairwise_example_similarity_C") is not None:
            c_vals.append(rm["avg_pairwise_example_similarity_C"])
        if rm.get("avg_pairwise_example_similarity_G") is not None:
            g_vals.append(rm["avg_pairwise_example_similarity_G"])
    return {
        "mean_avg_pairwise_example_similarity_S": statistics.mean(s_vals) if s_vals else None,
        "mean_avg_pairwise_example_similarity_C": statistics.mean(c_vals) if c_vals else None,
        "mean_avg_pairwise_example_similarity_G": statistics.mean(g_vals) if g_vals else None,
        "n_battles_with_retrieval_metrics": sum(
            1
            for r in judge_outcomes.values()
            if isinstance(r, dict) and r.get("retrieval_metrics")
        ),
    }


def setup_logging(log_file: Path = None):
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG"
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluator Agent for Literature Review Quality Assessment")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=DEFAULT_PLATFORM,
        help=f"LLM platform (default: {DEFAULT_PLATFORM})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--battle-ids-file",
        type=Path,
        help="Optional JSON file of battle IDs to evaluate only (overrides aggregated subset and --all-battles). "
             "Format: {\"selected_battles\": [...]}, {\"sampled_battle_ids\": [...]}, or a JSON array."
    )
    parser.add_argument(
        "--aggregated-subset-file",
        type=Path,
        default=None,
        help=f"Evaluator aggregated/results JSON; only keys of judge_outcomes are run. "
             f"Default when omitted: {DEFAULT_AGGREGATED_SUBSET_FILE}",
    )
    parser.add_argument(
        "--all-battles",
        action="store_true",
        help="Evaluate every battle in --battles-file instead of the aggregated JSON subset.",
    )
    parser.add_argument(
        "--battles-file",
        type=Path,
        default=BATTLES_FILE,
        help=f"Battles file path (default: {BATTLES_FILE})"
    )
    parser.add_argument(
        "--expert-outcomes-file",
        type=Path,
        default=None,
        help=f"Expert labels JSONL (default: {EXPERT_OUTCOMES_FILE} if present)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation and only aggregate existing results"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Path to existing results file (for aggregation only or resume)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file (only evaluate missing battles)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent workers for parallel processing (default: 4, set to 1 for serial)"
    )
    parser.add_argument(
        "--naive",
        action="store_true",
        help="Naive judge: rubric + current battle only (no Group S/C/G, no similarity retrieval).",
    )
    parser.add_argument(
        "--diverse-retrieval",
        action="store_true",
        help="LitJudge: MMR diversity-constrained selection over same candidate pools (fair vs top-k).",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=MMR_LAMBDA,
        help=f"MMR relevance weight (default: {MMR_LAMBDA})",
    )
    parser.add_argument(
        "--pair-sim-mode",
        type=str,
        choices=["embedding", "jaccard"],
        default=PAIR_SIM_MODE_DEFAULT,
        help="Group C/G text pairwise: embedding (local) or jaccard",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=LOCAL_EMBEDDING_MODEL,
        help="sentence-transformers model id for Group C/G PairSim + suggestion diversity",
    )
    parser.add_argument(
        "--pool-mult-s",
        type=int,
        default=DIVERSITY_POOL_MULT_S,
        help="Group S MMR pool = GROUP_S_SIZE * this (no extra LLM calls)",
    )
    parser.add_argument(
        "--pool-mult-g",
        type=int,
        default=DIVERSITY_POOL_MULT_G,
        help="Group G MMR pool = GROUP_G_SIZE * this (embedding/jaccard vs topic; no LLM)",
    )
    parser.add_argument(
        "--analyze-suggestion-diversity",
        action="store_true",
        help="After eval, compute query-level research-suggestion diversity (cluster entropy) and attach to outputs.",
    )
    parser.add_argument(
        "--suggestion-cluster-threshold",
        type=float,
        default=SUGGESTION_CLUSTER_COSINE_THRESHOLD,
        help="Agglomerative clustering cosine distance threshold in embedding space",
    )
    parser.add_argument(
        "--suggestion-diversity-text-source",
        type=str,
        choices=["judge_output", "winning_draft_llm"],
        default="winning_draft_llm",
        help="judge_output: diversity on judge text; winning_draft_llm: LLM-extract suggestions from LitJudge-preferred draft (D4 then D5).",
    )
    parser.add_argument(
        "--extraction-model",
        type=str,
        default=None,
        help="Model for --suggestion-diversity-text-source winning_draft_llm (default: same as --model)",
    )
    parser.add_argument(
        "--extraction-platform",
        type=str,
        default=None,
        help="Platform for extraction LLM (default: same as --platform)",
    )
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=8,
        help="Parallel threads for winning-draft LLM extraction (1 = serial + delay)",
    )
    parser.add_argument(
        "--context-cache-file",
        type=Path,
        default=None,
        help="JSONL file: reuse saved LitJudge user prompt + retrieval_metrics per battle; "
             "skips build_context on hit (saves retrieval-side LLM tokens, e.g. Group C scoring).",
    )
    parser.add_argument(
        "--context-cache-readonly",
        action="store_true",
        help="Only read context cache; do not append new entries (misses still run full build_context).",
    )

    args = parser.parse_args()
    
    # 设置日志
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"evaluator_{timestamp}.log"
    setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("Evaluator Agent - Expert-aligned Calibrated Evaluator")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Platform: {args.platform}")
    logger.info(f"Max workers: {args.max_workers} ({'parallel' if args.max_workers > 1 else 'serial'})")
    _lit = "naive (no S/C/G)" if args.naive else (
        "LitJudge + diversity MMR" if args.diverse_retrieval else "LitJudge (top-k)"
    )
    logger.info(f"Judge mode: {_lit}")
    logger.info(f"Output directory: {output_dir}")
    
    # 加载数据
    logger.info("\n" + "=" * 80)
    logger.info("Loading data...")
    logger.info("=" * 80)
    
    battles, sampled_ids = load_all_data(
        battles_file=args.battles_file,
        battle_ids_file=args.battle_ids_file,
        aggregated_subset_file=args.aggregated_subset_file,
        use_all_battles=args.all_battles,
    )
    
    logger.info(f"Loaded {len(battles)} battles")
    logger.info(f"Sampled battle IDs: {len(sampled_ids)}")
    
    eo_path = args.expert_outcomes_file or EXPERT_OUTCOMES_FILE
    expert_from_file = load_expert_outcomes_jsonl(Path(eo_path))
    expert_from_battles = load_expert_outcomes_from_battles(battles)
    expert_outcomes = {**expert_from_battles, **expert_from_file}
    logger.info(f"Expert outcomes for aggregation / demos: {len(expert_outcomes)} battles")
    
    # 评估或加载已有结果
    existing_results = None
    if args.results_file and args.results_file.exists():
        logger.info(f"\nLoading existing results from {args.results_file}")
        with open(args.results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        existing_results = results_data.get("judge_outcomes", {})
        logger.info(f"Loaded {len(existing_results)} existing results")
    
    if args.skip_evaluation and existing_results:
        judge_outcomes = existing_results
        stats = {}
        litjudge_retrieval, retrieval_diagnostics, suggestion_div = _compute_run_metadata(
            args, judge_outcomes, battles
        )
    else:
        # 初始化 ContextBuilder（LitJudge）或 NaiveContextBuilder
        logger.info("\n" + "=" * 80)
        logger.info("Initializing context builder...")
        logger.info("=" * 80)
        
        if args.context_cache_file and args.naive:
            logger.warning("--context-cache-file is ignored in naive judge mode")

        if args.naive:
            context_builder = NaiveContextBuilder(
                all_battles=battles,
                model_name=args.model,
                platform=args.platform,
            )
        else:
            context_builder = ContextBuilder(
                all_battles=battles,
                model_name=args.model,
                platform=args.platform,
                diversity_retrieval=args.diverse_retrieval,
                mmr_lambda=args.mmr_lambda,
                pair_sim_mode=args.pair_sim_mode,
                pool_mult_s=args.pool_mult_s,
                pool_mult_g=args.pool_mult_g,
                embedding_model_name=args.embedding_model,
            )
        
        # 初始化Evaluator
        logger.info("\n" + "=" * 80)
        logger.info("Initializing Evaluator...")
        logger.info("=" * 80)
        
        evaluator = Evaluator(
            context_builder=context_builder,
            model_name=args.model,
            platform=args.platform,
            expert_outcomes_map=expert_outcomes,
            context_cache_path=None if args.naive else args.context_cache_file,
            context_cache_write=not args.context_cache_readonly,
        )
        if args.context_cache_file and not args.naive:
            logger.info(
                f"Context cache: {args.context_cache_file} "
                f"(append={'off' if args.context_cache_readonly else 'on'})"
            )
        
        # 初始化实验跟踪器
        experiment_id = f"evaluator_{timestamp}"
        tracker = ExperimentTracker(experiment_id, str(output_dir / f"{experiment_id}_stats.json"))
        
        # 评估battles
        logger.info("\n" + "=" * 80)
        if existing_results and args.resume:
            logger.info(f"Resuming evaluation (found {len(existing_results)} existing results)...")
        else:
            logger.info("Evaluating battles...")
        logger.info("=" * 80)
        
        def progress_callback(current, total):
            if hasattr(evaluator.llm, 'last_usage') and evaluator.llm.last_usage:
                usage = evaluator.llm.last_usage
                tracker.add_call(usage, args.model)
        
        # 如果resume模式，使用已有结果
        resume_results = existing_results if (args.resume and existing_results) else None
        
        # 设置checkpoint文件路径
        checkpoint_file = None
        if not args.skip_evaluation:
            if args.resume and args.results_file and args.results_file.exists():
                checkpoint_file = str(args.results_file)
            else:
                checkpoint_file = str(output_dir / f"evaluator_checkpoint_{timestamp}.json")
        
        judge_outcomes, stats = evaluator.evaluate_battles(
            battles,
            progress_callback=progress_callback,
            existing_results=resume_results,
            show_progress=not args.no_progress,
            max_workers=args.max_workers,
            checkpoint_file=checkpoint_file
        )
        
        # 保存结果
        if args.resume and args.results_file and args.results_file.exists():
            # 如果是从已有文件恢复，更新原文件
            results_file = args.results_file
            logger.info(f"\nUpdating existing results file: {results_file}")
        else:
            # 否则创建新文件
            results_file = output_dir / f"evaluator_results_{timestamp}.json"
            logger.info(f"\nSaving results to: {results_file}")
        
        litjudge_retrieval, retrieval_diagnostics, suggestion_div = _compute_run_metadata(
            args, judge_outcomes, battles
        )

        results_data = {
            "model": args.model,
            "platform": args.platform,
            "timestamp": timestamp,
            "judge_mode": "naive" if args.naive else "litjudge",
            "context_cache_file": str(args.context_cache_file) if args.context_cache_file else None,
            "context_cache_readonly": bool(args.context_cache_readonly) if not args.naive else None,
            "litjudge_retrieval": litjudge_retrieval,
            "mmr_lambda": None if args.naive else args.mmr_lambda,
            "pair_sim_mode": None if args.naive else args.pair_sim_mode,
            "retrieval_diagnostics": retrieval_diagnostics,
            "research_suggestion_diversity": suggestion_div,
            "total_battles": len(battles),
            "judge_outcomes_count": len(judge_outcomes),
            "judge_outcomes": judge_outcomes,
            "stats": stats,
        }
        
        # 如果resume，保留原有的一些元数据
        if args.resume and args.results_file and args.results_file.exists():
            with open(args.results_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            # 保留原有的时间戳等信息
            results_data["original_timestamp"] = old_data.get("timestamp", timestamp)
            results_data["resumed"] = True
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_file}")
        
        # 显示统计信息
        final_stats = tracker.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("Experiment Statistics")
        logger.info("=" * 80)
        logger.info(f"Total calls: {final_stats['total_stats']['total_calls']}")
        logger.info(f"Total input tokens: {final_stats['total_stats']['total_input_token']:,}")
        logger.info(f"Total output tokens: {final_stats['total_stats']['total_output_token']:,}")
        logger.info(f"Total tokens: {final_stats['total_stats']['total_token']:,}")
        logger.info(f"Total cost: ${final_stats['total_stats']['total_cost']:.6f}")

    # 聚合结果
    logger.info("\n" + "=" * 80)
    logger.info("Aggregating results...")
    logger.info("=" * 80)
    
    aggregated = aggregate_results(
        battles=battles,
        judge_outcomes=judge_outcomes,
        expert_outcomes=expert_outcomes
    )
    
    # 保存聚合结果
    aggregated_file = output_dir / f"evaluator_aggregated_{timestamp}.json"
    aggregated_data = {
        "model": args.model,
        "platform": args.platform,
        "timestamp": timestamp,
        "judge_mode": "naive" if getattr(args, "naive", False) else "litjudge",
        "litjudge_retrieval": litjudge_retrieval,
        "mmr_lambda": None if args.naive else args.mmr_lambda,
        "pair_sim_mode": None if args.naive else args.pair_sim_mode,
        "retrieval_diagnostics": retrieval_diagnostics,
        "research_suggestion_diversity": suggestion_div,
        **aggregated,
        "judge_outcomes": judge_outcomes,
    }
    
    with open(aggregated_file, "w", encoding="utf-8") as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nAggregated results saved to: {aggregated_file}")
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total battles evaluated: {aggregated['judge_outcomes_count']}")
    logger.info("\nLeaderboard Correlations:")
    for dim, corr in aggregated.get("leaderboard_correlations", {}).items():
        logger.info(f"  {dim}: {corr:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
