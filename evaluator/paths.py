"""
Repository root and canonical data paths (cwd-independent).
"""

from pathlib import Path

# This file lives at repo_root/src/paths.py
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
# 默认子集：与某次 evaluator 聚合输出同名，用于只跑该次结果里出现过的 battle_id
DEFAULT_EVALUATOR_AGGREGATED_SUBSET = REPO_ROOT / "evaluator_aggregated_20260128_183256.json"
EVALUATOR_OUTPUT_DIR = DATA_DIR / "evaluator" / "outputs"
TOKEN_COST_DIR = DATA_DIR / "token_cost"
FIGURES_DIR = DATA_DIR / "figures"
BENCHMARK_DATA_DIR = DATA_DIR  # 仓库根目录下的 data/（与 battles.jsonl 同目录）
# Common filenames under DATA_DIR (token cost CSV may vary; default used by analyze_models / count_models)
OPENROUTER_ACTIVITY_CSV = TOKEN_COST_DIR / "openrouter_activity_2026-01-28 (1).csv"
