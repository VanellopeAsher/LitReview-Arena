"""
Configuration for the Evaluator Agent
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

_eval_dir = Path(__file__).resolve().parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))
from paths import DATA_DIR, EVALUATOR_OUTPUT_DIR, DEFAULT_EVALUATOR_AGGREGATED_SUBSET, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

# 维度定义（与section5_meta_evaluation.py保持一致；D1–D4 为分项，D5 为 Overall）
DIMENSIONS = {
    "D1": "Citation Coverage: Which draft cites a more complete and appropriate set of relevant papers, with fewer obvious omissions?",
    "D2": "Citation–Claim Support: Which draft more reliably supports its key claims with relevant citations?",
    "D3": "Review Landscape Structure: Which draft better organizes prior work into clear categories or comparisons that help a researcher understand relationships among approaches?",
    "D4": "Gaps and Directions: Which draft more clearly identifies important and non-obvious gaps or future directions that would meaningfully inform next steps?",
    "D5": "Overall Review Utility: From a researcher's perspective, which draft would you prefer to use as a starting point?",
}

# 4-way outcomes
OUTCOMES = ["A", "B", "Tie", "BothBad"]
NEUTRAL_OUTCOMES = ["Tie", "BothBad"]

# LLM配置（仓库根 .env：EVALUATOR_MODEL、可选 EVALUATOR_PLATFORM；CLI --model 仍可覆盖）
_FALLBACK_MODEL = "qwen/qwen3-235b-a22b-2507"
_FALLBACK_PLATFORM = "openrouter"
DEFAULT_MODEL = (os.environ.get("EVALUATOR_MODEL") or _FALLBACK_MODEL).strip() or _FALLBACK_MODEL
DEFAULT_PLATFORM = (os.environ.get("EVALUATOR_PLATFORM") or _FALLBACK_PLATFORM).strip() or _FALLBACK_PLATFORM

# 案例检索配置
MAX_DEMONSTRATIONS_PER_GROUP = 3  # 每组最多3个demonstrations
GROUP_S_SIZE = 3  # Structure-similar cases
GROUP_C_SIZE = 3  # Content-similar cases
GROUP_G_SIZE = 3  # Gap anchors

# LitJudge + diversity-constrained retrieval（Group C 的 LLM 候选数固定为 GROUP_C_SIZE * 3，与原版一致）
MMR_LAMBDA = 0.7  # 相关项权重；次项为 (1 - lambda) * redundant_similarity
DIVERSITY_POOL_MULT_S = 5  # Group S：MMR 候选池 = max_cases * 此倍数（仅结构相似度排序截断，无额外 LLM）
DIVERSITY_POOL_MULT_G = 5  # Group G：MMR 候选池 = max_anchors * 此倍数（gap 句与 topic 的 embedding/jaccard，无 LLM）
PAIR_SIM_MODE_DEFAULT = "embedding"  # Group C/G 文本 PairSim：embedding | jaccard（缺依赖时回退 jaccard）
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers 模型 id

# Research-suggestion diversity 分析（独立模块；凝聚聚类 + 余弦距离阈值）
SUGGESTION_CLUSTER_COSINE_THRESHOLD = 0.35  # 1 - cosine_similarity 尺度下与 sklearn cosine distance 对齐；可按实验调参

# 数据文件路径（仓库根 data/）；LitReviewBench 默认使用 battles.jsonl（内嵌 draft 正文）
BATTLES_FILE = DATA_DIR / "battles.jsonl"
EXPERT_OUTCOMES_FILE = DATA_DIR / "expert_outcomes.jsonl"  # 可选；用于 in-context 专家标签与聚合对比

# 未指定 --battle-ids-file 且未使用 --all-battles 时，只评测该聚合 JSON 中 judge_outcomes 列出的 battle
DEFAULT_AGGREGATED_SUBSET_FILE = DEFAULT_EVALUATOR_AGGREGATED_SUBSET

# 实验配置
MAX_RETRIES = 3
API_CALL_DELAY = 0.5  # 每次API调用之间的基础延迟（秒）
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 2.0

# 定期保存配置
CHECKPOINT_INTERVAL = 10  # 每处理多少个battles保存一次checkpoint（0表示禁用）

# 输出配置
OUTPUT_DIR = EVALUATOR_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
