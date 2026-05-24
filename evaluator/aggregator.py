"""
Aggregator module for the Evaluator Agent

Implements Bradley-Terry/Elo aggregation, leaderboard calculation,
and correlation computation with expert preferences.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from loguru import logger

try:
    from .bt_model import BradleyTerryModel
except ImportError:
    from evaluator.bt_model import BradleyTerryModel

try:
    from .config import DIMENSIONS, NEUTRAL_OUTCOMES, OUTCOMES
except ImportError:
    from evaluator.config import DIMENSIONS, NEUTRAL_OUTCOMES, OUTCOMES

_VALID_OUTCOMES = frozenset(OUTCOMES)

# Arena 导出键 -> LitJudge 维（与 arena_litjudge_fill / battles_leaderboard_from_json 一致）
ARENA_VOTE_KEY_TO_LITJUDGE_DIM = {
    "d0_overall_utility": "D5",
    "d1_literature_coverage": "D1",
    "d2_claim_support": "D2",
    "d3_paper_structure": "D3",
    "d4_research_suggestions": "D4",
}


def compute_leaderboard_correlation(
    expert_outcomes: Dict[str, Dict[str, str]],
    judge_outcomes: Dict[str, Dict[str, str]],
    battles: List[Dict],
    dimension: str
) -> float:
    """
    计算leaderboard correlation (Spearman's ρ)
    
    Args:
        expert_outcomes: {battle_id: {dimension: outcome}}
        judge_outcomes: {battle_id: {dimension: outcome}}
        battles: battles列表
        dimension: 维度名称
    
    Returns:
        Spearman correlation coefficient
    """
    # 使用与section5_meta_evaluation.py相同的方法
    # 参考bootstrap_bt_elo.py中的compute_bt_elo函数
    
    try:
        # 准备comparison数据
        expert_comparisons = []
        judge_comparisons = []
        
        # 创建agent到ID的映射
        agent_to_id = {}
        agent_id_counter = 0
        
        for battle in battles:
            battle_id = battle.get("battle_id")
            if battle_id not in expert_outcomes or battle_id not in judge_outcomes:
                continue
            
            expert_outcome = expert_outcomes[battle_id].get(dimension)
            judge_outcome = judge_outcomes[battle_id].get(dimension)
            
            if not expert_outcome or not judge_outcome:
                continue
            
            agent_a_key = battle.get("agent_a_key") or (battle.get("draft_a") or {}).get("system_id")
            agent_b_key = battle.get("agent_b_key") or (battle.get("draft_b") or {}).get("system_id")

            if not agent_a_key or not agent_b_key:
                continue
            
            # 分配ID
            if agent_a_key not in agent_to_id:
                agent_to_id[agent_a_key] = agent_id_counter
                agent_id_counter += 1
            if agent_b_key not in agent_to_id:
                agent_to_id[agent_b_key] = agent_id_counter
                agent_id_counter += 1
            
            agent_a_id = agent_to_id[agent_a_key]
            agent_b_id = agent_to_id[agent_b_key]
            
            # 处理expert outcome
            if expert_outcome == "A":
                expert_comparisons.append((agent_a_id, agent_b_id, "A", 1.0))
            elif expert_outcome == "B":
                expert_comparisons.append((agent_b_id, agent_a_id, "A", 1.0))
            elif expert_outcome in NEUTRAL_OUTCOMES:
                expert_comparisons.append((agent_a_id, agent_b_id, "Tie", 1.0))
            
            # 处理judge outcome
            if judge_outcome == "A":
                judge_comparisons.append((agent_a_id, agent_b_id, "A", 1.0))
            elif judge_outcome == "B":
                judge_comparisons.append((agent_b_id, agent_a_id, "A", 1.0))
            elif judge_outcome in NEUTRAL_OUTCOMES:
                judge_comparisons.append((agent_a_id, agent_b_id, "Tie", 1.0))
        
        if len(expert_comparisons) == 0 or len(judge_comparisons) == 0:
            return 0.0
        
        # 使用Bradley-Terry模型计算ratings
        expert_model = BradleyTerryModel()
        judge_model = BradleyTerryModel()
        
        expert_model.fit(expert_comparisons)
        judge_model.fit(judge_comparisons)
        
        # 获取ratings
        expert_ratings_dict = {}
        judge_ratings_dict = {}
        
        if expert_model.beta is not None and expert_model.model_ids is not None:
            for idx, model_id in enumerate(expert_model.model_ids):
                # 找到对应的agent_key
                agent_key = None
                for key, aid in agent_to_id.items():
                    if aid == model_id:
                        agent_key = key
                        break
                if agent_key:
                    expert_ratings_dict[agent_key] = expert_model.beta[idx]
        
        if judge_model.beta is not None and judge_model.model_ids is not None:
            for idx, model_id in enumerate(judge_model.model_ids):
                agent_key = None
                for key, aid in agent_to_id.items():
                    if aid == model_id:
                        agent_key = key
                        break
                if agent_key:
                    judge_ratings_dict[agent_key] = judge_model.beta[idx]
        
        # 提取共同的agents
        common_agents = set(expert_ratings_dict.keys()) & set(judge_ratings_dict.keys())
        
        if len(common_agents) < 2:
            return 0.0
        
        # 提取ratings
        expert_scores = [expert_ratings_dict[agent] for agent in common_agents]
        judge_scores = [judge_ratings_dict[agent] for agent in common_agents]
        
        # 计算Spearman correlation
        correlation, _ = spearmanr(expert_scores, judge_scores)
        
        return correlation if not np.isnan(correlation) else 0.0
    
    except Exception as e:
        logger.error(f"计算leaderboard correlation时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def aggregate_results(
    battles: List[Dict],
    judge_outcomes: Dict[str, Dict[str, str]],
    expert_outcomes: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict:
    """
    聚合评估结果
    
    Args:
        battles: battles列表
        judge_outcomes: judge outcomes {battle_id: {dimension: outcome}}
        expert_outcomes: 可选的expert outcomes（用于计算correlation）
    
    Returns:
        聚合结果字典
    """
    results = {
        "total_battles": len(battles),
        "judge_outcomes_count": len(judge_outcomes),
        "leaderboard_correlations": {}
    }
    
    # 如果没有expert outcomes，只返回基本统计
    if not expert_outcomes:
        return results
    
    # 计算每个维度的 leaderboard correlation
    for dimension in DIMENSIONS.keys():
        correlation = compute_leaderboard_correlation(
            expert_outcomes, judge_outcomes, battles, dimension
        )
        results["leaderboard_correlations"][dimension] = correlation
        logger.info(f"{dimension}: Correlation = {correlation:.4f}")
    
    return results


def load_expert_outcomes_from_battles(battles: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    从 battles 中提取 expert outcomes，用于 ρ 与 in-context 示范标签。

    优先级：``dimension_results``（主 bench）；否则 ``metadata.arena_votes``
    （d0→D5，d1–d4→D1–D4，取值须为 A/B/Tie/BothBad）。
    """
    expert_outcomes: Dict[str, Dict[str, str]] = {}

    for battle in battles:
        battle_id = battle.get("battle_id")
        if not battle_id:
            continue

        dimension_results = battle.get("dimension_results") or {}
        if isinstance(dimension_results, dict) and len(dimension_results) > 0:
            expert_outcomes[battle_id] = {
                k: v for k, v in dimension_results.items() if isinstance(v, str)
            }
            continue

        meta = battle.get("metadata") or {}
        av = meta.get("arena_votes") if isinstance(meta, dict) else None
        if not isinstance(av, dict):
            continue
        mapped: Dict[str, str] = {}
        for av_key, dim in ARENA_VOTE_KEY_TO_LITJUDGE_DIM.items():
            v = av.get(av_key)
            if isinstance(v, str) and v in _VALID_OUTCOMES:
                mapped[dim] = v
        if mapped:
            expert_outcomes[battle_id] = mapped

    return expert_outcomes
