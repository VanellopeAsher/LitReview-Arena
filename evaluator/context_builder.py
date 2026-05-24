"""
Context builder module for the Evaluator Agent

Implements structure extraction, case retrieval, and context assembly
as described in Section 6.1 of the paper.
"""

import sys
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

import numpy as np
from loguru import logger

# 添加父目录到路径以导入utils中的LLM
sys.path.insert(0, str(Path(__file__).parent.parent))

# 类型检查时导入LLM，运行时延迟导入
if TYPE_CHECKING:
    from utils import LLM
else:
    LLM = None  # 占位符，实际使用时动态导入

try:
    from .config import (
        MAX_DEMONSTRATIONS_PER_GROUP,
        GROUP_S_SIZE,
        GROUP_C_SIZE,
        GROUP_G_SIZE,
        DIMENSIONS,
        MMR_LAMBDA,
        DIVERSITY_POOL_MULT_S,
        DIVERSITY_POOL_MULT_G,
        PAIR_SIM_MODE_DEFAULT,
        LOCAL_EMBEDDING_MODEL,
    )
    from .structure_utils import (
        extract_skeleton_text,
        build_paragraph_network,
        compute_graph_similarity,
        extract_gap_anchors
    )
    from .data_loader import get_battle_responses, battle_topic_text
except ImportError:
    from evaluator.config import (
        MAX_DEMONSTRATIONS_PER_GROUP,
        GROUP_S_SIZE,
        GROUP_C_SIZE,
        GROUP_G_SIZE,
        DIMENSIONS,
        MMR_LAMBDA,
        DIVERSITY_POOL_MULT_S,
        DIVERSITY_POOL_MULT_G,
        PAIR_SIM_MODE_DEFAULT,
        LOCAL_EMBEDDING_MODEL,
    )
    from evaluator.structure_utils import (
        extract_skeleton_text,
        build_paragraph_network,
        compute_graph_similarity,
        extract_gap_anchors
    )
    from evaluator.data_loader import get_battle_responses, battle_topic_text


def _mmr_select(
    ranked: List[Tuple[float, Dict]],
    k: int,
    pair_sim_fn: Callable[[Dict, Dict], float],
    lambda_: float,
) -> List[Dict]:
    """Maximal Marginal Relevance：主项为 Rel，次项为与已选集合的最大 PairSim。"""
    if not ranked or k <= 0:
        return []
    if len(ranked) <= k:
        return [b for _, b in ranked]

    pool = list(ranked)
    selected: List[Dict] = []
    remaining = list(range(len(pool)))

    first_idx = max(remaining, key=lambda i: pool[i][0])
    selected.append(pool[first_idx][1])
    remaining.remove(first_idx)

    while len(selected) < k and remaining:
        best_i = None
        best_score = -1e18
        for i in remaining:
            rel = pool[i][0]
            bi = pool[i][1]
            max_sim = max(pair_sim_fn(bi, sj) for sj in selected)
            mmr_score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected.append(pool[best_i][1])
        remaining.remove(best_i)
    return selected


def _avg_pairwise(
    battles: List[Dict],
    pair_sim_fn: Callable[[Dict, Dict], float],
) -> Optional[float]:
    n = len(battles)
    if n < 2:
        return None
    s = 0.0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += pair_sim_fn(battles[i], battles[j])
            c += 1
    return s / c if c else None


class ContextBuilder:
    """
    上下文构建器，负责：
    1. 提取结构信息（skeleton text, graph network）
    2. 检索相似案例（Group S/C/G）
    3. 组装in-context packet
    """

    def __init__(
        self,
        all_battles: List[Dict],
        llm: Optional["LLM"] = None,
        model_name: str = "deepseek-ai/DeepSeek-V3.2",
        platform: str = "siliconflow",
        diversity_retrieval: bool = False,
        mmr_lambda: float = MMR_LAMBDA,
        pair_sim_mode: str = PAIR_SIM_MODE_DEFAULT,
        pool_mult_s: int = DIVERSITY_POOL_MULT_S,
        pool_mult_g: int = DIVERSITY_POOL_MULT_G,
        embedding_model_name: str = LOCAL_EMBEDDING_MODEL,
    ):
        self.all_battles = all_battles
        self.diversity_retrieval = diversity_retrieval
        self.mmr_lambda = mmr_lambda
        self.pair_sim_mode_requested = (pair_sim_mode or PAIR_SIM_MODE_DEFAULT).lower()
        self.pool_mult_s = pool_mult_s
        self.pool_mult_g = pool_mult_g
        self.embedding_model_name = embedding_model_name

        self._pair_sim_mode_effective: str = "jaccard"
        self._sentence_model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embed_lock = threading.Lock()

        if llm is None:
            from utils import LLM
            self.llm = LLM(model_name=model_name, platform=platform)
        else:
            self.llm = llm

        self._resolve_pair_sim_mode()

        self._structure_cache = {}
        self._gap_anchors_cache = {}
        self._precompute_structures()
        self._precompute_gap_anchors()

    def _resolve_pair_sim_mode(self) -> None:
        if self.pair_sim_mode_requested == "jaccard":
            self._pair_sim_mode_effective = "jaccard"
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer(self.embedding_model_name)
            self._pair_sim_mode_effective = "embedding"
        except Exception as e:
            logger.warning(
                f"sentence-transformers 不可用 ({e})，Group C/G 文本 PairSim 回退为 jaccard。"
            )
            self._pair_sim_mode_effective = "jaccard"
            self._sentence_model = None

    def _encode_topic(self, text: str) -> np.ndarray:
        if not text or not self._sentence_model:
            return np.zeros(384, dtype=np.float32)
        key = text[:2000]
        with self._embed_lock:
            if key in self._embedding_cache:
                return self._embedding_cache[key]
        v = self._sentence_model.encode(key, convert_to_numpy=True, normalize_embeddings=True)
        with self._embed_lock:
            self._embedding_cache[key] = v
        return v

    def _pair_sim_content(self, b1: Dict, b2: Dict) -> float:
        t1 = battle_topic_text(b1)
        t2 = battle_topic_text(b2)
        if self._pair_sim_mode_effective == "jaccard":
            return self._compute_text_similarity_fallback(t1, t2)
        e1 = self._encode_topic(t1)
        e2 = self._encode_topic(t2)
        return float(np.dot(e1, e2))

    def _pair_sim_structure(self, b1: Dict, b2: Dict) -> float:
        id1 = b1.get("battle_id")
        id2 = b2.get("battle_id")
        if not id1 or not id2 or id1 not in self._structure_cache or id2 not in self._structure_cache:
            return 0.0
        g1, _ = self._structure_cache[id1]
        g2, _ = self._structure_cache[id2]
        return compute_graph_similarity(g1, g2)

    def _sim_gap_strings(self, a: str, b: str) -> float:
        """Gap 句之间或与 query 的相关度 / 两两冗余（与 Group C 一致：embedding 或 jaccard）。"""
        if self._pair_sim_mode_effective == "jaccard":
            return self._compute_text_similarity_fallback(a, b)
        e1 = self._encode_topic(a)
        e2 = self._encode_topic(b)
        return float(np.dot(e1, e2))

    def _pair_sim_gap(self, d1: Dict, d2: Dict) -> float:
        return self._sim_gap_strings(d1["text"], d2["text"])

    def _precompute_structures(self):
        """预计算所有battles的结构信息"""
        logger.info("预计算battles的结构信息...")

        for battle in self.all_battles:
            battle_id = battle.get("battle_id")
            if not battle_id:
                continue

            response_a, response_b = get_battle_responses(battle)
            if not response_a or not response_b:
                continue

            try:
                section_headers, paragraph_leads = extract_skeleton_text(response_a)
                graph = build_paragraph_network(section_headers, paragraph_leads)
                self._structure_cache[battle_id] = (graph, (section_headers, paragraph_leads))
            except Exception as e:
                logger.warning(f"提取battle {battle_id}的结构时出错: {e}")
                continue

        logger.info(f"预计算了 {len(self._structure_cache)} 个battles的结构信息")

    def _precompute_gap_anchors(self):
        """预提取gap anchors（仅从human-written reviews）"""
        logger.info("预提取gap anchors...")

        for battle in self.all_battles:
            battle_id = battle.get("battle_id")
            if not battle_id:
                continue

            da = battle.get("draft_a") or {}
            db = battle.get("draft_b") or {}
            sid_a = (da.get("system_id") or "")
            sid_b = (db.get("system_id") or "")
            human_response = None
            if "human" in sid_a.lower():
                human_response = da.get("content")
            elif "human" in sid_b.lower():
                human_response = db.get("content")
            if human_response:
                try:
                    anchors = extract_gap_anchors(human_response)
                    if anchors:
                        self._gap_anchors_cache[battle_id] = anchors
                except Exception as e:
                    logger.warning(f"提取battle {battle_id}的gap anchors时出错: {e}")

        logger.info(f"预提取了 {len(self._gap_anchors_cache)} 个battles的gap anchors")

    def _global_unique_gap_strings(self, exclude_battle_id: Optional[str] = None) -> List[str]:
        """全库去重 gap 句（与旧版 retrieve_gap_anchors 顺序一致：缓存迭代顺序）。"""
        unique_anchors: List[str] = []
        seen = set()
        for bid, anchors in self._gap_anchors_cache.items():
            if exclude_battle_id and bid == exclude_battle_id:
                continue
            for anchor in anchors:
                key = anchor.lower().strip()
                if key in seen or len(key) <= 10:
                    continue
                seen.add(key)
                unique_anchors.append(anchor)
        return unique_anchors

    def _collect_gap_anchor_candidates(self, target_battle: Dict) -> List[str]:
        """同 subfield 的 gap 句去重列表（不含本题）；若为空则回退全库去重，仍排除本题。"""
        target_id = target_battle.get("battle_id")
        target_subfield = self.get_subfield(target_battle)
        seen = set()
        out: List[str] = []
        for battle in self.all_battles:
            if self.get_subfield(battle) != target_subfield:
                continue
            bid = battle.get("battle_id")
            if not bid or bid == target_id or bid not in self._gap_anchors_cache:
                continue
            for anchor in self._gap_anchors_cache[bid]:
                key = anchor.lower().strip()
                if len(key) <= 10 or key in seen:
                    continue
                seen.add(key)
                out.append(anchor)
        if out:
            return out
        return self._global_unique_gap_strings(exclude_battle_id=target_id)

    def get_subfield(self, battle: Dict) -> str:
        """获取battle的subfield"""
        meta = battle.get("metadata") or {}
        if isinstance(meta, dict):
            sf = meta.get("subfield")
            if sf:
                return str(sf)
        tax = battle.get("taxonomy") or {}
        if isinstance(tax, dict) and tax.get("subfield"):
            return str(tax["subfield"])
        fields = battle.get("fields", [])
        if fields and isinstance(fields, list) and len(fields) > 0:
            field_str = fields[0]
            if isinstance(field_str, str) and ":" in field_str:
                return field_str.split(":")[-1].strip()
        return "unknown"

    def retrieve_structure_similar_cases(
        self,
        target_battle: Dict,
        max_cases: int = GROUP_S_SIZE
    ) -> List[Dict]:
        cases, _ = self._retrieve_structure_similar_cases_with_metrics(target_battle, max_cases)
        return cases

    def _retrieve_structure_similar_cases_with_metrics(
        self,
        target_battle: Dict,
        max_cases: int = GROUP_S_SIZE
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        target_id = target_battle.get("battle_id")
        if target_id not in self._structure_cache:
            return [], {}

        target_graph, _ = self._structure_cache[target_id]
        target_subfield = self.get_subfield(target_battle)

        similarities: List[Tuple[float, Dict]] = []
        for battle in self.all_battles:
            battle_id = battle.get("battle_id")
            if battle_id == target_id:
                continue
            if self.get_subfield(battle) != target_subfield:
                continue
            if battle_id in self._structure_cache:
                graph, _ = self._structure_cache[battle_id]
                similarity = compute_graph_similarity(target_graph, graph)
                similarities.append((similarity, battle))

        similarities.sort(key=lambda x: x[0], reverse=True)

        if not similarities:
            return [], {}

        if self.diversity_retrieval:
            pool_n = min(len(similarities), max_cases * self.pool_mult_s)
            pool = similarities[:pool_n]
            selected = _mmr_select(
                pool,
                max_cases,
                self._pair_sim_structure,
                self.mmr_lambda,
            )
        else:
            selected = [b for _, b in similarities[:max_cases]]

        avg_s = _avg_pairwise(selected, self._pair_sim_structure)
        return selected, {"avg_pairwise_example_similarity_S": avg_s}

    def _compute_semantic_similarity(
        self,
        query1: str,
        query2: str
    ) -> float:
        prompt = f"""You are an expert at evaluating semantic similarity between research queries.

Given two research queries, rate their semantic similarity on a scale from 0.0 to 1.0, where:
- 1.0 means the queries are about the exact same research topic/concept
- 0.8-0.9 means very similar topics with minor differences
- 0.6-0.7 means related topics but with notable differences
- 0.4-0.5 means somewhat related but distinct topics
- 0.0-0.3 means unrelated topics

Query 1: {query1}

Query 2: {query2}

Respond with ONLY a number between 0.0 and 1.0 (e.g., 0.85), no explanation needed."""

        try:
            response = self.llm.generate(
                prompt,
                temperature=0.1
            )
            match = re.search(r'0?\.\d+|1\.0|0', response.strip())
            if match:
                similarity = float(match.group())
                return max(0.0, min(1.0, similarity))
            logger.warning(f"无法解析相似度分数: {response}")
            return 0.0
        except Exception as e:
            logger.warning(f"计算语义相似度时出错: {e}")
            return self._compute_text_similarity_fallback(query1, query2)

    def _compute_text_similarity_fallback(
        self,
        query1: str,
        query2: str
    ) -> float:
        keywords1 = set(query1.lower().split())
        keywords2 = set(query2.lower().split())
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        return intersection / union if union > 0 else 0.0

    def retrieve_content_similar_cases(
        self,
        target_battle: Dict,
        max_cases: int = GROUP_C_SIZE
    ) -> List[Dict]:
        cases, _ = self._retrieve_content_similar_cases_with_metrics(target_battle, max_cases)
        return cases

    def _retrieve_content_similar_cases_with_metrics(
        self,
        target_battle: Dict,
        max_cases: int = GROUP_C_SIZE
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        target_query = battle_topic_text(target_battle)
        target_subfield = self.get_subfield(target_battle)

        if not target_query:
            return [], {}

        candidate_battles: List[Dict] = []
        for battle in self.all_battles:
            battle_id = battle.get("battle_id")
            if battle_id == target_battle.get("battle_id"):
                continue
            if self.get_subfield(battle) != target_subfield:
                continue
            battle_query = battle_topic_text(battle)
            if not battle_query:
                continue
            candidate_battles.append(battle)

        if not candidate_battles:
            return [], {}

        text_similarities: List[Tuple[float, Dict]] = []
        for battle in candidate_battles:
            battle_query = battle_topic_text(battle)
            text_sim = self._compute_text_similarity_fallback(target_query, battle_query)
            text_similarities.append((text_sim, battle))

        text_similarities.sort(key=lambda x: x[0], reverse=True)
        llm_pool_size = max_cases * 3
        top_candidates = [battle for _, battle in text_similarities[:llm_pool_size]]

        if not top_candidates:
            return [], {}

        logger.debug(f"使用LLM计算 {len(top_candidates)} 个候选cases的语义相似度...")
        similarities: List[Tuple[float, Dict]] = []
        for battle in top_candidates:
            battle_query = battle_topic_text(battle)
            similarity = self._compute_semantic_similarity(target_query, battle_query)
            similarities.append((similarity, battle))

        similarities.sort(key=lambda x: x[0], reverse=True)

        if self.diversity_retrieval:
            pool = similarities
            selected = _mmr_select(
                pool,
                max_cases,
                self._pair_sim_content,
                self.mmr_lambda,
            )
        else:
            selected = [b for _, b in similarities[:max_cases]]

        avg_c = _avg_pairwise(selected, self._pair_sim_content)
        return selected, {"avg_pairwise_example_similarity_C": avg_c}

    def paired_calibration_redundancy(
        self,
        target_battle: Dict,
        max_cases_s: int = GROUP_S_SIZE,
        max_cases_c: int = GROUP_C_SIZE,
    ) -> Dict[str, Any]:
        """
        在**同一组候选、同一相关分排序**下，同时计算 Group S / C / G 的
        top-k 选例与 MMR 选例的**平均两两相似度**（越高 = 示例间越冗余、越不多样）。

        用于公平对比：``mean_pairwise_*_mmr`` 应 **≤** ``mean_pairwise_*_topk``，
        因而 ``calibration_redundancy_drop_* = topk - mmr`` 多为 **正**（MMR 降低冗余）。

        Group G 的相关分为 topic 与 gap 句的 embedding/jaccard 相似度（无额外 LLM）。

        不依赖 ``self.diversity_retrieval``（两种策略在同一函数内并行算出）。
        """
        out: Dict[str, Any] = {}

        # ----- Group S -----
        target_id = target_battle.get("battle_id")
        if target_id not in self._structure_cache:
            out["mean_pairwise_S_topk"] = None
            out["mean_pairwise_S_mmr"] = None
            out["n_pool_S"] = 0
        else:
            target_graph, _ = self._structure_cache[target_id]
            target_subfield = self.get_subfield(target_battle)
            similarities: List[Tuple[float, Dict]] = []
            for battle in self.all_battles:
                battle_id = battle.get("battle_id")
                if battle_id == target_id:
                    continue
                if self.get_subfield(battle) != target_subfield:
                    continue
                if battle_id in self._structure_cache:
                    graph, _ = self._structure_cache[battle_id]
                    similarities.append(
                        (compute_graph_similarity(target_graph, graph), battle)
                    )
            similarities.sort(key=lambda x: x[0], reverse=True)
            out["n_pool_S"] = len(similarities)
            if not similarities:
                out["mean_pairwise_S_topk"] = None
                out["mean_pairwise_S_mmr"] = None
            else:
                topk_s = [b for _, b in similarities[:max_cases_s]]
                pool_n = min(len(similarities), max_cases_s * self.pool_mult_s)
                pool = similarities[:pool_n]
                mmr_s = _mmr_select(
                    pool,
                    max_cases_s,
                    self._pair_sim_structure,
                    self.mmr_lambda,
                )
                out["mean_pairwise_S_topk"] = _avg_pairwise(topk_s, self._pair_sim_structure)
                out["mean_pairwise_S_mmr"] = _avg_pairwise(mmr_s, self._pair_sim_structure)

        # ----- Group C（单次 LLM 打分，再分 top-k / MMR）-----
        target_query = battle_topic_text(target_battle)
        target_subfield = self.get_subfield(target_battle)
        if not target_query:
            out["mean_pairwise_C_topk"] = None
            out["mean_pairwise_C_mmr"] = None
            out["n_pool_C"] = 0
        else:
            candidate_battles: List[Dict] = []
            for battle in self.all_battles:
                battle_id = battle.get("battle_id")
                if battle_id == target_battle.get("battle_id"):
                    continue
                if self.get_subfield(battle) != target_subfield:
                    continue
                if battle_topic_text(battle):
                    candidate_battles.append(battle)
            text_similarities: List[Tuple[float, Dict]] = []
            for battle in candidate_battles:
                bq = battle_topic_text(battle)
                text_similarities.append(
                    (self._compute_text_similarity_fallback(target_query, bq), battle)
                )
            text_similarities.sort(key=lambda x: x[0], reverse=True)
            llm_pool_size = max_cases_c * 3
            top_candidates = [b for _, b in text_similarities[:llm_pool_size]]
            out["n_pool_C"] = len(top_candidates)
            if not top_candidates:
                out["mean_pairwise_C_topk"] = None
                out["mean_pairwise_C_mmr"] = None
            else:
                rel_list: List[Tuple[float, Dict]] = []
                for battle in top_candidates:
                    bq = battle_topic_text(battle)
                    rel_list.append(
                        (self._compute_semantic_similarity(target_query, bq), battle)
                    )
                rel_list.sort(key=lambda x: x[0], reverse=True)
                topk_c = [b for _, b in rel_list[:max_cases_c]]
                mmr_c = _mmr_select(
                    rel_list,
                    max_cases_c,
                    self._pair_sim_content,
                    self.mmr_lambda,
                )
                out["mean_pairwise_C_topk"] = _avg_pairwise(topk_c, self._pair_sim_content)
                out["mean_pairwise_C_mmr"] = _avg_pairwise(mmr_c, self._pair_sim_content)

        # ----- Group G（topic–gap 相关度：embedding/jaccard；无 LLM）-----
        max_cases_g = GROUP_G_SIZE
        candidates_g = self._collect_gap_anchor_candidates(target_battle)
        tq_g = battle_topic_text(target_battle) or ""
        if not candidates_g:
            out["mean_pairwise_G_topk"] = None
            out["mean_pairwise_G_mmr"] = None
            out["n_pool_G"] = 0
        else:
            ranked_g: List[Tuple[float, Dict]] = []
            if tq_g:
                for anchor in candidates_g:
                    ranked_g.append(
                        (self._sim_gap_strings(tq_g, anchor), {"text": anchor})
                    )
                ranked_g.sort(key=lambda x: x[0], reverse=True)
            else:
                for anchor in sorted(candidates_g, key=lambda x: x.lower()):
                    ranked_g.append((0.0, {"text": anchor}))
            out["n_pool_G"] = len(ranked_g)
            pool_ng = min(len(ranked_g), max_cases_g * self.pool_mult_g)
            pool_g = ranked_g[:pool_ng]
            topk_g = [b for _, b in ranked_g[:max_cases_g]]
            mmr_g = _mmr_select(
                pool_g,
                max_cases_g,
                self._pair_sim_gap,
                self.mmr_lambda,
            )
            out["mean_pairwise_G_topk"] = _avg_pairwise(topk_g, self._pair_sim_gap)
            out["mean_pairwise_G_mmr"] = _avg_pairwise(mmr_g, self._pair_sim_gap)

        # 冗余下降（正 => MMR 更分散）；多样性指数（高 => 更分散）
        for grp in ("S", "C", "G"):
            tk = out.get(f"mean_pairwise_{grp}_topk")
            mm = out.get(f"mean_pairwise_{grp}_mmr")
            if tk is not None and mm is not None:
                out[f"calibration_redundancy_drop_{grp}"] = float(tk) - float(mm)
            else:
                out[f"calibration_redundancy_drop_{grp}"] = None
            if mm is not None:
                out[f"calibration_diversity_index_{grp}_mmr"] = 1.0 - float(mm)
            else:
                out[f"calibration_diversity_index_{grp}_mmr"] = None
            if tk is not None:
                out[f"calibration_diversity_index_{grp}_topk"] = 1.0 - float(tk)
            else:
                out[f"calibration_diversity_index_{grp}_topk"] = None

        out["mmr_lambda"] = self.mmr_lambda
        out["pool_mult_s"] = self.pool_mult_s
        out["pool_mult_g"] = self.pool_mult_g
        out["pair_sim_mode_effective"] = self._pair_sim_mode_effective
        return out

    def _retrieve_gap_anchors_with_metrics(
        self,
        target_battle: Dict,
        max_anchors: int = GROUP_G_SIZE,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        按当前 battle 的 topic 与 gap 句的相似度排序，top-k 或 MMR 选 Group G；
        返回 avg_pairwise_example_similarity_G（与 S/C 机制对齐）。
        """
        candidates = self._collect_gap_anchor_candidates(target_battle)
        if not candidates:
            return [], {}

        target_query = battle_topic_text(target_battle) or ""
        ranked: List[Tuple[float, Dict]] = []
        if target_query:
            for anchor in candidates:
                ranked.append(
                    (self._sim_gap_strings(target_query, anchor), {"text": anchor})
                )
            ranked.sort(key=lambda x: x[0], reverse=True)
        else:
            for anchor in sorted(candidates, key=lambda x: x.lower()):
                ranked.append((0.0, {"text": anchor}))

        if self.diversity_retrieval:
            pool_n = min(len(ranked), max_anchors * self.pool_mult_g)
            pool = ranked[:pool_n]
            selected_wrapped = _mmr_select(
                pool,
                max_anchors,
                self._pair_sim_gap,
                self.mmr_lambda,
            )
        else:
            selected_wrapped = [b for _, b in ranked[:max_anchors]]

        selected = [d["text"] for d in selected_wrapped]
        avg_g = _avg_pairwise(selected_wrapped, self._pair_sim_gap)
        return selected, {"avg_pairwise_example_similarity_G": avg_g}

    def build_demonstration_text(
        self,
        battle: Dict,
        outcome: Optional[Dict[str, str]] = None
    ) -> str:
        response_a, response_b = get_battle_responses(battle)
        response_a = response_a or ""
        response_b = response_b or ""

        demo_text = f"Query: {battle_topic_text(battle)}\n\n"
        demo_text += f"Draft A:\n{response_a}\n\n"
        demo_text += f"Draft B:\n{response_b}\n\n"

        if outcome:
            demo_text += "Expert Outcomes:\n"
            for dim, result in outcome.items():
                demo_text += f"  {dim}: {result}\n"

        return demo_text

    def build_context(
        self,
        target_battle: Dict,
        expert_outcomes: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        structure_cases, m_s = self._retrieve_structure_similar_cases_with_metrics(
            target_battle, GROUP_S_SIZE
        )
        content_cases, m_c = self._retrieve_content_similar_cases_with_metrics(
            target_battle, GROUP_C_SIZE
        )
        gap_anchors, m_g = self._retrieve_gap_anchors_with_metrics(
            target_battle, GROUP_G_SIZE
        )

        retrieval_metrics: Dict[str, Any] = {
            **m_s,
            **m_c,
            **m_g,
            "diversity_retrieval": self.diversity_retrieval,
            "mmr_lambda": self.mmr_lambda,
            "pool_mult_g": self.pool_mult_g,
            "pair_sim_mode_requested": self.pair_sim_mode_requested,
            "pair_sim_mode_effective": self._pair_sim_mode_effective,
        }

        context = """You are an expert evaluator for literature reviews. Your task is to compare two draft literature reviews (Draft A and Draft B) and make judgments across five dimensions.

"""

        context += "Evaluation Dimensions:\n"
        for dim, description in DIMENSIONS.items():
            context += f"{dim}: {description}\n"

        context += "\nFor each dimension, you must choose one of: A, B, Tie, or BothBad.\n"
        context += "- A: Draft A is better\n"
        context += "- B: Draft B is better\n"
        context += "- Tie: Both drafts are equally good\n"
        context += "- BothBad: Neither draft is acceptable\n\n"

        if structure_cases:
            context += "=== Structure-Similar Examples (for D3) ===\n"
            for i, case in enumerate(structure_cases[:MAX_DEMONSTRATIONS_PER_GROUP], 1):
                case_id = case.get("battle_id")
                case_expert_outcomes = None
                if expert_outcomes and case_id in expert_outcomes:
                    case_expert_outcomes = expert_outcomes[case_id]
                elif not expert_outcomes or case_id not in expert_outcomes:
                    case_dim_results = case.get("dimension_results", {})
                    if case_dim_results:
                        case_expert_outcomes = case_dim_results
                demo_text = self.build_demonstration_text(case, outcome=case_expert_outcomes)
                context += f"\nExample {i}:\n{demo_text}\n"

        if content_cases:
            context += "\n=== Content-Similar Examples (for D1/D2) ===\n"
            for i, case in enumerate(content_cases[:MAX_DEMONSTRATIONS_PER_GROUP], 1):
                case_id = case.get("battle_id")
                case_expert_outcomes = None
                if expert_outcomes and case_id in expert_outcomes:
                    case_expert_outcomes = expert_outcomes[case_id]
                elif not expert_outcomes or case_id not in expert_outcomes:
                    case_dim_results = case.get("dimension_results", {})
                    if case_dim_results:
                        case_expert_outcomes = case_dim_results
                demo_text = self.build_demonstration_text(case, outcome=case_expert_outcomes)
                context += f"\nExample {i}:\n{demo_text}\n"

        if gap_anchors:
            context += "\n=== Gap Anchors (for D4) ===\n"
            context += "Examples of meaningful research gaps and future directions from expert-written reviews:\n"
            for i, anchor in enumerate(gap_anchors[:MAX_DEMONSTRATIONS_PER_GROUP], 1):
                context += f"{i}. {anchor}\n"

        context += "\n=== Current Battle to Evaluate ===\n"
        response_a, response_b = get_battle_responses(target_battle)
        response_a = response_a or ""
        response_b = response_b or ""

        context += f"Query: {battle_topic_text(target_battle)}\n\n"
        context += f"Draft A:\n{response_a}\n\n"
        context += f"Draft B:\n{response_b}\n\n"

        sample_lines = [f'  "{dim}": "A"' for dim in DIMENSIONS.keys()]
        context += (
            "Please evaluate both drafts and provide your judgments as a JSON object with the following format:\n{\n"
            + ",\n".join(sample_lines)
            + "\n}\n\nYou must provide a decision for each dimension. Do not use \"NonDecisive\" or any other values."
        )

        return context, retrieval_metrics
