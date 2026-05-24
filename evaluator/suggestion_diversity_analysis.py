"""
独立分析：按 topic/query 聚合 judge 产出的建议文本，计算 research-suggestion diversity。

不在 retrieval 或 ContextBuilder 中调用；读 evaluator results JSON + battles 元数据。

可直接运行：
  python -m evaluator.suggestion_diversity_analysis --results ... --battles-file ...
或（仓库根目录下）：
  python evaluator/suggestion_diversity_analysis.py --results ... --battles-file ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 允许 ``python evaluator/suggestion_diversity_analysis.py`` 直接执行（非 -m）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from loguru import logger
from sklearn.cluster import AgglomerativeClustering

from evaluator.config import (
    API_CALL_DELAY,
    DEFAULT_MODEL,
    DEFAULT_PLATFORM,
    LOCAL_EMBEDDING_MODEL,
    SUGGESTION_CLUSTER_COSINE_THRESHOLD,
)
from evaluator.data_loader import get_battle_responses

# 抽取用：避免超长 draft 撑爆 context
_DRAFT_CHAR_BUDGET = 14000

EXTRACTION_PROMPT = """You extract research-oriented suggestions from ONE literature review draft.

What to extract (as separate items):
- Gaps in existing work, limitations, or missing angles
- Concrete future research directions or open questions
- Non-obvious next steps a researcher could take

Rules:
- Each item must be ONE standalone English sentence (or short phrase) describing ONE suggestion.
- Do NOT copy the judge; only use the DRAFT text below.
- Do NOT summarize the whole review; only suggestion-like content.
- If the draft has no such content, return an empty list.

Output ONLY valid JSON (no markdown fences):
{{"items": ["...", "..."]}}

DRAFT:
---
{draft}
---
"""


def _pick_winning_draft_letter(
    judge_record: Dict[str, Any],
    primary_dim: str = "D4",
    fallback_dim: str = "D5",
) -> Optional[str]:
    """
    用 LitJudge 分项结果选择「更优」draft：默认先看 D4（与 gaps/suggestions 语义一致），
    Tie/BothBad 时回退 D5（Overall）。
    """
    for dim in (primary_dim, fallback_dim):
        v = judge_record.get(dim)
        if v in ("A", "B"):
            return v
    return None


def _parse_extraction_json(raw: str) -> List[str]:
    import re

    text = (raw or "").strip()
    if not text:
        return []
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = obj.get("items")
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        s = str(x).strip()
        if len(s) > 5:
            out.append(s)
    return out


def extract_research_suggestions_with_llm(
    draft_text: str,
    llm: Any,
    temperature: float = 0.0,
) -> List[str]:
    """单次调用：从一篇 draft 中抽取若干条 research suggestion 字符串。"""
    body = (draft_text or "")[:_DRAFT_CHAR_BUDGET]
    prompt = EXTRACTION_PROMPT.format(draft=body)
    raw = llm.generate(prompt, temperature=temperature)
    return _parse_extraction_json(raw)


def _battles_by_id(battles: List[Dict]) -> Dict[str, Dict]:
    return {str(b["battle_id"]): b for b in battles if b.get("battle_id")}


def collect_topic_texts_from_winning_drafts(
    judge_outcomes: Dict[str, Dict[str, Any]],
    battles: List[Dict],
    llm: Any,
    primary_dim: str = "D4",
    fallback_dim: str = "D5",
    throttle_s: float = API_CALL_DELAY,
    extraction_max_workers: int = 8,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    对每个 battle：按 LitJudge 选 A/B → 取对应 draft → LLM 抽取 items →
    按 topic_id 汇总所有 items（同一 topic 下多 battle 的 suggestions 合并）。

    extraction_max_workers>1 时用线程池并行调用 API；为 1 时串行并在请求间 sleep 节流。
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    bmap = _battle_topic_map(battles)
    by_id = _battles_by_id(battles)
    topic_texts: Dict[str, List[str]] = defaultdict(list)
    skipped: Dict[str, int] = defaultdict(int)

    jobs: List[Tuple[str, str]] = []
    for battle_id, rec in judge_outcomes.items():
        bid = str(battle_id)
        tid = bmap.get(bid)
        if not tid:
            skipped["no_topic"] += 1
            continue
        letter = _pick_winning_draft_letter(rec, primary_dim, fallback_dim)
        if not letter:
            skipped["no_clear_winner"] += 1
            continue
        battle = by_id.get(bid)
        if not battle:
            skipped["no_battle_row"] += 1
            continue
        da, db = get_battle_responses(battle)
        draft = da if letter == "A" else db
        if not draft or not str(draft).strip():
            skipped["empty_draft"] += 1
            continue
        jobs.append((tid, str(draft)))

    def _extract_pair(tid: str, draft: str) -> Tuple[str, List[str]]:
        return tid, extract_research_suggestions_with_llm(draft, llm)

    n_jobs = len(jobs)
    if extraction_max_workers <= 1:
        iterator = tqdm(jobs, desc="LLM extract (serial)") if tqdm else jobs
        for idx, (tid, draft) in enumerate(iterator):
            if idx > 0 and throttle_s > 0:
                time.sleep(throttle_s)
            _, items = _extract_pair(tid, draft)
            if not items:
                skipped["no_items_extracted"] += 1
                continue
            for itx in items:
                topic_texts[tid].append(itx)
    else:
        workers = max(1, min(extraction_max_workers, n_jobs or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_pair, tid, d) for tid, d in jobs]
            done_iter = as_completed(futures)
            if tqdm:
                done_iter = tqdm(done_iter, total=len(futures), desc="LLM extract (parallel)")
            for fut in done_iter:
                tid, items = fut.result()
                if not items:
                    skipped["no_items_extracted"] += 1
                    continue
                for itx in items:
                    topic_texts[tid].append(itx)

    diag = {
        "primary_dim": primary_dim,
        "fallback_dim": fallback_dim,
        "extraction_max_workers": extraction_max_workers,
        "n_extraction_jobs": n_jobs,
        "skipped_counts": dict(skipped),
        "n_battles_in_outcomes": len(judge_outcomes),
    }
    return dict(topic_texts), diag


def _normalized_cluster_entropy(labels: np.ndarray) -> Tuple[float, int]:
    """归一化簇熵：H / log(K)，K 为簇数；主表用此标量。"""
    n = len(labels)
    if n == 0:
        return 0.0, 0
    uniq, counts = np.unique(labels, return_counts=True)
    k = len(uniq)
    if k <= 1:
        return 0.0, int(k)
    p = counts.astype(np.float64) / n
    h = -np.sum(p * np.log(p + 1e-12))
    h_norm = float(h / math.log(k))
    return h_norm, int(k)


def _entropy_weighted_concentration(h_norm: float, n_clusters: int) -> float:
    """
    Popular-topic concentration（entropy-weighted）：
      concentration = (1 - effective_cluster_ratio) * log(K)
      effective_cluster_ratio = exp(H) / K = K^(h_norm - 1)
    其中 K 为语义簇数，H = h_norm * log(K)。

    直觉：在可选语义方向更多（K 大）时，若仍呈现集中，惩罚应更大。
    """
    if n_clusters <= 1:
        return 0.0
    k = float(n_clusters)
    effective_ratio = k ** (float(h_norm) - 1.0)
    return float((1.0 - effective_ratio) * math.log(k))


def diversity_for_texts(
    texts: List[str],
    embedding_model_name: str = LOCAL_EMBEDDING_MODEL,
    distance_threshold: float = SUGGESTION_CLUSTER_COSINE_THRESHOLD,
    sentence_model: Any = None,
) -> Dict[str, Any]:
    """
    对同一 query 下的多条建议文本：嵌入 → 凝聚层次聚类（余弦距离阈值）→ 归一化簇熵。
    """
    texts = [t.strip() for t in texts if t and str(t).strip()]
    if not texts:
        return {
            "normalized_cluster_entropy": None,
            "num_semantic_clusters": 0,
            "popular_topic_concentration": None,
            "n_texts": 0,
        }
    if len(texts) == 1:
        return {
            "normalized_cluster_entropy": 0.0,
            "num_semantic_clusters": 1,
            "popular_topic_concentration": 0.0,
            "n_texts": 1,
        }

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "需要 sentence-transformers 以计算 research-suggestion diversity"
        ) from e

    model = sentence_model or SentenceTransformer(embedding_model_name)
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X)
    h_norm, n_clust = _normalized_cluster_entropy(labels)
    pop_conc = _entropy_weighted_concentration(h_norm, n_clust)

    return {
        "normalized_cluster_entropy": h_norm,
        "num_semantic_clusters": n_clust,
        "popular_topic_concentration": pop_conc,
        "n_texts": len(texts),
    }


def _battle_topic_map(battles: List[Dict]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for b in battles:
        bid = b.get("battle_id")
        tid = b.get("topic_id")
        if bid and tid:
            m[str(bid)] = str(tid)
    return m


def _run_diversity_on_topic_texts(
    topic_texts: Dict[str, List[str]],
    embedding_model_name: str,
    distance_threshold: float,
) -> Tuple[Dict[str, Any], List[float], Any]:
    """
    对 topic_id -> 多条文本 聚类算熵；返回 per_topic、熵列表、sentence 模型（若已加载）。
    """
    needs_embed = any(len(v) > 1 for v in topic_texts.values())
    st_model = None
    if needs_embed:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "需要 sentence-transformers 以计算 research-suggestion diversity"
            ) from e
        st_model = SentenceTransformer(embedding_model_name)

    per_topic: Dict[str, Any] = {}
    entropies: List[float] = []
    concentrations: List[float] = []
    total_n_texts = 0
    weighted_conc_sum = 0.0

    for tid, texts in topic_texts.items():
        d = diversity_for_texts(
            texts,
            embedding_model_name=embedding_model_name,
            distance_threshold=distance_threshold,
            sentence_model=st_model,
        )
        per_topic[tid] = d
        if d.get("normalized_cluster_entropy") is not None:
            entropies.append(d["normalized_cluster_entropy"])
        if d.get("popular_topic_concentration") is not None:
            concentrations.append(d["popular_topic_concentration"])
        n_t = int(d.get("n_texts") or 0)
        c_t = d.get("popular_topic_concentration")
        if n_t > 0 and c_t is not None:
            total_n_texts += n_t
            weighted_conc_sum += float(c_t) * n_t

    concentration_micro = (
        float(weighted_conc_sum / total_n_texts) if total_n_texts > 0 else None
    )

    return per_topic, entropies, concentrations, concentration_micro, st_model


def compute_run_level_research_suggestion_diversity(
    judge_outcomes: Dict[str, Dict[str, Any]],
    battles: List[Dict],
    embedding_model_name: str = LOCAL_EMBEDDING_MODEL,
    distance_threshold: float = SUGGESTION_CLUSTER_COSINE_THRESHOLD,
    text_source: str = "judge_output",
    *,
    extraction_llm: Any = None,
    extraction_model: str = DEFAULT_MODEL,
    extraction_platform: str = DEFAULT_PLATFORM,
    primary_dim: str = "D4",
    fallback_dim: str = "D5",
    extraction_max_workers: int = 8,
) -> Dict[str, Any]:
    """
    query（topic）级：汇总文本后算每 topic 的 normalized_cluster_entropy，再对 topic 平均。

    text_source:
      - ``judge_output``: 每条 battle 用 judge 落盘的 suggestion_text / raw（旧行为）
      - ``winning_draft_llm``: 按 LitJudge 在 D4（或回退 D5）上选 A/B，从**胜方 draft** 用 LLM 抽取 research suggestions
    """
    extraction_diag: Optional[Dict[str, Any]] = None

    if text_source == "winning_draft_llm":
        if not battles:
            raise ValueError("winning_draft_llm 需要 battles（含 draft_a/draft_b content）")
        llm = extraction_llm
        if llm is None:
            from evaluator.utils import LLM

            llm = LLM(model_name=extraction_model, platform=extraction_platform)
        topic_texts, extraction_diag = collect_topic_texts_from_winning_drafts(
            judge_outcomes,
            battles,
            llm,
            primary_dim=primary_dim,
            fallback_dim=fallback_dim,
            extraction_max_workers=extraction_max_workers,
        )
    else:
        bmap = _battle_topic_map(battles)
        topic_texts = defaultdict(list)
        for battle_id, rec in judge_outcomes.items():
            tid = bmap.get(str(battle_id))
            if not tid:
                continue
            txt = rec.get("suggestion_text_for_diversity") or rec.get("raw_judge_response")
            if txt and str(txt).strip():
                topic_texts[tid].append(str(txt).strip())
        topic_texts = dict(topic_texts)

    per_topic, entropies, concentrations, concentration_micro, _ = _run_diversity_on_topic_texts(
        topic_texts, embedding_model_name, distance_threshold
    )
    run_mean = float(np.mean(entropies)) if entropies else None
    pop_conc_mean = float(np.mean(concentrations)) if concentrations else None

    out: Dict[str, Any] = {
        "research_suggestion_diversity_cluster_entropy_mean": run_mean,
        "popular_topic_concentration_mean": pop_conc_mean,
        "popular_topic_concentration_micro": concentration_micro,
        "per_topic": per_topic,
        "n_topics_with_text": len(entropies),
        "embedding_model": embedding_model_name,
        "cluster_cosine_distance_threshold": distance_threshold,
        "text_source": text_source,
    }
    if extraction_diag is not None:
        out["draft_extraction_diagnostic"] = extraction_diag
    return out


def main_cli():
    parser = argparse.ArgumentParser(
        description="Compute research-suggestion diversity from evaluator results JSON."
    )
    parser.add_argument("--results", type=Path, required=True, help="evaluator_results_*.json")
    parser.add_argument("--battles-file", type=Path, help="battles.jsonl for topic_id mapping + drafts")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path (default: results path with _suggestion_diversity suffix)",
    )
    parser.add_argument("--embedding-model", type=str, default=LOCAL_EMBEDDING_MODEL)
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=SUGGESTION_CLUSTER_COSINE_THRESHOLD,
    )
    parser.add_argument(
        "--text-source",
        type=str,
        choices=["judge_output", "winning_draft_llm"],
        default="winning_draft_llm",
        help="winning_draft_llm: LitJudge D4/D5 胜方 draft → LLM 抽取 suggestions（推荐）；judge_output: 用 judge 落盘文本",
    )
    parser.add_argument(
        "--extraction-model",
        type=str,
        default=DEFAULT_MODEL,
        help="LLM for extraction when text-source=winning_draft_llm",
    )
    parser.add_argument(
        "--extraction-platform",
        type=str,
        default=DEFAULT_PLATFORM,
        help="API platform for extraction LLM",
    )
    parser.add_argument("--primary-dim", type=str, default="D4", help="首选分项决定 A/B（默认 D4）")
    parser.add_argument("--fallback-dim", type=str, default="D5", help="Tie/BothBad 时回退分项")
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=8,
        help="并行抽取 LLM 的线程数；1 表示串行并带请求间延迟",
    )
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
    judge_outcomes = data.get("judge_outcomes") or {}

    battles: List[Dict] = []
    if args.battles_file and args.battles_file.exists():
        from evaluator.data_loader import load_battles

        battles = load_battles(args.battles_file)
    else:
        logger.warning("未提供 battles-file：topic 映射为空；winning_draft_llm 需要 battles-file")

    out = compute_run_level_research_suggestion_diversity(
        judge_outcomes,
        battles,
        embedding_model_name=args.embedding_model,
        distance_threshold=args.distance_threshold,
        text_source=args.text_source,
        extraction_model=args.extraction_model,
        extraction_platform=args.extraction_platform,
        primary_dim=args.primary_dim,
        fallback_dim=args.fallback_dim,
        extraction_max_workers=args.extraction_workers,
    )

    out_path = args.output
    if not out_path:
        out_path = args.results.with_name(
            args.results.stem + "_suggestion_diversity.json"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main_cli()
