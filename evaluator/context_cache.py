"""
LitJudge 上下文缓存：按 battle 保存 build_context 产出的 user prompt 与 retrieval_metrics。

命中缓存时跳过 ContextBuilder.build_context，从而避免 Group C 的语义相似度 LLM 等检索开销。
无效条件：battle_id、topic 文本、检索超参、expert outcomes 任一变化则自动失效（需重建）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from loguru import logger

CONTEXT_CACHE_VERSION = 1


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def topic_fingerprint(battle: Dict[str, Any]) -> str:
    try:
        from .data_loader import battle_topic_text
    except ImportError:
        from evaluator.data_loader import battle_topic_text
    t = battle_topic_text(battle) or ""
    return _sha16(t)


def expert_outcomes_fingerprint(expert_outcomes: Optional[Dict[str, Any]]) -> str:
    if not expert_outcomes:
        return "none"
    try:
        payload = json.dumps(expert_outcomes, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        payload = str(expert_outcomes)
    return _sha16(payload)


def retrieval_settings_fingerprint(context_builder: Any) -> str:
    """与 build_context 相关的检索/多样性配置；变化则视为不同缓存。"""
    try:
        from .config import GROUP_S_SIZE, GROUP_C_SIZE, GROUP_G_SIZE
    except ImportError:
        from evaluator.config import GROUP_S_SIZE, GROUP_C_SIZE, GROUP_G_SIZE
    ab = getattr(context_builder, "all_battles", None) or []
    parts = [
        str(CONTEXT_CACHE_VERSION),
        str(len(ab)),
        str(context_builder.diversity_retrieval),
        str(context_builder.mmr_lambda),
        context_builder.pair_sim_mode_requested,
        str(context_builder.pool_mult_s),
        str(context_builder.pool_mult_g),
        str(context_builder.embedding_model_name),
        str(GROUP_S_SIZE),
        str(GROUP_C_SIZE),
        str(GROUP_G_SIZE),
    ]
    return _sha16("|".join(parts))


def make_cache_record(
    battle_id: str,
    topic_fp: str,
    retrieval_fp: str,
    expert_fp: str,
    context: str,
    retrieval_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "v": CONTEXT_CACHE_VERSION,
        "battle_id": battle_id,
        "topic_fp": topic_fp,
        "retrieval_fp": retrieval_fp,
        "expert_fp": expert_fp,
        "context": context,
        "retrieval_metrics": retrieval_metrics,
    }


def _entry_matches(
    entry: Dict[str, Any],
    battle_id: str,
    topic_fp: str,
    retrieval_fp: str,
    expert_fp: str,
) -> bool:
    if entry.get("v") != CONTEXT_CACHE_VERSION:
        return False
    if str(entry.get("battle_id")) != str(battle_id):
        return False
    if entry.get("topic_fp") != topic_fp:
        return False
    if entry.get("retrieval_fp") != retrieval_fp:
        return False
    if entry.get("expert_fp") != expert_fp:
        return False
    if "context" not in entry or "retrieval_metrics" not in entry:
        return False
    return True


def load_cache_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """JSONL：每行一条；同一 battle_id 重复时后者覆盖前者。"""
    index: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Context cache skip bad line {line_no} in {path}: {e}")
                continue
            bid = obj.get("battle_id")
            if bid is not None:
                index[str(bid)] = obj
    return index


def append_cache_record(path: Path, record: Dict[str, Any], file_lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with file_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def try_get_cached(
    index: Dict[str, Dict[str, Any]],
    battle: Dict[str, Any],
    context_builder: Any,
    expert_outcomes: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    battle_id = battle.get("battle_id")
    if not battle_id:
        return None
    bid = str(battle_id)
    entry = index.get(bid)
    if not entry:
        return None
    tp = topic_fingerprint(battle)
    rp = retrieval_settings_fingerprint(context_builder)
    ep = expert_outcomes_fingerprint(expert_outcomes)
    if not _entry_matches(entry, bid, tp, rp, ep):
        return None
    return entry["context"], entry["retrieval_metrics"]
