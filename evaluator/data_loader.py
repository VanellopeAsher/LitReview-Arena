"""
Data loading for the Evaluator Agent.

Battles must embed `draft_a` / `draft_b` with `content` (e.g. `battles.jsonl`).
Subset: by default only battles listed in a prior aggregated JSON's `judge_outcomes` keys;
`--battle-ids-file` or `--all-battles` override this.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

from loguru import logger

from .config import BATTLES_FILE, EXPERT_OUTCOMES_FILE, DEFAULT_AGGREGATED_SUBSET_FILE


def battle_topic_text(battle: Dict) -> str:
    """Topic string for retrieval (LitReviewBench uses `topic_query`)."""
    return (battle.get("topic_query") or battle.get("query") or "").strip()


def load_battle_ids_from_file(battle_ids_file: Path) -> Set[str]:
    """Load battle IDs from JSON (selected_battles / sampled_battle_ids / array)."""
    logger.info(f"从文件加载battle IDs: {battle_ids_file}")

    with open(battle_ids_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "selected_battles" in data:
        sampled_ids = set(data["selected_battles"])
    elif "sampled_battle_ids" in data:
        sampled_ids = set(data["sampled_battle_ids"])
    elif isinstance(data, list):
        sampled_ids = set(data)
    else:
        raise ValueError(
            f"无法从文件 {battle_ids_file} 中提取battle IDs。"
            f"文件应包含 'selected_battles' 或 'sampled_battle_ids' 字段，或直接是数组。"
        )

    logger.info(f"加载了 {len(sampled_ids)} 个battle IDs")

    return sampled_ids


def load_battle_ids_from_aggregated_json(aggregated_file: Path) -> Set[str]:
    """Battle IDs = keys of `judge_outcomes` in an evaluator aggregated/results JSON."""
    aggregated_file = Path(aggregated_file)
    logger.info(f"从聚合结果提取 battle IDs: {aggregated_file}")
    with open(aggregated_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    jo = data.get("judge_outcomes")
    if not isinstance(jo, dict):
        raise ValueError(f"{aggregated_file} 缺少有效的 'judge_outcomes' 对象")
    ids = set(jo.keys())
    logger.info(f"共 {len(ids)} 个 battle_id")
    return ids


def load_battles(battles_file: Path = None) -> List[Dict]:
    """
    Load battles: `.jsonl` one object per line, or a single JSON array in `.json`.
    """
    if battles_file is None:
        battles_file = BATTLES_FILE

    battles_file = Path(battles_file)
    logger.info(f"加载battle数据: {battles_file}")

    if battles_file.suffix.lower() == ".jsonl":
        battles = []
        with open(battles_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    battles.append(json.loads(line))
    else:
        with open(battles_file, "r", encoding="utf-8") as f:
            battles = json.load(f)
        if not isinstance(battles, list):
            raise ValueError(f"Expected JSON array in {battles_file}")

    logger.info(f"总共 {len(battles)} 条battle记录")
    return battles


def get_battle_responses(battle: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Return (draft_a_text, draft_b_text) from embedded `draft_a` / `draft_b` content."""
    da = battle.get("draft_a") or {}
    db = battle.get("draft_b") or {}
    ca = da.get("content")
    cb = db.get("content")
    if isinstance(ca, str) and isinstance(cb, str) and ca.strip() and cb.strip():
        return ca, cb
    return None, None


def filter_sampled_battles(
    all_battles: List[Dict],
    sampled_ids: Set[str],
) -> List[Dict]:
    """Keep battles whose id is in sampled_ids and both embedded drafts are non-empty."""
    filtered_battles = []

    for battle in all_battles:
        battle_id = battle.get("battle_id")

        if battle_id not in sampled_ids:
            continue

        response_a, response_b = get_battle_responses(battle)
        if response_a and response_b:
            filtered_battles.append(battle)

    logger.info(f"筛选出 {len(filtered_battles)} 个有效的 battles")
    return filtered_battles


def all_battle_ids(battles: List[Dict]) -> Set[str]:
    return {b["battle_id"] for b in battles if b.get("battle_id")}


def load_expert_outcomes_jsonl(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Load expert_outcomes.jsonl -> {battle_id: {D1: ..., ...}}."""
    if path is None:
        path = EXPERT_OUTCOMES_FILE
    path = Path(path)
    out: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        logger.info(f"未找到 expert_outcomes 文件 {path}，跳过")
        return out

    logger.info(f"加载专家标注: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            bid = rec.get("battle_id")
            oc = rec.get("outcomes") or {}
            if bid and oc:
                out[bid] = oc
    logger.info(f"加载了 {len(out)} 条 expert outcomes")
    return out


def load_all_data(
    battles_file: Optional[Path] = None,
    battle_ids_file: Optional[Path] = None,
    aggregated_subset_file: Optional[Path] = None,
    use_all_battles: bool = False,
) -> Tuple[List[Dict], Set[str]]:
    """
    Load filtered battles and the id set used for filtering.

    Precedence: ``battle_ids_file`` > ``use_all_battles`` > aggregated JSON subset.

    Default: only battles whose IDs appear in ``aggregated_subset_file``'s ``judge_outcomes``
    (see ``DEFAULT_AGGREGATED_SUBSET_FILE`` in config).
    """
    battles_path = Path(battles_file) if battles_file else Path(BATTLES_FILE)

    all_battles = load_battles(battles_path)

    if battle_ids_file:
        sampled_ids = load_battle_ids_from_file(Path(battle_ids_file))
    elif use_all_battles:
        sampled_ids = all_battle_ids(all_battles)
        logger.info(f"--all-battles：使用全部 {len(sampled_ids)} 个 battle_id")
    else:
        agg_path = Path(
            aggregated_subset_file
            if aggregated_subset_file is not None
            else DEFAULT_AGGREGATED_SUBSET_FILE
        )
        if not agg_path.is_file():
            raise FileNotFoundError(
                f"未找到默认子集文件: {agg_path}\n"
                f"请放置该聚合 JSON，或传入 --aggregated-subset-file，或使用 --all-battles 评测全量。"
            )
        sampled_ids = load_battle_ids_from_aggregated_json(agg_path)

    filtered_battles = filter_sampled_battles(all_battles, sampled_ids)

    return filtered_battles, sampled_ids
