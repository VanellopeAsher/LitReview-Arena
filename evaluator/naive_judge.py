"""
Naive judge: rubric + current battle only.

No Group S/C/G demonstrations, no gap anchors, no structure/graph retrieval,
and no LLM-based semantic similarity (ContextBuilder 中的检索与相似度均不执行).
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from .config import DIMENSIONS
    from .data_loader import get_battle_responses, battle_topic_text
except ImportError:
    from evaluator.config import DIMENSIONS
    from evaluator.data_loader import get_battle_responses, battle_topic_text


class NaiveContextBuilder:
    """
    Drop-in replacement for ContextBuilder for Evaluator: 仅实现 ``build_context``，
    签名与 ContextBuilder 一致，便于 ``Evaluator`` 复用。
    """

    def __init__(
        self,
        all_battles: List[Dict],
        llm=None,
        model_name: str = "",
        platform: str = "",
    ):
        self.all_battles = all_battles
        # llm / model_name / platform 仅为与 ContextBuilder 构造参数兼容，本类不使用 LLM。

    def build_context(
        self,
        target_battle: Dict,
        expert_outcomes: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """任务说明 + 五维 rubric + 当前 battle + JSON 输出格式。"""
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

        context += "=== Current Battle to Evaluate ===\n"
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

        return context, {}
