"""
Minimal example agent for LitReviewBench evaluation.

Implement an agent with generate(query: str) -> str and run:
  python -m evaluator.main --battles-file data/battles.jsonl --expert-outcomes-file data/expert_outcomes.jsonl --limit 5
"""

import os
from openai import OpenAI


class SimpleAgent:
    """
    Minimal agent that calls OpenAI/OpenRouter to generate literature review drafts.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENROUTER_API_KEY"):
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.model = os.getenv("LITREVIEW_MODEL", "openai/gpt-4o-mini")
        else:
            base_url = None
            self.model = os.getenv("LITREVIEW_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url) if self.api_key else None

    def generate(self, query: str) -> str:
        """
        Generate a literature review draft for the given topic query.

        Args:
            query: Topic query (e.g. "Conduct a literature review on X.")

        Returns:
            Draft text as a string.
        """
        if not self.client:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment"
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert academic writer. Produce a concise literature review draft based on the user's topic. Use clear sections, cite relevant work where appropriate, and identify gaps or future directions.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
