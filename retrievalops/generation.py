"""Answer generation through any OpenAI-compatible endpoint (local vLLM by default)."""

from __future__ import annotations

import os

from retrievalops.config import GenerationConfig

PROMPT = """You are a contract analysis assistant.
Answer the question using ONLY the numbered context passages. If they do not contain the answer,
say "The provided context does not answer this question." Cite passages like [1], [2].

Context:
{context}

Question: {question}

Answer:"""


class Generator:
    def __init__(self, cfg: GenerationConfig):
        from openai import OpenAI

        api_key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else "EMPTY"
        if cfg.api_key_env and not api_key:
            raise RuntimeError(f"Environment variable {cfg.api_key_env} is not set")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key)

    def answer(self, question: str, passages: list[str]) -> str:
        context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages[: self.cfg.context_k], start=1))
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        return resp.choices[0].message.content or ""
