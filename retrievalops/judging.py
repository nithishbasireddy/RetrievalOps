"""LLM relevance judge over an OpenAI-compatible endpoint (local vLLM), with a persistent cache.

Each (query, passage) pair is graded 0/1/2. The grade's probability distribution is read from the
first token's log-probabilities, so every judgment carries a soft expected relevance in [0, 1]:
    rel = (P(1) * 0.5 + P(2) * 1.0) / (P(0) + P(1) + P(2))
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)

SYSTEM = "You are an expert legal analyst who annotates search results for relevance."
TEMPLATE = """Query: {query}

Passage (from the document "{doc}"):
\"\"\"
{passage}
\"\"\"

How relevant is the passage to the query?
2 = the passage contains text that answers the query (fully or in part)
1 = the passage is on the same topic or the same contract but does not contain the answer
0 = the passage is irrelevant, or comes from a different document than the one the query is about

Reply with a single digit: 0, 1 or 2."""

# v2: also shows the opening of the source document (where contracts name their parties), so the
# judge can tell whether a passage comes from the document the query is about. v1 only showed a
# file name, and on ContractNLI 97% of v1's strict positives came from the wrong NDA.
TEMPLATE_V2 = """Query: {query}

The passage below comes from a document that begins like this:
\"\"\"
{header}
\"\"\"

Passage:
\"\"\"
{passage}
\"\"\"

How relevant is the passage to the query?
2 = the passage contains text that answers the query (fully or in part)
1 = the passage is on the same topic or the same document but does not contain the answer
0 = the passage is irrelevant, or comes from a different document than the one the query is about
    (check the parties / names in the document opening against the query)

Reply with a single digit: 0, 1 or 2."""
TEMPLATES = {"v1": TEMPLATE, "v2": TEMPLATE_V2}


@dataclass(frozen=True)
class Judgment:
    p0: float
    p1: float
    p2: float

    @property
    def relevance(self) -> float:
        z = self.p0 + self.p1 + self.p2
        return (0.5 * self.p1 + self.p2) / z if z > 0 else 0.0

    @property
    def p_answer(self) -> float:
        """Strict relevance: probability that the passage contains the answer (grade 2)."""
        z = self.p0 + self.p1 + self.p2
        return self.p2 / z if z > 0 else 0.0

    def score(self, mode: str) -> float:
        if mode == "graded":
            return self.relevance
        if mode == "strict":
            return self.p_answer
        raise ValueError(f"unknown relevance mode {mode!r}")

    @property
    def label(self) -> int:
        return max(range(3), key=lambda i: (self.p0, self.p1, self.p2)[i])


@dataclass(frozen=True)
class JudgeRequest:
    query: str
    doc: str
    passage: str
    header: str = ""  # opening of the source document (used by prompt v2)

    def key(self, model: str, prompt: str = "v1") -> str:
        h = hashlib.sha256()
        parts = [model, prompt, self.query, self.doc, self.passage]
        if prompt != "v1":  # v1 keys stay identical to earlier runs
            parts.append(self.header)
        for part in parts:
            h.update(part.encode("utf-8", errors="replace"))
            h.update(b"\x00")
        return h.hexdigest()


class JudgmentCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # long lock timeout + WAL so several processes can share the cache safely
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=600)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS j (key TEXT PRIMARY KEY, p0 REAL, p1 REAL, p2 REAL)")
        self.lock = threading.Lock()

    def get_many(self, keys: Sequence[str]) -> dict[str, Judgment]:
        out: dict[str, Judgment] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = f"SELECT key, p0, p1, p2 FROM j WHERE key IN ({','.join('?' * len(chunk))})"
            for k, p0, p1, p2 in self.conn.execute(q, chunk):
                out[k] = Judgment(p0, p1, p2)
        return out

    def put(self, key: str, j: Judgment) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO j VALUES (?, ?, ?, ?)", (key, j.p0, j.p1, j.p2))

    def commit(self) -> None:
        with self.lock:
            self.conn.commit()


def parse_digit_logprobs(top_logprobs: Iterable) -> Judgment:
    """Probability mass on '0', '1', '2' among the first token's top alternatives."""
    p = [0.0, 0.0, 0.0]
    for alt in top_logprobs:
        tok = alt.token.strip()
        if tok in ("0", "1", "2"):
            p[int(tok)] += math.exp(alt.logprob)
    return Judgment(*p)


class LLMJudge:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8010/v1",
        cache_path: str | Path = "data/cache/judgments.sqlite",
        max_passage_chars: int = 4000,
        concurrency: int = 64,
        disable_thinking: bool = True,
        prompt: str = "v1",
    ):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=300, max_retries=5)
        self.cache = JudgmentCache(cache_path)
        self.max_passage_chars = max_passage_chars
        self.concurrency = concurrency
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
        if prompt not in TEMPLATES:
            raise ValueError(f"unknown judge prompt {prompt!r}; choose from {sorted(TEMPLATES)}")
        self.prompt = prompt

    def _call(self, req: JudgeRequest) -> Judgment:
        content = TEMPLATES[self.prompt].format(
            query=req.query, doc=req.doc, header=req.header, passage=req.passage[: self.max_passage_chars])
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
            extra_body=self.extra_body,
        )
        lp = resp.choices[0].logprobs
        if lp is None or not lp.content:
            raise RuntimeError("endpoint returned no logprobs")
        j = parse_digit_logprobs(lp.content[0].top_logprobs)
        if j.p0 + j.p1 + j.p2 < 1e-6:
            log.warning("No digit among top tokens for a judgment (got %r); treated as irrelevant",
                        resp.choices[0].message.content)
            j = Judgment(1.0, 0.0, 0.0)
        return j

    def judge(self, requests: Sequence[JudgeRequest], desc: str = "judging") -> list[Judgment]:
        keys = [r.key(self.model, self.prompt) for r in requests]
        have = self.cache.get_many(list(dict.fromkeys(keys)))
        todo = {k: r for k, r in zip(keys, requests) if k not in have}
        if todo:
            log.info("%s: %d cached, %d new judgments with %s", desc, len(have), len(todo), self.model)
            with ThreadPoolExecutor(self.concurrency) as pool:
                futures = {pool.submit(self._call, r): k for k, r in todo.items()}
                for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc=desc)):
                    j = fut.result()
                    have[futures[fut]] = j
                    self.cache.put(futures[fut], j)
                    if i % 500 == 0:
                        self.cache.commit()
            self.cache.commit()
        return [have[k] for k in keys]
