"""Generate synthetic (question, gold passage) pairs from a corpus with an LLM.

    python scripts/make_synthetic.py configs/lbr_full_grid_v1.yaml --style naive \
        --per-subset 100 --model Qwen/Qwen3.5-27B --url http://127.0.0.1:8011/v1 \
        --out data/synthetic/lbr_synth_naive.json

Passages are sampled uniformly over the documents that real queries of each subset refer to,
cut at sentence boundaries (300-700 chars). Styles:
  naive          generic "write a question answered by this passage" (RAGAS/ARES-like)
  style_matched  additionally shows 5 real (unlabelled) user queries of the same corpus
The passage becomes the question's gold span, so the output loads through `dataset.query_file`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from tqdm import tqdm

from retrievalops.config import load_experiment
from retrievalops.datasets import load_dataset

NAIVE = """Below is a passage from the document "{doc}".

\"\"\"
{passage}
\"\"\"

Write one question that a user could ask whose answer is contained in this passage.
Output only the question, nothing else."""

STYLE = """Users of a legal search system ask questions like these:
{examples}

Below is a passage from the document "{doc}".

\"\"\"
{passage}
\"\"\"

Write one new question, in the same style as the examples above, whose answer is contained in this
passage. Output only the question, nothing else."""

_SENT_END = re.compile(r"[.;:!?]\s|\n")


def sample_passage(text: str, rng: random.Random, lo: int = 300, hi: int = 700) -> tuple[int, int] | None:
    for _ in range(20):
        if len(text) < lo:
            return None
        start = rng.randrange(0, max(1, len(text) - lo))
        m = _SENT_END.search(text, start)
        if m is None:
            continue
        start = m.end()
        while start < len(text) and text[start].isspace():
            start += 1
        target = start + rng.randint(lo, hi)
        ends = [m.end() for m in _SENT_END.finditer(text, start + lo, min(len(text), target + 200))]
        if not ends:
            continue
        end = min(ends, key=lambda e: abs(e - target))
        while end > start and text[end - 1].isspace():
            end -= 1
        span = text[start:end]
        # skip tables of numbers / signature blocks: require mostly letters and a minimum length
        if len(span) >= 0.8 * lo and sum(ch.isalpha() for ch in span) >= 0.6 * len(span):
            return start, end
    return None


def clean_question(q: str) -> str:
    lines = [ln.strip() for ln in q.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    q = re.sub(r"^(question|q)\s*[:.]\s*", "", lines[0], flags=re.IGNORECASE)
    return q.strip().strip("\"'").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--style", choices=["naive", "style_matched"], required=True)
    ap.add_argument("--per-subset", type=int, default=100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    exp = load_experiment(args.experiment)
    ds = load_dataset(exp.dataset)
    docs = {d.doc_id: d.text for d in ds.documents}
    rng = random.Random(args.seed)
    client = OpenAI(base_url=args.url, api_key="EMPTY", timeout=300, max_retries=5)

    jobs = []
    for subset in sorted({q.subset for q in ds.queries}):
        real = [q for q in ds.queries if q.subset == subset]
        subset_docs = sorted({s.doc_id for q in real for s in q.gold_spans})
        made = 0
        while made < args.per_subset:
            doc = rng.choice(subset_docs)
            span = sample_passage(docs[doc], rng)
            if span is None:
                continue
            examples = [q.text for q in rng.sample(real, 5)]
            jobs.append({"qid": f"{subset}/synth{made}", "subset": subset, "doc": doc, "span": span,
                         "examples": examples})
            made += 1

    def run(job: dict) -> dict:
        passage = docs[job["doc"]][job["span"][0] : job["span"][1]]
        tmpl = NAIVE if args.style == "naive" else STYLE
        prompt = tmpl.format(doc=Path(job["doc"]).stem, passage=passage,
                             examples="\n".join(f"- {e}" for e in job["examples"]))
        resp = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}], max_tokens=160,
            temperature=0.7, seed=zlib.crc32(f"{args.seed}/{job['qid']}".encode()),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return {"qid": job["qid"], "subset": job["subset"], "style": args.style,
                "query": clean_question(resp.choices[0].message.content or ""),
                "snippets": [{"file_path": job["doc"], "span": list(job["span"])}]}

    with ThreadPoolExecutor(32) as pool:
        tests = list(tqdm(pool.map(run, jobs), total=len(jobs), desc=f"synthetic {args.style}"))
    bad = [t for t in tests if len(t["query"]) < 10]
    tests = [t for t in tests if len(t["query"]) >= 10]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"tests": tests}, indent=1))
    print(f"wrote {len(tests)} questions to {args.out} ({len(bad)} empty/short dropped)")
    for t in tests[:: max(1, len(tests) // 6)]:
        print(f"  [{t['subset']}] {t['query']}")


if __name__ == "__main__":
    main()
