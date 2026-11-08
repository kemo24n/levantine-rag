"""
end to end check with a real llm. needs LLM_PROVIDER + key in .env

python -m eval.answer_eval
python -m eval.answer_eval --provider groq --limit 10

what i check (simple on purpose, no llm-as-judge yet):
  correct  -> the key fact (must_have) is in the answer
  cited    -> answer cites the source that really has the answer
  refused  -> for out of scope questions it says NOT_SURE instead of inventing
"""
import argparse
import json
import time
from pathlib import Path

from rag.answer import NOT_SURE, Assistant
from rag.arabic_text import clean_ar
from rag.llm import load_env

from .run_eval import load_questions


def has(answer, fact):
    return clean_ar(fact) in clean_ar(answer)


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.0, help="free tiers rate limit you, go slow")
    args = ap.parse_args()

    bot = Assistant(args.retriever, args.provider)
    if bot.llm is None:
        raise SystemExit("set LLM_PROVIDER (gemini / groq / ollama) first, see .env.example")

    qs = load_questions()
    if args.limit:
        qs = qs[: args.limit]

    rows = []
    for q in qs:
        out = bot.ask(q["dialect"])
        a = out["answer"]
        if q["doc"]:
            row = {
                "id": q["id"],
                "correct": has(a, q["must_have"]),
                "cited": any(s.doc == q["doc"] and s.section == q["section"] for s in out["sources"]),
            }
        else:
            row = {"id": q["id"], "refused": has(a, NOT_SURE[:20])}
        row["q"], row["answer"] = q["dialect"], a
        rows.append(row)
        print(row["id"], {k: v for k, v in row.items() if isinstance(v, bool)}, a[:80])
        time.sleep(args.sleep)

    inscope = [r for r in rows if "correct" in r]
    outscope = [r for r in rows if "refused" in r]
    summary = {
        "provider": args.provider or "env",
        "correct": sum(r["correct"] for r in inscope) / max(len(inscope), 1),
        "cited_right_source": sum(r["cited"] for r in inscope) / max(len(inscope), 1),
        "refused_out_of_scope": sum(r["refused"] for r in outscope) / max(len(outscope), 1),
    }
    print(json.dumps(summary, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/answers.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
