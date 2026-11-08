"""
retrieval eval. for every question i know which section has the answer,
so i check where that section shows up in the results.

python -m eval.run_eval
python -m eval.run_eval --with-dense      (needs sentence-transformers + internet once)
"""
import argparse
import json
from pathlib import Path

from rag.chunks import load_chunks
from rag.search import make

K = 5


def load_questions(path="data/eval/questions.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_of_answer(results, q):
    for i, (c, _) in enumerate(results):
        if c.doc == q["doc"] and c.section == q["section"]:
            return i + 1
    return None


def check(retriever, questions, field):
    hit1 = hit3 = rr = 0.0
    misses = []
    for q in questions:
        r = rank_of_answer(retriever.search(q[field], K), q)
        if r == 1:
            hit1 += 1
        if r and r <= 3:
            hit3 += 1
        if r:
            rr += 1 / r
        else:
            misses.append(q["id"])
    n = len(questions)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "mrr@5": rr / n, "missed": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-dense", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    # out of scope questions dont have a right section, they're for the answer eval
    questions = [q for q in load_questions() if q["doc"]]

    names = ["bm25_plain", "bm25_no_dialect", "bm25_arabic", "chargrams", "hybrid"]
    if args.with_dense:
        names += ["dense", "hybrid_dense"]

    # holdout = dialect questions i wrote AFTER building the dialect dict.
    # i dont touch the dict based on these, otherwise the number is fake
    holdout = load_questions("data/eval/holdout.jsonl")

    results = {}
    for name in names:
        r = make(name).fit(chunks)
        results[name] = {f: check(r, questions, f) for f in ("msa", "dialect")}
        results[name]["holdout"] = check(r, holdout, "dialect")

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"{len(questions)} dev questions (msa + dialect), {len(holdout)} holdout dialect questions, "
        f"{len(chunks)} chunks, section level match, top {K}",
        "",
        "| retriever | MSA hit@1 | dialect hit@1 | dialect hit@3 | holdout hit@1 | holdout hit@3 | holdout MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        m, d, h = r["msa"], r["dialect"], r["holdout"]
        lines.append(
            f"| {name} | {m['hit@1']:.0%} | {d['hit@1']:.0%} | {d['hit@3']:.0%} "
            f"| {h['hit@1']:.0%} | {h['hit@3']:.0%} | {h['mrr@5']:.2f} |"
        )
    table = "\n".join(lines)
    (out / "retrieval.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
