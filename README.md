# levantine-rag

Customer support assistant for a restaurant chain that answers questions written in **Syrian / Lebanese dialect**, from documents written in **Modern Standard Arabic**, and cites where every answer came from.

Customers write `قديش بتاخدو توصيل لجبيل؟`. The policy doc says `رسوم التوصيل إلى جبيل بثلاثة دولارات`. Almost no words in common, so normal search breaks. Most Arabic RAG demos only test with fusha questions and never hit this problem. This project measures it and tries a few fixes.

The business ("بيت الليمون") is made up. The docs are sample data I wrote to look like what real restaurants in the region send me.

## Results

Retrieval, section level (did the right section come back?), from `python -m eval.run_eval`:

| retriever | MSA hit@1 | dialect hit@1 | dialect hit@3 | holdout hit@1 | holdout hit@3 | holdout MRR |
|---|---|---|---|---|---|---|
| bm25_plain | 75% | 40% | 50% | 10% | 15% | 0.15 |
| bm25_no_dialect | 95% | 57% | 78% | 40% | 60% | 0.52 |
| bm25_arabic | 95% | 80% | 98% | 35% | 60% | 0.50 |
| chargrams | 95% | 85% | 98% | 55% | 70% | 0.64 |
| hybrid | 95% | 80% | 98% | 45% | 75% | 0.59 |

40 dev questions (each one in fusha and in dialect) + 20 holdout dialect questions written after the dialect dictionary was done and never used to tune it.

What I took from it:

- Splitting Arabic on spaces is useless for dialect (10% on holdout).
- Normalization + a light stemmer fixes most fusha questions on its own (75% to 95%).
- The dialect dictionary looks great on dev (98% hit@3) and drops to 60% on holdout. It overfits to the questions I wrote it from. Without the holdout set I would have reported 98%.
- Character n-grams generalize best because they don't depend on a word list.
- Fusing BM25 + n-grams (RRF) gives the best holdout hit@3 but hurts hit@1.

Failure analysis and the reasoning behind each choice are in [notes/design_notes.md](notes/design_notes.md).

## How it works

```
question (dialect)
   -> clean_ar: hamza / taa marbuta / tashkeel / arabic digits
   -> dialect map + light stemmer (for bm25)
   -> retrievers: bm25 (hand written) | char 2-5 grams tfidf | e5 embeddings (optional)
   -> reciprocal rank fusion
   -> top 4 sections, numbered
   -> llm (gemini / groq / ollama) answers only from them, cites [n],
      or says a fixed "not sure" sentence if the answer isn't there
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # works with LLM_PROVIDER=none, no key needed

python -m eval.run_eval     # retrieval numbers
python -m pytest -q         # tests, run offline
python ask.py "فيني ادفع عالويش؟"
streamlit run app.py        # chat ui, rtl
```

With an LLM (all free): put a Gemini or Groq key in `.env`, or install [Ollama](https://ollama.com) and `ollama pull qwen2.5:7b`. Then:

```bash
python -m eval.answer_eval --limit 10
```

This checks if the key fact is in the answer, if the right source is cited, and if out of scope questions (`عندكن سوشي؟`) get the "not sure" reply instead of an invented answer.

Embeddings (optional, ~470MB download):

```bash
pip install sentence-transformers
python -m eval.run_eval --with-dense
```

## Layout

```
rag/arabic_text.py   cleaning, dialect map, stemmer
rag/chunks.py        split docs by section
rag/search.py        bm25, char ngrams, dense, hybrid
rag/llm.py           gemini / groq / ollama clients
rag/answer.py        prompt, citations, Assistant
eval/run_eval.py     retrieval metrics
eval/answer_eval.py  end to end answer checks
data/docs/           sample restaurant docs (arabic)
data/eval/           dev + holdout questions
notes/               design notes
```

## Next

- LLM query rewriting (dialect to fusha) before retrieval, measured on holdout
- Dense numbers on the same table
- Bigger holdout set, 20 questions means one question = 5%

Built by Kamel Nassif.
