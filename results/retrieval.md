40 dev questions (msa + dialect), 20 holdout dialect questions, 35 chunks, section level match, top 5

| retriever | MSA hit@1 | dialect hit@1 | dialect hit@3 | holdout hit@1 | holdout hit@3 | holdout MRR |
|---|---|---|---|---|---|---|
| bm25_plain | 75% | 40% | 50% | 10% | 15% | 0.15 |
| bm25_no_dialect | 95% | 57% | 78% | 40% | 60% | 0.52 |
| bm25_arabic | 95% | 80% | 98% | 35% | 60% | 0.50 |
| chargrams | 95% | 85% | 98% | 55% | 70% | 0.64 |
| hybrid | 95% | 80% | 98% | 45% | 75% | 0.59 |
