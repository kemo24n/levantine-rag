# notes to myself (why i did things this way)

## the problem
customers in syria and lebanon write in dialect on whatsapp: "قديش التوصيل لجبيل؟".
restaurant docs are written in fusha: "رسوم التوصيل إلى جبيل".
almost no shared words. normal keyword search fails, and most arabic rag demos
online only test fusha questions so they never see this.

## why sections as chunks
the docs are already split by `##` headings and each section answers one kind
of question. fixed size chunks (500 tokens with overlap) would mix delivery fees
with delivery times. sections are small here so i only cut if > 700 chars.
section title + doc title go into the searchable text because they have the
best keywords ("رسوم", "الحد الأدنى").

## retrievers i tried
1. bm25_plain: just .split(). baseline, to see how bad it is.
2. bm25_no_dialect: arabic cleaning (hamza, taa marbuta, tashkeel, digits) + light stemmer.
3. bm25_arabic: same + dialect dictionary (قديش -> كم, دليفري -> توصيل ...).
4. chargrams: tfidf on 2-5 char ngrams. arabic glues و ب ال ل to words so
   partial matching catches "بتوصلو" vs "التوصيل" without a stemmer.
5. hybrid: rrf of bm25_arabic + chargrams. rrf uses ranks only, because bm25
   scores and cosine scores are not on the same scale.
6. dense (optional): multilingual-e5-small. remember the "query: " / "passage: "
   prefixes, e5 is trained with them.

## what the numbers said (results/retrieval.md)
- plain split is terrible on dialect: 40% hit@1 on dev, 10% on holdout.
- cleaning + stemming alone almost fixes fusha (75% -> 95%).
- the dialect dictionary looked amazing on dev (98% hit@3)... but i built it
  while reading the dev questions. so i wrote 20 NEW dialect questions after
  (holdout) and didnt touch the dict. holdout hit@3 = 60%. the dict overfits.
  this was the most useful thing i learned in the project.
- char ngrams generalize best on holdout (70% hit@3, 0.64 mrr) because they
  dont depend on a word list.
- hybrid gets best holdout hit@3 (75%) but worse hit@1 than chargrams alone.
  bm25 pulls in wrong sections when the dialect word isnt in the dict.

## holdout questions that still fail and why
- h2 "لأيمتى بتسهرو بالليل؟" -> "بتسهرو" (you stay open late) has no fusha match at all.
- h5 "الأكل بيتأخر كتير لحتى يوصل؟" -> gets delivery zones, opening hours and the
  late delivery complaint section, but not "مدة التوصيل". "يوصل" matches every
  delivery section, nothing says "duration".
- h12 "بتاخدو دولار ولا سوري؟" -> goes to the menu because every price has "دولار".
  "سوري" means syrian pounds, docs say "العملة المحلية". no keyword method will
  get this. needs embeddings or an llm rewrite.
- h16 "شو بيصير إذا الأكل وصل بعد ساعتين؟" -> hybrid misses it (chargrams alone
  gets it). "ساعتين" vs "90 دقيقة", no overlap, bm25 drags wrong sections up.

## answer side
- the llm only sees the top 4 chunks, numbered [1]..[4], and must cite them.
  i parse the [n] to show sources under the answer.
- fixed "not sure" sentence (NOT_SURE). if the docs dont have it the bot must
  say exactly that. makes refusals easy to test, and a restaurant bot inventing
  a price is worse than no bot.
- temperature 0.2, its customer support not poetry.
- provider is swappable (gemini / groq / ollama) so it runs for free.

## next things to try
- [ ] llm query rewrite: turn the dialect question into a fusha search query
      before retrieval. should fix h2 and h12. measure on holdout.
- [ ] run dense + hybrid_dense on my laptop and add the numbers.
- [ ] llm as judge for answer quality instead of only keyword match.
- [ ] more holdout questions, 20 is small. one question = 5%.
