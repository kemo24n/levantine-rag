import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .arabic_text import clean_ar, fix_dialect, tokens


class BM25:
    """plain bm25, wrote it by hand so i can control the tokenizer.
    rank_bm25 does the same thing but hides the details"""

    def __init__(self, tokenize, k1=1.5, b=0.75):
        self.tokenize = tokenize
        self.k1, self.b = k1, b

    def fit(self, chunks):
        self.chunks = chunks
        self.docs = [self.tokenize(c.for_search()) for c in chunks]
        self.lens = [len(d) for d in self.docs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        return self

    def score(self, q_words, i):
        tf, dl = self.tf[i], self.lens[i]
        s = 0.0
        for w in q_words:
            if w not in tf:
                continue
            f = tf[w]
            s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
        return s

    def search(self, q, k=5):
        q_words = self.tokenize(q)
        scores = [self.score(q_words, i) for i in range(len(self.docs))]
        best = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], scores[i]) for i in best]


def raw_split(text):
    # baseline: what you get if you just .split() arabic and pray
    return text.split()


class CharGrams:
    """tfidf on character n-grams. arabic sticks prefixes to words
    (و ب ال ...) so matching pieces of words works better than whole words"""

    def __init__(self):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)

    def prep(self, text):
        return " ".join(fix_dialect(clean_ar(text).split()))

    def fit(self, chunks):
        self.chunks = chunks
        self.m = self.vec.fit_transform([self.prep(c.for_search()) for c in chunks])
        return self

    def search(self, q, k=5):
        qv = self.vec.transform([self.prep(q)])
        sims = (self.m @ qv.T).toarray().ravel()
        best = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in best]


class Dense:
    """embeddings with sentence-transformers. optional because the model
    is ~470MB and needs internet the first time. e5 wants the
    'query: ' / 'passage: ' prefixes, without them results get worse"""

    def __init__(self, model="intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer  # lazy on purpose

        self.model = SentenceTransformer(model)

    def fit(self, chunks):
        self.chunks = chunks
        texts = ["passage: " + c.for_search() for c in chunks]
        self.m = self.model.encode(texts, normalize_embeddings=True)
        return self

    def search(self, q, k=5):
        qv = self.model.encode(["query: " + q], normalize_embeddings=True)[0]
        sims = self.m @ qv
        best = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in best]


class Hybrid:
    """reciprocal rank fusion. scores from bm25 and cosine are on totally
    different scales so i only use the ranks. 60 is the usual constant"""

    def __init__(self, parts, rrf_k=60, depth=20):
        self.parts = parts
        self.rrf_k, self.depth = rrf_k, depth

    def fit(self, chunks):
        for p in self.parts:
            p.fit(chunks)
        return self

    def search(self, q, k=5):
        fused, seen = {}, {}
        for p in self.parts:
            for rank, (c, _) in enumerate(p.search(q, self.depth)):
                fused[c.id] = fused.get(c.id, 0) + 1 / (self.rrf_k + rank + 1)
                seen[c.id] = c
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(seen[i], fused[i]) for i in best]


def make(name):
    if name == "bm25_plain":
        return BM25(raw_split)
    if name == "bm25_arabic":
        return BM25(tokens)
    if name == "bm25_no_dialect":
        return BM25(lambda t: tokens(t, dialect=False))
    if name == "chargrams":
        return CharGrams()
    if name == "dense":
        return Dense()
    if name == "hybrid":
        return Hybrid([BM25(tokens), CharGrams()])
    if name == "hybrid_dense":
        return Hybrid([BM25(tokens), CharGrams(), Dense()])
    raise ValueError(f"dont know retriever '{name}'")
