import re

from .chunks import load_chunks
from .llm import get_llm
from .search import make

# the exact sentence the bot says when the docs dont have the answer.
# fixed text so the eval can check for it
NOT_SURE = "ما عندي معلومة أكيدة عن هالشي، فيك تسأل الفرع عالواتساب."

SYSTEM = f"""انت موظف خدمة زبائن في مطاعم بيت الليمون.
جاوب الزبون فقط من المعلومات الموجودة بين <sources>.
- إذا الجواب مو موجود بالمصادر قول حرفياً: {NOT_SURE}
- لا تخترع أسعار أو أوقات أو أرقام.
- جاوب بنفس لهجة الزبون، باختصار، جملة أو جملتين.
- حط رقم المصدر بين قوسين مربعين بعد المعلومة، مثل [1]."""


def build_prompt(question, hits):
    parts = []
    for i, (c, _) in enumerate(hits, 1):
        parts.append(f"[{i}] ({c.doc_title} / {c.section})\n{c.text}")
    sources = "\n\n".join(parts)
    return f"<sources>\n{sources}\n</sources>\n\nسؤال الزبون: {question}"


def cited(answer):
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


class Assistant:
    def __init__(self, retriever="hybrid", provider=None, top_k=4, docs="data/docs"):
        self.search = make(retriever).fit(load_chunks(docs))
        self.llm = get_llm(provider)
        self.top_k = top_k

    def ask(self, question):
        hits = self.search.search(question, self.top_k)
        if self.llm is None:
            # no llm mode: just hand back the best passage
            best = hits[0][0]
            text = f"{best.text} [1]"
        else:
            text = self.llm(SYSTEM, build_prompt(question, hits))
        sources = [hits[n - 1][0] for n in cited(text) if 0 < n <= len(hits)]
        return {"answer": text, "sources": sources, "hits": hits}
