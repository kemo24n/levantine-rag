"""
python ask.py "قديش التوصيل لجبيل؟"
python ask.py            (interactive, type 'exit' to quit)
"""
import sys

from rag.answer import Assistant
from rag.llm import load_env


def show(bot, q):
    out = bot.ask(q)
    print("\n" + out["answer"])
    for s in out["sources"]:
        print(f"   - {s.doc_title} / {s.section}")
    print()


if __name__ == "__main__":
    load_env()
    bot = Assistant()
    if len(sys.argv) > 1:
        show(bot, " ".join(sys.argv[1:]))
    else:
        while True:
            q = input("سؤالك: ").strip()
            if q in ("exit", "quit", "خلص"):
                break
            if q:
                show(bot, q)
