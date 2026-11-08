import streamlit as st

from rag.answer import Assistant
from rag.llm import load_env

load_env()
st.set_page_config(page_title="بيت الليمون", page_icon="🍋", layout="centered")

# rtl for the whole page, streamlit is ltr by default
st.markdown(
    "<style>.stApp, .stMarkdown, .stChatMessage, textarea {direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    retriever = st.selectbox("retriever", ["hybrid", "chargrams", "bm25_arabic", "bm25_plain"])
    show_hits = st.checkbox("show retrieved chunks", value=False)


@st.cache_resource
def get_bot(name):
    return Assistant(name)


bot = get_bot(retriever)
st.title("🍋 مساعد بيت الليمون")
st.caption("اسأل بلهجتك: الأسعار، التوصيل، الحجز، النقاط...")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, text in st.session_state.chat:
    st.chat_message(role).write(text)

q = st.chat_input("اكتب سؤالك هون")
if q:
    st.chat_message("user").write(q)
    out = bot.ask(q)
    with st.chat_message("assistant"):
        st.write(out["answer"])
        if out["sources"]:
            st.caption(" · ".join(f"{s.doc_title} / {s.section}" for s in out["sources"]))
        if show_hits:
            for c, score in out["hits"]:
                st.text(f"{score:.3f}  {c.section}: {c.text[:90]}")
    st.session_state.chat += [("user", q), ("assistant", out["answer"])]
