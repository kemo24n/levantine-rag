from rag.answer import NOT_SURE, Assistant, cited
from rag.arabic_text import clean_ar, light_stem, tokens
from rag.chunks import load_chunks
from rag.search import make


def test_clean_handles_hamza_taa_and_digits():
    assert clean_ar("الإفطار") == clean_ar("الافطار")
    assert clean_ar("مجانيّة") == "مجانيه"
    assert clean_ar("٣٠ دولاراً") == "30 دولارا"


def test_dialect_words_get_swapped():
    assert "كم" in tokens("قديش السعر")
    assert "دمشق" in tokens("بالشام")
    # "عالمالكي" should end up like "المالكي" in the docs
    assert tokens("عالمالكي") == tokens("المالكي")


def test_stemmer_doesnt_eat_short_words():
    assert light_stem("كم") == "كم"
    assert light_stem("30") == "30"


def test_chunks_keep_section_names():
    chunks = load_chunks()
    assert len(chunks) > 20
    assert all(c.section for c in chunks)


def test_hybrid_finds_delivery_fee():
    r = make("hybrid").fit(load_chunks())
    top = r.search("قديش بتاخدو توصيل لجبيل؟", 3)
    assert any(c.section == "رسوم التوصيل والحد الأدنى للطلب" for c, _ in top)


def test_citations_parse():
    assert cited("السعر 7 دولار [1] والكباب [3]") == [1, 3]


def test_assistant_with_fake_llm():
    bot = Assistant(provider="none")
    # swap in a fake llm so the test doesnt need internet
    bot.llm = lambda system, prompt: NOT_SURE if "سوشي" in prompt else "التوصيل لجبيل بثلاثة دولارات [1]"
    out = bot.ask("قديش التوصيل لجبيل؟")
    assert out["sources"] and "[1]" in out["answer"]
    assert bot.ask("عندكن سوشي؟")["answer"] == NOT_SURE
