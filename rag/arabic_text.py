import re

# tashkeel + the small quranic marks, we never need them for search
HARAKAT = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"

# arabic-indic digits -> normal digits, people type both
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# levantine words that kill BM25 because the docs are written in fusha.
# kept it small on purpose, only words i actually saw in the test questions
# or that customers send all the time on whatsapp
DIALECT = {
    "شو": "ما",
    "ايش": "ما",
    "ليش": "لماذا",
    "وين": "اين",
    "قديش": "كم",
    "اديش": "كم",
    "قدي": "كم",
    "ايمتى": "متى",
    "امتى": "متى",
    "لايمتى": "متى",
    "هلق": "الان",
    "هلا": "الان",
    "بدي": "اريد",
    "بدنا": "نريد",
    "فيني": "يمكن",
    "فينا": "يمكن",
    "بقدر": "يمكن",
    "مافي": "لا يوجد",
    "كتير": "كثير",
    "شوي": "قليل",
    "ببلاش": "مجاني",
    "بحقها": "سعر",
    "مصاري": "مبلغ",
    "المصاري": "المبلغ",
    "الولاد": "الاطفال",
    "للولاد": "للاطفال",
    "عيلية": "عائليه",
    "العيلية": "العائليه",
    "باركينغ": "موقف سيارات",
    "باركنج": "موقف سيارات",
    "دليفري": "توصيل",
    "الدليفري": "التوصيل",
    "الشوفير": "السائق",
    "للشوفير": "للسائق",
    "فيزا": "بطاقات مصرفيه",
    "الويش": "whish",
    "عالويش": "whish",
    "ويش": "whish",
    "الشام": "دمشق",
    "بالشام": "دمشق",
    "بيسكر": "يغلق",
    "مسكر": "مغلق",
    "لغيت": "الغاء",
    "اتأخرت": "تاخر",
    "اتاخرت": "تاخر",
    "اتأخر": "تاخر",
    "اتاخر": "تاخر",
    "اشتكي": "شكوى",
    "بشتكي": "شكوى",
    "الكيك": "قالب الحلوى",
    "برا": "خارج",
    "عالراس": "الشخص",
    "بتزينولنا": "ديكور",
    "بتزينو": "ديكور",
    "منيح": "جيد",
}

# words that carry no meaning for retrieval, fusha + dialect.
# NOTE: dont put "لا" here, "لا يوجد" matters for some answers
STOPWORDS = set("""
في من على الى إلى عن مع هل ما ماذا كيف هذا هذه ذلك التي الذي او أو ثم قد كل
انا أنا نحن نحنا انت انتو هو هي هم عندكن عندكم عنا عندي معي الي إلي لي
يا بس كمان شي شيء اذا إذا لما يلي اللي تبع منكن منكم يعني
عم رح ح لازم بدو
""".split())

# prefixes / suffixes for a very light stemmer. not a real stemmer,
# just enough so "بالتوصيل" and "التوصيل" and "توصيل" end up the same
PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف", "ك")
SUFFIXES = ("ات", "ون", "ين", "ها", "هم", "كم", "كن", "نا", "ه", "ي", "ة")


def clean_ar(text):
    """normalize arabic so the same word always looks the same"""
    text = text.translate(DIGITS)
    text = HARAKAT.sub("", text)
    text = text.replace(TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("ة", "ه")
    # keep % and . inside numbers, kill other punctuation
    text = re.sub(r"[^\w\s%]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def fix_dialect(words):
    out = []
    for w in words:
        # "عالمالكي" = "على المالكي", very common in levantine typing
        if w.startswith("عال") and len(w) > 4 and w not in DIALECT:
            w = w[1:]
        swap = DIALECT.get(w)
        if swap is None:
            out.append(w)
        else:
            out.extend(swap.split())
    return out


def light_stem(w):
    if w.isdigit() or len(w) <= 3:
        return w
    for p in PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[: -len(s)]
            break
    return w


def tokens(text, dialect=True, stem=True):
    # dialect map runs on cleaned text, so its keys are written cleaned too
    words = clean_ar(text).split()
    if dialect:
        words = fix_dialect([clean_ar(w) for w in words])
    words = [w for w in words if w not in STOPWORDS]
    if stem:
        words = [light_stem(w) for w in words]
    return [w for w in words if w]


# the dialect dict keys have hamzas etc, clean them once so lookups match
DIALECT = {clean_ar(k): clean_ar(v) for k, v in DIALECT.items()}
STOPWORDS = {clean_ar(w) for w in STOPWORDS}
