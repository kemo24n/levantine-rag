from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 700  # sections are short anyway, this is just a safety net


@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    text: str
    doc_title: str = ""

    def for_search(self):
        # titles go in the searchable text too, they have the best keywords
        return f"{self.doc_title} {self.section}\n{self.text}"


def cut_long(body):
    if len(body) <= MAX_CHARS:
        return [body]
    parts, cur = [], ""
    for sent in body.split(". "):
        if cur and len(cur) + len(sent) > MAX_CHARS:
            parts.append(cur.strip())
            cur = ""
        cur += sent + ". "
    if cur.strip():
        parts.append(cur.strip())
    return parts


def split_doc(path):
    path = Path(path)
    doc = path.stem
    title, section, buf = "", "", []
    out = []

    def flush():
        body = " ".join(buf).strip()
        if not body:
            return
        for piece in cut_long(body):
            out.append(Chunk(f"{doc}#{len(out)}", doc, section, piece, title))

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("## "):
            flush()
            buf = []
            section = line[3:].strip()
        elif line.startswith("# "):
            title = line[2:].strip()
        elif line:
            buf.append(line)
    flush()
    return out


def load_chunks(folder="data/docs"):
    chunks = []
    for f in sorted(Path(folder).glob("*.md")):
        chunks.extend(split_doc(f))
    return chunks
