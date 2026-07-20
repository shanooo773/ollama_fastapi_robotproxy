from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_documents(kb_dir: str) -> List[tuple[str, str]]:
    base = Path(kb_dir)
    if not base.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    documents = []
    for path in sorted(base.glob("**/*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = _read_file(path).strip()
            if text:
                documents.append((path.name, text))
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def build_chunks(kb_dir: str, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for source, text in load_documents(kb_dir):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(id=f"{source}::{i}", text=piece, source=source, chunk_index=i))
    return chunks
