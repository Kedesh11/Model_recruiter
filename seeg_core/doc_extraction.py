from __future__ import annotations
import re
from io import BytesIO
from typing import List

import requests

try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:  # pragma: no cover
    pdf_extract_text = None  # type: ignore
try:
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None  # type: ignore

from .ocr import ocr_image_bytes, ocr_pdf_with_fitz


def download_bytes(url: str) -> bytes | None:
    try:
        if not isinstance(url, str) or not url:
            return None
        if not url.startswith(("http://", "https://")):
            return None
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def extract_text_from_pdf(data: bytes) -> str:
    if not data:
        return ""
    if pdf_extract_text is None:
        return ocr_pdf_with_fitz(data)
    try:
        txt = pdf_extract_text(BytesIO(data)) or ""
    except Exception:
        txt = ""
    if not txt or len(txt.strip()) < 50:
        ocr_txt = ocr_pdf_with_fitz(data)
        if ocr_txt:
            return ocr_txt
    return txt


def extract_text_from_docx(data: bytes) -> str:
    if not data or docx is None:
        return ""
    try:
        d = docx.Document(BytesIO(data))
        paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    except Exception:
        return ""


def extract_text_from_txt(data: bytes) -> str:
    if not data:
        return ""
    for enc in ("utf-8", "latin1"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""


def summarize_text_simple(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_chars:
        return t
    head = t[: max_chars // 2]
    tail = t[- max_chars // 3 :]
    return head + " … " + tail


def chunk_text(t: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    if not t:
        return []
    s = re.sub(r"\s+", " ", str(t)).strip()
    if len(s) <= max_chars:
        return [s]
    chunks: List[str] = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        if end >= len(s):
            break
        start = max(0, end - overlap)
    return chunks


def summarize_hierarchical(text: str, chunk_size: int = 1500, overlap: int = 200, max_levels: int = 3) -> str:
    if not text:
        return ""
    chunks = chunk_text(text, max_chars=chunk_size, overlap=overlap)
    summaries = [summarize_text_simple(c, max_chars=chunk_size // 2) for c in chunks]
    merged = " \n".join(summaries)
    level = 1
    while len(merged) > chunk_size and level < max_levels:
        level += 1
        chunks = chunk_text(merged, max_chars=chunk_size, overlap=overlap)
        summaries = [summarize_text_simple(c, max_chars=chunk_size // 2) for c in chunks]
        merged = " \n".join(summaries)
    return summarize_text_simple(merged, max_chars=chunk_size)
