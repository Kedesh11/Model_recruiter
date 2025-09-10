from __future__ import annotations
from io import BytesIO

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore


def ocr_image_bytes(data: bytes, lang: str = "eng+fra") -> str:
    if not data or Image is None or pytesseract is None:
        return ""
    try:
        img = Image.open(BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img, lang=lang) or ""
    except Exception:
        return ""


def ocr_pdf_with_fitz(data: bytes, dpi: int = 200, max_pages: int = 20, lang: str = "eng+fra") -> str:
    if not data or fitz is None or pytesseract is None or Image is None:
        return ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return ""
    texts: list[str] = []
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            if i > max_pages:
                break
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                t = ocr_image_bytes(img_bytes, lang=lang)
                if t and t.strip():
                    texts.append(t)
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return "\n\n".join(texts)
