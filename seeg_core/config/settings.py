from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class AppConfig:
    supabase_url: str
    supabase_key: str
    vision_text: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "/app/chroma"


def _read_vision_text() -> str:
    # 1) Si VISION_TEXT est défini directement
    vt = os.getenv("VISION_TEXT", "").strip()
    if vt:
        return vt
    # 2) Sinon lire fichier pointé par VISION_FILE
    path = os.getenv("VISION_FILE", "").strip()
    if not path:
        # Fallback relatif au projet: scripts/vision_seeg.md
        candidate = Path(__file__).resolve().parents[2] / "scripts" / "vision_seeg.md"
        if candidate.exists():
            path = str(candidate)
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    # Charge .env si présent
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()

    # Valeurs optionnelles avec défauts sûrs
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    st_model = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip() or "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir = os.getenv("CHROMA_DIR", "/app/chroma").strip() or "/app/chroma"

    return AppConfig(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        vision_text=_read_vision_text(),
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        st_model=st_model,
        chroma_dir=chroma_dir,
    )
