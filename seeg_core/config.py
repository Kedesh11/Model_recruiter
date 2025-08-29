from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    vision_text: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


def get_settings() -> Settings:
    """Load environment variables and return typed settings.
    Requires SUPABASE_URL and SUPABASE_KEY in environment or .env.
    Optional: VISION_TEXT (SEEG strategic vision text), VISION_FILE (path to a file containing the vision text).
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    vision = os.getenv("VISION_TEXT", "").strip()
    vision_file = os.getenv("VISION_FILE", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-5"
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment/.env")
    # If no inline vision text provided, try to read from file (VISION_FILE),
    # otherwise fallback to default scripts/vision_seeg.md if present.
    if not vision:
        path: Optional[str] = None
        if vision_file:
            path = vision_file
        else:
            # default relative to project root
            try:
                here = os.path.dirname(os.path.dirname(__file__))  # seeg_core/ -> project root
                candidate = os.path.join(here, "scripts", "vision_seeg.md")
                if os.path.exists(candidate):
                    path = candidate
            except Exception:
                path = None
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    vision = f.read().strip()
            except Exception:
                vision = ""

    return Settings(
        supabase_url=url,
        supabase_key=key,
        vision_text=vision,
        openai_api_key=openai_key,
        openai_model=openai_model,
    )
