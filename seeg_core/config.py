from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    vision_text: str = ""


def get_settings() -> Settings:
    """Load environment variables and return typed settings.
    Requires SUPABASE_URL and SUPABASE_KEY in environment or .env.
    Optional: VISION_TEXT (SEEG strategic vision text).
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    vision = os.getenv("VISION_TEXT", "").strip()
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment/.env")
    return Settings(supabase_url=url, supabase_key=key, vision_text=vision)
