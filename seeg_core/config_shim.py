from __future__ import annotations
from dataclasses import dataclass

# Délègue à la config centralisée basée sur dotenv + os.environ
from seeg_core.config.settings import get_app_config


@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    vision_text: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


def get_settings() -> Settings:
    cfg = get_app_config()
    return Settings(
        supabase_url=cfg.supabase_url,
        supabase_key=cfg.supabase_key,
        vision_text=cfg.vision_text,
        openai_api_key=cfg.openai_api_key,
        openai_model=cfg.openai_model,
    )
