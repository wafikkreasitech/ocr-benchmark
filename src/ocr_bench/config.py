"""Configuration loaded from .env (pydantic-settings).

All toggles default to safe/off so a fresh clone behaves like the plain
benchmark. See .env.example for the full set.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Post-processing: SymSpell + KBBI spell correction
    enable_symspell_correction: bool = False
    enable_word_segmentation: bool = False       # KBBI-based space re-insertion
    symspell_max_edit_distance: int = 1          # conservative: 1
    kbbi_top_n: int = 0                          # 0 = load all 194k; >0 = cap (for edge devices)
    kbbi_csv_path: str = "kbbi/kbbi_v6.1.0_full.csv"

    # Runner
    iou_threshold: float = 0.5                    # detection match threshold

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()