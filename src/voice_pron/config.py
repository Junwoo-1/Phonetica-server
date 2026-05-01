"""환경 변수 기반 설정."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICE_", env_file=".env", extra="ignore")

    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    phoneme_model: str = "Kkonjeong/wav2vec2-base-korean"
    phoneme_device: str = "cuda"

    max_duration_sec: float = 60.0
    max_file_bytes: int = 20 * 1024 * 1024
    silence_threshold: float = 0.95
    low_language_prob: float = 0.7

    concurrency: int = 1


settings = Settings()
