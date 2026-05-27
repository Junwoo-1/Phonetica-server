"""모델 파일만 캐시에 내려받는 스크립트 (추론 없음).

사용: uv run python scripts/download_models.py
"""

from __future__ import annotations

import time

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from voice_pron.config import settings


def download_phoneme(model_id: str) -> None:
    print(f"[download] phoneme → {model_id}")
    t0 = time.perf_counter()
    Wav2Vec2Processor.from_pretrained(model_id)
    Wav2Vec2ForCTC.from_pretrained(model_id)
    print(f"[download] phoneme done ({time.perf_counter() - t0:.1f}s)")


def main() -> None:
    download_phoneme(settings.phoneme_model)
    print("[download] all models cached")


if __name__ == "__main__":
    main()
