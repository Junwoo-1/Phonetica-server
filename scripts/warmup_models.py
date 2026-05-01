"""HF 모델 사전 다운로드 + 더미 추론 워밍업.

서버 첫 요청 지연을 줄이기 위해 미리 실행한다.

사용: uv run python scripts/warmup_models.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from voice_pron.asr.whisper_asr import WhisperASR
from voice_pron.audio.loader import AudioBuffer
from voice_pron.config import settings
from voice_pron.phoneme.wav2vec_jamo import JamoPhonemeRecognizer


def make_dummy_audio(seconds: float = 1.0, sr: int = 16000) -> AudioBuffer:
    samples = (0.01 * np.random.randn(int(seconds * sr))).astype(np.float32)
    return AudioBuffer(samples=samples, sr=sr, duration=seconds, silence_ratio=0.0)


def main() -> None:
    print(f"[warmup] loading whisper: {settings.whisper_model}")
    t0 = time.perf_counter()
    asr = WhisperASR(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    print(f"[warmup] whisper loaded in {time.perf_counter() - t0:.1f}s")

    print(f"[warmup] loading phoneme: {settings.phoneme_model}")
    t0 = time.perf_counter()
    phoneme = JamoPhonemeRecognizer(
        model_id=settings.phoneme_model,
        device=settings.phoneme_device,
        torch_dtype=torch.float16,
    )
    print(f"[warmup] phoneme loaded in {time.perf_counter() - t0:.1f}s")

    dummy = make_dummy_audio()
    print("[warmup] running dummy inference...")
    import asyncio

    async def _run():
        # ASR 워밍업
        async for _ in asr.transcribe_stream(dummy):
            pass
        # Phoneme 워밍업
        result = await phoneme.recognize(dummy)
        print(f"[warmup] phoneme produced {len(result.raw_sequence)} tokens")

    asyncio.run(_run())
    print("[warmup] done")


if __name__ == "__main__":
    main()
