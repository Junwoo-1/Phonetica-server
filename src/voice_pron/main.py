"""FastAPI 엔트리포인트 + lifespan (모델 1회 로드)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from voice_pron.api.routes import router
from voice_pron.asr.whisper_asr import WhisperASR
from voice_pron.config import settings
from voice_pron.g2p.pronouncer import StandardPronouncer
from voice_pron.phoneme.wav2vec_jamo import JamoPhonemeRecognizer
from voice_pron.pipeline.orchestrator import ModelBundle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading whisper model: %s", settings.whisper_model)
    asr = WhisperASR(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )

    logger.info("loading phoneme model: %s", settings.phoneme_model)
    phoneme = JamoPhonemeRecognizer(
        model_id=settings.phoneme_model,
        device=settings.phoneme_device,
        torch_dtype=torch.float16,
    )

    g2p = StandardPronouncer()

    app.state.models = ModelBundle(asr=asr, g2p=g2p, phoneme=phoneme)
    logger.info("models loaded")

    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("shutdown complete")


app = FastAPI(
    title="voice-pron",
    description="한국어 자모 단위 발음 정확도 측정 API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)
