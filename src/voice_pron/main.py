"""FastAPI 엔트리포인트 + lifespan (모델 1회 로드 및 단어 캐싱)."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

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

    # ⭐️ [NEW] WordBank.json을 읽어 서버 메모리에 캐싱(사전 연산)합니다.
    word_cache = {}
    wordbank_path = Path("WordBank.json")
    if wordbank_path.exists():
        with open(wordbank_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("wordList", []):
                word = item["word"]
                pron = item.get("pronunciation", word)
                jamo_seq = g2p.to_jamo_sequence(pron)
                word_cache[word] = jamo_seq
        logger.info("WordBank.json에서 %d개의 단어를 성공적으로 캐싱했습니다.", len(word_cache))
    else:
        logger.warning("WordBank.json 파일을 찾을 수 없습니다. (%s) 캐시가 비어있습니다.", wordbank_path)
        
    app.state.word_cache = word_cache
    logger.info("models and cache loaded")

    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("shutdown complete")


app = FastAPI(
    title="voice-pron",
    description="한국어 자모 단위 발음 정확도 측정 API (하이브리드 매칭)",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(router)
