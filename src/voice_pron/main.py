"""FastAPI 엔트리포인트 + lifespan (모델 1회 로드 및 WordBank 캐싱)."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from voice_pron.api.routes import router
from voice_pron.config import settings
from voice_pron.g2p.jamo_utils import decompose_hangul
from voice_pron.g2p.pronouncer import StandardPronouncer
from voice_pron.phoneme.wav2vec_jamo import JamoPhonemeRecognizer
from voice_pron.pipeline.orchestrator import ModelBundle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

WORDBANK_PATH = Path(__file__).resolve().parents[2] / "WordBank.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading phoneme model: %s", settings.phoneme_model)
    phoneme = JamoPhonemeRecognizer(
        model_id=settings.phoneme_model,
        device=settings.phoneme_device,
        torch_dtype=torch.float16,
    )

    g2p = StandardPronouncer()
    app.state.models = ModelBundle(g2p=g2p, phoneme=phoneme)

    word_cache: dict = {}
    if WORDBANK_PATH.exists():
        with open(WORDBANK_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("wordList", []):
            word = item["word"]
            pron = item.get("pronunciation")
            if pron:
                # 수동 발음형이 제공되면 g2pkk를 건너뛰고 직접 자모 분해
                jamo_seq = decompose_hangul(pron)
            else:
                jamo_seq = g2p.to_jamo_sequence(word)
            word_cache[word] = jamo_seq
        logger.info("cached %d words from %s", len(word_cache), WORDBANK_PATH)
    else:
        logger.warning("WordBank not found at %s, cache is empty", WORDBANK_PATH)

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
    description="한국어 자모 단위 발음 정확도 측정 API (closed-vocab 매칭)",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(router)
