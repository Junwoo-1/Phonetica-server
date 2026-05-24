"""POST /pronounce: 오디오 파일 업로드 → SSE 스트림."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse

from voice_pron.api.sse import event
from voice_pron.audio.loader import AudioLoadError, load_audio_bytes
from voice_pron.config import settings
from voice_pron.pipeline.orchestrator import ModelBundle, run_pipeline

router = APIRouter()

# 단일 GPU에서 동시 추론 직렬화
_gpu_semaphore = asyncio.Semaphore(settings.concurrency)


@router.post("/pronounce")
async def pronounce(
    request: Request,
    file: UploadFile = File(...),
    candidates: str = Form(...),  # ⭐️ [NEW] 유니티가 보낸 후보군 목록 받기
):
    if file.size is not None and file.size > settings.max_file_bytes:
        raise HTTPException(status_code=413, detail="file too large")

    data = await file.read()
    if len(data) > settings.max_file_bytes:
        raise HTTPException(status_code=413, detail="file too large")

    request_id = str(uuid.uuid4())
    models: ModelBundle = request.app.state.models
    word_cache: dict = request.app.state.word_cache  # ⭐️ [NEW] 캐싱된 단어장

    try:
        audio = load_audio_bytes(data)
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if audio.duration > settings.max_duration_sec:
        raise HTTPException(
            status_code=413,
            detail=f"audio too long ({audio.duration:.1f}s > {settings.max_duration_sec}s)",
        )

    async def stream():
        yield event(
            "accepted",
            {
                "filename": file.filename,
                "size_bytes": len(data),
            },
            request_id,
        )
        async with _gpu_semaphore:
            try:
                # ⭐️ [NEW] run_pipeline에 캐시와 후보 단어 문자열을 넘겨줍니다.
                async for ev in run_pipeline(audio, models, word_cache, candidates, request_id):
                    if await request.is_disconnected():
                        break
                    yield ev
            except Exception as e:
                yield event(
                    "error",
                    {"code": "INTERNAL", "message": str(e)},
                    request_id,
                )

    return EventSourceResponse(stream())


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    ready = hasattr(request.app.state, "models")
    return {"status": "ok" if ready else "loading"}
