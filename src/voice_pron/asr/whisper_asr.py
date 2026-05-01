"""faster-whisper 기반 한국어 ASR 래퍼.

faster-whisper의 segment generator는 동기 iterator이므로 ThreadPoolExecutor로
async 환경에서 안전하게 소비한다.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from faster_whisper import WhisperModel

from voice_pron.audio.loader import AudioBuffer


@dataclass
class ASRSegment:
    index: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


@dataclass
class ASRResult:
    text: str
    segments: list[ASRSegment]
    language: str
    language_prob: float


class WhisperASR:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    async def transcribe_stream(self, audio: AudioBuffer) -> AsyncIterator[ASRSegment]:
        """faster-whisper segment generator를 async iterator로 변환한다."""
        loop = asyncio.get_running_loop()
        # transcribe()는 (segments_iter, info)를 즉시 반환하지만 segments는 lazy
        segments_iter, info = await loop.run_in_executor(
            None, self._transcribe_sync, audio
        )
        idx = 0
        # iterator를 thread executor에서 next() 호출로 끌어낸다
        sentinel = object()

        def _next():
            try:
                return next(segments_iter)
            except StopIteration:
                return sentinel

        while True:
            seg = await loop.run_in_executor(None, _next)
            if seg is sentinel:
                break
            yield ASRSegment(
                index=idx,
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                avg_logprob=float(seg.avg_logprob),
                no_speech_prob=float(seg.no_speech_prob),
            )
            idx += 1
        # info는 finalize에서 사용
        self._last_info = info

    async def transcribe_full(self, audio: AudioBuffer) -> ASRResult:
        """모든 세그먼트를 모아 단일 결과로 반환한다."""
        segments: list[ASRSegment] = []
        async for seg in self.transcribe_stream(audio):
            segments.append(seg)
        info = getattr(self, "_last_info", None)
        full_text = " ".join(s.text for s in segments).strip()
        return ASRResult(
            text=full_text,
            segments=segments,
            language=info.language if info else "ko",
            language_prob=float(info.language_probability) if info else 0.0,
        )

    def _transcribe_sync(self, audio: AudioBuffer):
        return self.model.transcribe(
            audio.samples,
            language="ko",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
