"""파이프라인 통합 테스트.

실제 ASR/phoneme 모델 로딩은 무거우므로 mock 모델로 SSE 흐름과 단계별
이벤트 순서/형식만 검증한다. 실제 음성 검증은 scripts/manual_check.py로 수행.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import pytest

from voice_pron.asr.whisper_asr import ASRSegment
from voice_pron.audio.loader import AudioBuffer
from voice_pron.g2p.pronouncer import StandardPronouncer
from voice_pron.phoneme.wav2vec_jamo import PhonemeResult, TimedJamo
from voice_pron.pipeline.orchestrator import ModelBundle, run_pipeline


@dataclass
class _Info:
    language: str = "ko"
    language_probability: float = 0.99


class _MockASR:
    def __init__(self, text: str = "안녕"):
        self._text = text
        self._last_info = _Info()

    async def transcribe_stream(self, audio: AudioBuffer) -> AsyncIterator[ASRSegment]:
        yield ASRSegment(
            index=0,
            start=0.0,
            end=audio.duration,
            text=self._text,
            avg_logprob=-0.1,
            no_speech_prob=0.01,
        )

    async def transcribe_full(self, audio):
        from voice_pron.asr.whisper_asr import ASRResult

        segs = []
        async for s in self.transcribe_stream(audio):
            segs.append(s)
        return ASRResult(text=self._text, segments=segs, language="ko", language_prob=0.99)


class _MockPhoneme:
    def __init__(self, jamo_chars: list[str]):
        self._chars = jamo_chars

    async def recognize(self, audio):
        from voice_pron.g2p.jamo_utils import syllabify_with_spaces

        tokens = syllabify_with_spaces(self._chars)
        timed = [
            TimedJamo(char=t.char, pos=t.pos, syl=t.syl, start=0.0, end=0.1, logp=-0.05)
            for t in tokens
        ]
        return PhonemeResult(
            jamo_tokens=tokens,
            timed_jamos=timed,
            ctc_confidence=0.9,
            raw_sequence=self._chars,
        )


def _audio_buf(seconds: float = 1.0) -> AudioBuffer:
    sr = 16000
    return AudioBuffer(
        samples=np.zeros(int(sr * seconds), dtype=np.float32),
        sr=sr,
        duration=seconds,
        silence_ratio=0.1,
    )


def _parse(ev: dict[str, str]) -> tuple[str, dict]:
    payload = json.loads(ev["data"])
    return payload["event"], payload["data"]


@pytest.mark.asyncio
async def test_pipeline_perfect_match():
    models = ModelBundle(
        asr=_MockASR("안녕"),
        g2p=StandardPronouncer(),
        phoneme=_MockPhoneme(["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"]),
    )
    events = []
    async for ev in run_pipeline(_audio_buf(), models, str(uuid.uuid4())):
        events.append(_parse(ev))

    names = [e[0] for e in events]
    assert names == [
        "audio_loaded",
        "asr_progress",
        "asr_completed",
        "g2p_completed",
        "phoneme_completed",
        "alignment_completed",
        "score",
        "done",
    ]
    score = next(d for n, d in events if n == "score")
    assert score["overall_score"] == 100.0
    assert score["counts"]["match"] == 6


@pytest.mark.asyncio
async def test_pipeline_drops_coda():
    # 정답: 안녕 (ㅇㅏㄴ ㄴㅕㅇ), 발화: 아녀 (코다 모두 누락)
    models = ModelBundle(
        asr=_MockASR("안녕"),
        g2p=StandardPronouncer(),
        phoneme=_MockPhoneme(["ㅇ", "ㅏ", "ㄴ", "ㅕ"]),
    )
    events = []
    async for ev in run_pipeline(_audio_buf(), models, str(uuid.uuid4())):
        events.append(_parse(ev))

    score = next(d for n, d in events if n == "score")
    assert score["counts"]["del"] >= 1
    assert score["per_position"]["coda"]["accuracy"] < 1.0


@pytest.mark.asyncio
async def test_pipeline_silence_short_circuits():
    audio = AudioBuffer(
        samples=np.zeros(16000, dtype=np.float32), sr=16000, duration=1.0, silence_ratio=0.99
    )
    models = ModelBundle(
        asr=_MockASR("안녕"),
        g2p=StandardPronouncer(),
        phoneme=_MockPhoneme(["ㅇ", "ㅏ"]),
    )
    events = []
    async for ev in run_pipeline(audio, models, str(uuid.uuid4())):
        events.append(_parse(ev))

    names = [e[0] for e in events]
    assert names[-1] == "error"
    assert events[-1][1]["code"] == "NO_SPEECH"


@pytest.mark.asyncio
async def test_pipeline_no_hangul_text():
    models = ModelBundle(
        asr=_MockASR("hello world"),
        g2p=StandardPronouncer(),
        phoneme=_MockPhoneme(["ㅇ", "ㅏ"]),
    )
    events = []
    async for ev in run_pipeline(_audio_buf(), models, str(uuid.uuid4())):
        events.append(_parse(ev))

    names = [e[0] for e in events]
    assert "error" in names
    err = next(d for n, d in events if n == "error")
    assert err["code"] == "NO_HANGUL"
