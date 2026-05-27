"""파이프라인 통합 테스트.

실제 phoneme 모델 로딩은 무거우므로 mock 모델로 SSE 흐름과 단계별
이벤트 순서/형식만 검증한다. 실제 음성 검증은 scripts/manual_check.py로 수행.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from voice_pron.audio.loader import AudioBuffer
from voice_pron.g2p.pronouncer import StandardPronouncer
from voice_pron.phoneme.wav2vec_jamo import PhonemeResult, TimedJamo
from voice_pron.pipeline.orchestrator import ModelBundle, run_pipeline


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


def _bundle(jamo_chars: list[str]) -> ModelBundle:
    return ModelBundle(g2p=StandardPronouncer(), phoneme=_MockPhoneme(jamo_chars))


@pytest.mark.asyncio
async def test_pipeline_perfect_match():
    models = _bundle(["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"])
    events = []
    async for ev in run_pipeline(
        _audio_buf(),
        models,
        word_cache={},
        candidates_str="안녕,사과",
        request_id=str(uuid.uuid4()),
    ):
        events.append(_parse(ev))

    names = [e[0] for e in events]
    assert names == [
        "audio_loaded",
        "phoneme_completed",
        "g2p_completed",
        "alignment_completed",
        "score",
        "done",
    ]
    score = next(d for n, d in events if n == "score")
    assert score["recognized_word"] == "안녕"
    assert score["overall_score"] == 100.0
    assert score["counts"]["match"] == 6
    assert score["detailed_jamos"]
    assert all(d["score"] == 100.0 for d in score["detailed_jamos"])
    assert score["heard_jamos"]
    assert [d["char"] for d in score["heard_jamos"]] == ["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"]


@pytest.mark.asyncio
async def test_pipeline_drops_coda():
    # 정답: 안녕 (ㅇㅏㄴ ㄴㅕㅇ), 발화: 아녀 (코다 모두 누락) → del 케이스
    models = _bundle(["ㅇ", "ㅏ", "ㄴ", "ㅕ"])
    events = []
    async for ev in run_pipeline(
        _audio_buf(),
        models,
        word_cache={},
        candidates_str="안녕",
        request_id=str(uuid.uuid4()),
    ):
        events.append(_parse(ev))

    score = next(d for n, d in events if n == "score")
    assert score["counts"]["del"] >= 1
    assert score["per_position"]["coda"]["accuracy"] < 1.0
    # 누락된 coda 위치는 detailed_jamos에 score=0으로 기록
    coda_entries = [d for d in score["detailed_jamos"] if d["pos"] == "coda"]
    assert any(d["score"] == 0.0 for d in coda_entries)
    # heard_jamos에는 발음한 4개 자모만 존재
    assert len(score["heard_jamos"]) == 4


@pytest.mark.asyncio
async def test_pipeline_substitutes_consonant():
    # 정답: 안녕 (ㅇㅏㄴ ㄴㅕㅇ), 발화: 안며ㅇ (둘째 음절 onset ㄴ→ㅁ) → sub 케이스
    models = _bundle(["ㅇ", "ㅏ", "ㄴ", "ㅁ", "ㅕ", "ㅇ"])
    events = []
    async for ev in run_pipeline(
        _audio_buf(),
        models,
        word_cache={},
        candidates_str="안녕",
        request_id=str(uuid.uuid4()),
    ):
        events.append(_parse(ev))

    score = next(d for n, d in events if n == "score")
    assert score["counts"]["sub"] >= 1
    # 치환된 위치는 100점이 아님
    onsets = [d for d in score["detailed_jamos"] if d["pos"] == "onset"]
    assert any(d["score"] < 100.0 for d in onsets)
    # heard_jamos에는 사용자가 실제로 발음한 ㅁ이 포함됨 (정답 ㄴ과 다름)
    heard_chars = [d["char"] for d in score["heard_jamos"]]
    assert "ㅁ" in heard_chars


@pytest.mark.asyncio
async def test_pipeline_extra_jamo_insertion():
    # 정답: 사과 (ㅅㅏ ㄱㅘ), 발화: 사ㄴ과 (사이에 ㄴ 추가) → ins 케이스
    models = _bundle(["ㅅ", "ㅏ", "ㄴ", "ㄱ", "ㅘ"])
    events = []
    async for ev in run_pipeline(
        _audio_buf(),
        models,
        word_cache={},
        candidates_str="사과",
        request_id=str(uuid.uuid4()),
    ):
        events.append(_parse(ev))

    score = next(d for n, d in events if n == "score")
    assert score["counts"]["ins"] >= 1
    # 추가된 자모는 ref 좌표계에 없으므로 detailed_jamos는 정답 단어 길이 그대로
    assert len(score["detailed_jamos"]) == 4  # 사과 = ㅅㅏㄱㅘ
    # heard_jamos에는 추가된 ㄴ 포함
    heard_chars = [d["char"] for d in score["heard_jamos"]]
    assert heard_chars.count("ㄴ") == 1
    assert len(score["heard_jamos"]) == 5


@pytest.mark.asyncio
async def test_pipeline_silence_short_circuits():
    audio = AudioBuffer(
        samples=np.zeros(16000, dtype=np.float32), sr=16000, duration=1.0, silence_ratio=0.99
    )
    models = _bundle(["ㅇ", "ㅏ"])
    events = []
    async for ev in run_pipeline(
        audio, models, word_cache={}, candidates_str="안녕", request_id=str(uuid.uuid4())
    ):
        events.append(_parse(ev))

    names = [e[0] for e in events]
    assert names[-1] == "error"
    assert events[-1][1]["code"] == "NO_SPEECH"


@pytest.mark.asyncio
async def test_pipeline_uses_word_cache():
    # word_cache에 있는 단어는 캐시 시퀀스를 사용
    g2p = StandardPronouncer()
    cached = g2p.to_jamo_sequence("안녕")
    models = _bundle(["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"])
    events = []
    async for ev in run_pipeline(
        _audio_buf(),
        models,
        word_cache={"안녕": cached},
        candidates_str="안녕",
        request_id=str(uuid.uuid4()),
    ):
        events.append(_parse(ev))

    score = next(d for n, d in events if n == "score")
    assert score["recognized_word"] == "안녕"
    assert score["overall_score"] == 100.0


@pytest.mark.asyncio
async def test_pipeline_picks_best_candidate():
    # 사용자가 "안녕"을 발음 → 후보 "안녕"이 "사과"보다 잘 맞아야 함
    models = _bundle(["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"])
    events = []
    async for ev in run_pipeline(
        _audio_buf(),
        models,
        word_cache={},
        candidates_str="사과,안녕,포도",
        request_id=str(uuid.uuid4()),
    ):
        events.append(_parse(ev))

    score = next(d for n, d in events if n == "score")
    assert score["recognized_word"] == "안녕"
