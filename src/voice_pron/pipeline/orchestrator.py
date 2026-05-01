"""오디오 처리 파이프라인: ASR → G2P → Phoneme → Alignment → Score.

각 단계 결과를 SSE 이벤트 dict로 yield한다.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from voice_pron.align.cost import jamo_distance
from voice_pron.align.needleman import align as nw_align
from voice_pron.api.sse import event
from voice_pron.asr.whisper_asr import ASRResult, ASRSegment, WhisperASR
from voice_pron.audio.loader import AudioBuffer
from voice_pron.config import settings
from voice_pron.g2p.jamo_utils import JamoToken
from voice_pron.g2p.pronouncer import StandardPronouncer
from voice_pron.phoneme.wav2vec_jamo import JamoPhonemeRecognizer
from voice_pron.scoring.score import ScoreReport, compute_scores


@dataclass
class ModelBundle:
    asr: WhisperASR
    g2p: StandardPronouncer
    phoneme: JamoPhonemeRecognizer


def _serialize_jamo(t: JamoToken) -> dict[str, Any]:
    return {"char": t.char, "pos": t.pos, "syl": t.syl}


def _serialize_score(report: ScoreReport) -> dict[str, Any]:
    return {
        "overall_score": report.overall_score,
        "per": report.per,
        "weighted_per": report.weighted_per,
        "counts": report.counts,
        "per_position": {
            pos: {
                "matched": stat.matched,
                "total": stat.total,
                "accuracy": round(stat.accuracy, 4),
            }
            for pos, stat in report.per_position.items()
        },
        "per_jamo": {
            key: {
                "ref_count": stat.ref_count,
                "correct": stat.correct,
                "errors": stat.errors,
            }
            for key, stat in report.per_jamo.items()
        },
        "problem_jamos": report.problem_jamos,
    }


async def run_pipeline(
    audio: AudioBuffer,
    models: ModelBundle,
    request_id: str,
) -> AsyncIterator[dict[str, str]]:
    started = time.perf_counter()

    yield event(
        "audio_loaded",
        {
            "duration_sec": round(audio.duration, 3),
            "sample_rate": audio.sr,
            "silence_ratio": round(audio.silence_ratio, 4),
        },
        request_id,
    )

    if audio.silence_ratio >= settings.silence_threshold:
        yield event(
            "error",
            {"code": "NO_SPEECH", "message": "오디오에 발화가 감지되지 않았습니다."},
            request_id,
        )
        return

    # phoneme 인식은 백그라운드로 시작 (ASR과 동시 진행)
    phon_task = asyncio.create_task(models.phoneme.recognize(audio))

    # ASR 세그먼트를 스트리밍하며 progress 이벤트 방출, 완료 후 결과 조립
    asr_segments: list[ASRSegment] = []
    try:
        async for seg in models.asr.transcribe_stream(audio):
            asr_segments.append(seg)
            yield event(
                "asr_progress",
                {
                    "segment_index": seg.index,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text,
                    "avg_logprob": round(seg.avg_logprob, 4),
                },
                request_id,
            )
    except Exception as e:
        phon_task.cancel()
        yield event(
            "error",
            {"code": "ASR_FAILED", "message": f"음성 인식 실패: {e}"},
            request_id,
        )
        return

    info = getattr(models.asr, "_last_info", None)
    asr_result = ASRResult(
        text=" ".join(s.text for s in asr_segments).strip(),
        segments=asr_segments,
        language=info.language if info else "ko",
        language_prob=float(info.language_probability) if info else 0.0,
    )

    yield event(
        "asr_completed",
        {
            "text": asr_result.text,
            "language": asr_result.language,
            "language_prob": round(asr_result.language_prob, 4),
        },
        request_id,
    )

    if not asr_result.text:
        phon_task.cancel()
        yield event(
            "error",
            {"code": "NO_SPEECH", "message": "인식된 텍스트가 없습니다."},
            request_id,
        )
        return

    # G2P
    pronunciation = models.g2p.to_pronunciation(asr_result.text)
    ref_jamo = models.g2p.to_jamo_sequence(asr_result.text)
    yield event(
        "g2p_completed",
        {
            "pronunciation": pronunciation,
            "ref_jamo": [_serialize_jamo(t) for t in ref_jamo],
        },
        request_id,
    )

    if not ref_jamo:
        phon_task.cancel()
        yield event(
            "error",
            {
                "code": "NO_HANGUL",
                "message": "인식된 텍스트에 한글이 없어 평가할 수 없습니다.",
            },
            request_id,
        )
        return

    # phoneme 결과 대기
    try:
        phoneme_result = await phon_task
    except Exception as e:
        yield event(
            "error",
            {"code": "PHONEME_FAILED", "message": f"음소 인식 실패: {e}"},
            request_id,
        )
        return

    yield event(
        "phoneme_completed",
        {
            "hyp_jamo": [
                {
                    "char": t.char,
                    "pos": t.pos,
                    "syl": t.syl,
                    "start": round(t.start, 3),
                    "end": round(t.end, 3),
                    "logp": round(t.logp, 4),
                }
                for t in phoneme_result.timed_jamos
            ],
            "ctc_confidence": round(phoneme_result.ctc_confidence, 4),
        },
        request_id,
    )

    # 정렬 + 점수
    alignment = nw_align(ref_jamo, phoneme_result.jamo_tokens, jamo_distance)
    yield event(
        "alignment_completed",
        {
            "pairs": [
                {
                    "ref": f"{p.ref.char}@{p.ref.pos}" if p.ref else None,
                    "hyp": f"{p.hyp.char}@{p.hyp.pos}" if p.hyp else None,
                    "op": p.op,
                    "cost": round(p.cost, 4),
                }
                for p in alignment.pairs
            ],
            "total_cost": round(alignment.total_cost, 4),
        },
        request_id,
    )

    score = compute_scores(alignment, ref_jamo)
    score_payload = _serialize_score(score)
    score_payload["low_confidence"] = (
        asr_result.language_prob < settings.low_language_prob
        or phoneme_result.ctc_confidence < 0.3
    )
    yield event("score", score_payload, request_id)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    yield event("done", {"elapsed_ms": elapsed_ms}, request_id)
