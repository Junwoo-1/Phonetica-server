"""오디오 처리 파이프라인: Phoneme → 후보 단어 정렬 → Score.

각 단계 결과를 SSE 이벤트 dict로 yield한다.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from voice_pron.align.cost import jamo_distance, reference_weight
from voice_pron.align.needleman import align as nw_align
from voice_pron.api.sse import event
from voice_pron.audio.loader import AudioBuffer
from voice_pron.config import settings
from voice_pron.g2p.jamo_utils import JamoToken
from voice_pron.g2p.pronouncer import StandardPronouncer
from voice_pron.phoneme.wav2vec_jamo import JamoPhonemeRecognizer
from voice_pron.scoring.score import ScoreReport, compute_scores


@dataclass
class ModelBundle:
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
        "detailed_jamos": report.detailed_jamos,
    }


async def run_pipeline(
    audio: AudioBuffer,
    models: ModelBundle,
    word_cache: dict[str, list[JamoToken]],
    candidates_str: str,
    request_id: str,
) -> AsyncIterator[dict[str, Any]]:
    started = time.perf_counter()

    candidate_words = [c.strip() for c in candidates_str.split(",") if c.strip()]

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

    try:
        phoneme_result = await models.phoneme.recognize(audio)
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

    # 후보 단어들과 NW 정렬해 weighted error rate 최저 후보를 정답으로 선택
    best_word: str | None = None
    best_ref_jamo: list[JamoToken] | None = None
    best_alignment = None
    min_error_rate = float("inf")

    for word in candidate_words:
        cand_jamo = word_cache.get(word) or models.g2p.to_jamo_sequence(word)
        if not cand_jamo:
            continue
        alignment = nw_align(cand_jamo, phoneme_result.jamo_tokens, jamo_distance)
        total_weight = sum(reference_weight(t) for t in cand_jamo) or 1.0
        error_rate = alignment.total_cost / total_weight
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            best_word = word
            best_ref_jamo = cand_jamo
            best_alignment = alignment

    if best_word is None or best_ref_jamo is None or best_alignment is None:
        yield event(
            "error",
            {"code": "NO_VALID_CANDIDATES", "message": "후보 단어를 분해할 수 없습니다."},
            request_id,
        )
        return

    pronunciation = models.g2p.to_pronunciation(best_word)
    yield event(
        "g2p_completed",
        {
            "pronunciation": pronunciation,
            "ref_jamo": [_serialize_jamo(t) for t in best_ref_jamo],
        },
        request_id,
    )

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
                for p in best_alignment.pairs
            ],
            "total_cost": round(best_alignment.total_cost, 4),
        },
        request_id,
    )

    score = compute_scores(best_alignment, best_ref_jamo)
    score_payload = _serialize_score(score)
    score_payload["recognized_word"] = best_word
    score_payload["heard_jamos"] = [
        {"syl": t.syl, "pos": t.pos, "char": t.char}
        for t in phoneme_result.jamo_tokens
    ]
    score_payload["low_confidence"] = phoneme_result.ctc_confidence < 0.3
    score_payload["ref_jamo"] = [_serialize_jamo(t) for t in best_ref_jamo]
    yield event("score", score_payload, request_id)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    yield event("done", {"elapsed_ms": elapsed_ms}, request_id)
