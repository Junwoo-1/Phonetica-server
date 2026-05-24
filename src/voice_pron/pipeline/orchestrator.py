"""오디오 처리 파이프라인: ASR → G2P → Phoneme → Alignment → Score.

각 단계 결과를 SSE 이벤트 dict로 yield한다.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from voice_pron.align.cost import jamo_distance, reference_weight
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
        "detailed_jamos": report.detailed_jamos, # ⭐️ [NEW] 디테일 점수 직렬화
    }


async def run_pipeline(
    audio: AudioBuffer,
    models: ModelBundle,
    word_cache: dict[str, list[JamoToken]], # ⭐️ [NEW] 캐시 딕셔너리
    candidates_str: str,                    # ⭐️ [NEW] 유니티가 보낸 후보군
    request_id: str,
) -> AsyncIterator[dict[str, Any]]:
    started = time.perf_counter()

    # 후보군 문자열을 리스트로 변환
    candidate_words = [c.strip() for c in candidates_str.split(",") if c.strip()]
    if not candidate_words:
        yield event("error", {"code": "NO_CANDIDATES", "message": "전달된 후보 단어가 없습니다."}, request_id)
        return

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

    # phoneme 인식은 백그라운드로 시작
    phon_task = asyncio.create_task(models.phoneme.recognize(audio))

    # ASR (보조 지표/거울 피드백용)
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
        yield event("error", {"code": "ASR_FAILED", "message": f"음성 인식 실패: {e}"}, request_id)
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

    # 기존에는 여기서 한글이 없으면 abort 했으나, 이제는 후보군 기준으로 채점하므로 스킵합니다.

    # phoneme 결과 대기
    try:
        phoneme_result = await phon_task
    except Exception as e:
        yield event("error", {"code": "PHONEME_FAILED", "message": f"음소 인식 실패: {e}"}, request_id)
        return

    yield event(
        "phoneme_completed",
        {
            "hyp_jamo": [
                {
                    "char": t.char, "pos": t.pos, "syl": t.syl,
                    "start": round(t.start, 3), "end": round(t.end, 3), "logp": round(t.logp, 4)
                }
                for t in phoneme_result.timed_jamos
            ],
            "ctc_confidence": round(phoneme_result.ctc_confidence, 4),
        },
        request_id,
    )

    # ⭐️ [NEW] 하이브리드 매칭 (Alignment Loop)
    best_word = None
    best_ref_jamo = None
    best_alignment = None
    min_error_rate = float("inf")

    # 유니티가 보낸 단어들을 하나씩 비교합니다.
    for word in candidate_words:
        # 캐시된 자모가 있으면 즉시 가져오고, 없으면 G2P를 돌립니다.
        if word in word_cache:
            cand_jamo = word_cache[word]
        else:
            cand_jamo = models.g2p.to_jamo_sequence(models.g2p.to_pronunciation(word))

        alignment = nw_align(cand_jamo, phoneme_result.jamo_tokens, jamo_distance)
        
        # 길이 차이 보정을 위해 가중치 대비 에러율을 산출합니다.
        total_weight = sum(reference_weight(t) for t in cand_jamo) or 1.0
        error_rate = alignment.total_cost / total_weight

        if error_rate < min_error_rate:
            min_error_rate = error_rate
            best_word = word
            best_ref_jamo = cand_jamo
            best_alignment = alignment

    # 가장 매칭이 잘 된 단어로 G2P 이벤트를 쏴줍니다 (유니티 호환성 유지)
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

    # 최종 채점 
    score = compute_scores(best_alignment, best_ref_jamo)
    score_payload = _serialize_score(score)
    
    # ⭐️ [NEW] 매칭된 정답과 Whisper 오인식 텍스트를 함께 내려줍니다.
    score_payload["recognized_word"] = best_word
    score_payload["whisper_text"] = asr_result.text 

    score_payload["low_confidence"] = (
        asr_result.language_prob < settings.low_language_prob
        or phoneme_result.ctc_confidence < 0.3
    )
    score_payload["ref_jamo"] = [_serialize_jamo(t) for t in best_ref_jamo]
    yield event("score", score_payload, request_id)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    yield event("done", {"elapsed_ms": elapsed_ms}, request_id)
