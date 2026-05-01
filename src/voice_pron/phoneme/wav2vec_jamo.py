"""Kkonjeong/wav2vec2-base-korean 기반 자모 단위 phoneme 인식기.

CTC greedy decode로 호환자모 시퀀스를 추출하고, 한국어 음절 구조에 맞춰
초성/중성/종성 위치를 부여한다. CTC alignment를 사용해 자모별 시간 구간도 함께 산출한다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from voice_pron.audio.loader import AudioBuffer
from voice_pron.g2p.jamo_utils import JamoToken, syllabify_with_spaces


@dataclass
class TimedJamo:
    char: str
    pos: str
    syl: int
    start: float
    end: float
    logp: float


@dataclass
class PhonemeResult:
    jamo_tokens: list[JamoToken]
    timed_jamos: list[TimedJamo]
    ctc_confidence: float
    raw_sequence: list[str]


DEFAULT_MODEL_ID = "Kkonjeong/wav2vec2-base-korean"


class JamoPhonemeRecognizer:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id, dtype=torch_dtype).to(device)
        self.model.eval()
        self.pad_id = self.processor.tokenizer.pad_token_id
        # frame stride 계산용: wav2vec2-base는 conv stack에서 320 sample/frame (16kHz 기준)
        self._frame_samples = 320

    async def recognize(self, audio: AudioBuffer) -> PhonemeResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._recognize_sync, audio)

    @torch.inference_mode()
    def _recognize_sync(self, audio: AudioBuffer) -> PhonemeResult:
        inputs = self.processor(
            audio.samples,
            sampling_rate=audio.sr,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device, dtype=self.model.dtype)
        logits = self.model(input_values).logits[0]  # (T, V)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        pred_ids = log_probs.argmax(dim=-1).cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()

        raw_sequence: list[str] = []
        timed: list[TimedJamo] = []

        # CTC collapse: 같은 토큰 연속 + pad 토큰 제거
        prev_id = -1
        run_start_frame = 0
        run_logps: list[float] = []

        # 한 번에 collapse하면서 시간 구간을 함께 만들기
        for t, tok_id in enumerate(pred_ids):
            if tok_id == prev_id:
                run_logps.append(float(log_probs_np[t, tok_id]))
                continue
            # 직전 run 마감
            if prev_id != -1 and prev_id != self.pad_id:
                tok = self.processor.tokenizer.convert_ids_to_tokens(int(prev_id))
                if tok and tok != "[PAD]":
                    raw_sequence.append(tok)
                    timed.append(
                        TimedJamo(
                            char=tok,
                            pos="",  # 나중에 syllabify에서 채움
                            syl=-1,
                            start=run_start_frame * self._frame_samples / audio.sr,
                            end=t * self._frame_samples / audio.sr,
                            logp=float(np.mean(run_logps)) if run_logps else 0.0,
                        )
                    )
            prev_id = int(tok_id)
            run_start_frame = t
            run_logps = [float(log_probs_np[t, tok_id])]

        # 마지막 run 마감
        if prev_id != -1 and prev_id != self.pad_id:
            tok = self.processor.tokenizer.convert_ids_to_tokens(int(prev_id))
            if tok and tok != "[PAD]":
                raw_sequence.append(tok)
                timed.append(
                    TimedJamo(
                        char=tok,
                        pos="",
                        syl=-1,
                        start=run_start_frame * self._frame_samples / audio.sr,
                        end=len(pred_ids) * self._frame_samples / audio.sr,
                        logp=float(np.mean(run_logps)) if run_logps else 0.0,
                    )
                )

        # 음절화: raw_sequence에 위치 부여
        jamo_tokens = syllabify_with_spaces(raw_sequence)

        # timed list와 jamo_tokens 매칭: 공백을 제외한 raw 인덱스가 jamo_tokens와 1:1 대응
        # syllabify 과정에서 일부 자모가 누락될 수 있으므로 char 단위로 순회 매칭
        timed_no_space = [t for t in timed if t.char != " "]
        # syllabify 결과는 char 순서가 보존되므로 같은 인덱스끼리 위치/음절 정보 보강
        if len(timed_no_space) == len(jamo_tokens):
            for t_obj, jt in zip(timed_no_space, jamo_tokens):
                t_obj.pos = jt.pos
                t_obj.syl = jt.syl
        else:
            # 길이 불일치 (희귀): 최선으로 위치만 onset으로 채움
            for t_obj in timed_no_space:
                t_obj.pos = t_obj.pos or "onset"

        # 신뢰도: 모든 frame logp 평균(blank 제외)
        valid_mask = pred_ids != self.pad_id
        if valid_mask.any():
            best_logps = log_probs_np[np.arange(len(pred_ids)), pred_ids]
            ctc_confidence = float(np.exp(np.mean(best_logps[valid_mask])))
        else:
            ctc_confidence = 0.0

        return PhonemeResult(
            jamo_tokens=jamo_tokens,
            timed_jamos=timed_no_space,
            ctc_confidence=ctc_confidence,
            raw_sequence=raw_sequence,
        )
