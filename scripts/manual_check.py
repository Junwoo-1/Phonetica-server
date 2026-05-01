"""CLI: WAV/MP3 파일을 직접 파이프라인에 흘려 보고 자모별 점수를 출력한다.

사용: uv run python scripts/manual_check.py path/to/audio.wav
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

import torch

from voice_pron.asr.whisper_asr import WhisperASR
from voice_pron.audio.loader import load_audio_bytes
from voice_pron.config import settings
from voice_pron.g2p.pronouncer import StandardPronouncer
from voice_pron.phoneme.wav2vec_jamo import JamoPhonemeRecognizer
from voice_pron.pipeline.orchestrator import ModelBundle, run_pipeline


async def main(path: str) -> None:
    data = Path(path).read_bytes()
    audio = load_audio_bytes(data)
    print(f"loaded {audio.duration:.2f}s @ {audio.sr}Hz, silence_ratio={audio.silence_ratio:.3f}")

    print("loading models...")
    asr = WhisperASR(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    phoneme = JamoPhonemeRecognizer(
        model_id=settings.phoneme_model,
        device=settings.phoneme_device,
        torch_dtype=torch.float16,
    )
    g2p = StandardPronouncer()
    models = ModelBundle(asr=asr, g2p=g2p, phoneme=phoneme)

    print("running pipeline...\n")
    async for ev in run_pipeline(audio, models, str(uuid.uuid4())):
        payload = json.loads(ev["data"])
        name = payload["event"]
        body = payload["data"]
        if name == "asr_progress":
            print(f"[asr] {body['start']:.2f}-{body['end']:.2f}s: {body['text']}")
        elif name == "asr_completed":
            print(f"[ASR] text: {body['text']!r} (lang_prob={body['language_prob']})")
        elif name == "g2p_completed":
            print(f"[G2P] pronunciation: {body['pronunciation']!r}")
            print(f"      ref jamo: {[t['char'] for t in body['ref_jamo']]}")
        elif name == "phoneme_completed":
            print(f"[PHONEME] hyp jamo: {[t['char'] for t in body['hyp_jamo']]}")
            print(f"          ctc_confidence: {body['ctc_confidence']}")
        elif name == "score":
            print("\n=== SCORE ===")
            print(f"overall: {body['overall_score']}/100")
            print(f"PER: {body['per']}, weighted_PER: {body['weighted_per']}")
            print(f"counts: {body['counts']}")
            print("per_position:")
            for pos, stat in body["per_position"].items():
                print(f"  {pos:8s} {stat['matched']}/{stat['total']} = {stat['accuracy']}")
            if body["problem_jamos"]:
                print(f"problem jamos: {body['problem_jamos']}")
        elif name == "error":
            print(f"[ERROR] {body['code']}: {body['message']}")
        elif name == "done":
            print(f"\n[done] elapsed_ms={body['elapsed_ms']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: manual_check.py <audio_file>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
