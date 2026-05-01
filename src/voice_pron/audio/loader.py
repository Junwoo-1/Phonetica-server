"""오디오 파일 로딩, 16kHz mono 변환, 무음 비율 계산."""

from __future__ import annotations

import io
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
SILENCE_DB = -40.0  # 이 이상 조용하면 무음으로 본다


@dataclass
class AudioBuffer:
    samples: np.ndarray  # float32, mono, target_sr
    sr: int
    duration: float
    silence_ratio: float


class AudioLoadError(Exception):
    pass


def load_audio_bytes(data: bytes) -> AudioBuffer:
    """파일 바이트에서 16kHz mono float32 AudioBuffer를 생성한다.

    soundfile이 실패하면(예: mp3) librosa로 fallback (audioread/ffmpeg 사용).
    """
    if not data:
        raise AudioLoadError("empty audio payload")

    samples: np.ndarray
    sr: int
    try:
        with io.BytesIO(data) as buf:
            samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        try:
            samples, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
        except Exception as e:
            raise AudioLoadError(f"failed to decode audio: {e}") from e

    if samples.size == 0:
        raise AudioLoadError("decoded audio has no samples")

    # mono 변환
    if samples.ndim == 2:
        # soundfile은 (frames, channels), librosa는 (channels, frames). shape로 판별
        if samples.shape[0] < samples.shape[1]:
            samples = samples.mean(axis=0)
        else:
            samples = samples.mean(axis=1)
    samples = samples.astype(np.float32, copy=False)

    if sr != TARGET_SR:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    duration = float(len(samples) / sr)
    silence_ratio = _compute_silence_ratio(samples, sr)

    return AudioBuffer(
        samples=samples,
        sr=sr,
        duration=duration,
        silence_ratio=silence_ratio,
    )


def _compute_silence_ratio(samples: np.ndarray, sr: int, frame_ms: int = 30) -> float:
    """프레임별 RMS dB가 SILENCE_DB 이하인 프레임 비율."""
    if samples.size == 0:
        return 1.0
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, samples.size // frame_len)
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    silent = float(np.mean(db < SILENCE_DB))
    return silent
