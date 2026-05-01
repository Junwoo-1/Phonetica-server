"""한글 자모 분해와 위치(초/중/종) 태깅 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jamo import h2j, j2hcj

Position = Literal["onset", "nucleus", "coda"]


@dataclass(frozen=True)
class JamoToken:
    char: str
    pos: Position
    syl: int

    def key(self) -> tuple[str, Position]:
        return (self.char, self.pos)


HANGUL_SYLLABLE_START = 0xAC00
HANGUL_SYLLABLE_END = 0xD7A3

# 호환 자모(compatibility jamo, U+3131~U+3163) 범위
COMPAT_CONSONANTS = {
    "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄸ", "ㄹ",
    "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
    "ㅃ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ",
    "ㅌ", "ㅍ", "ㅎ",
}
COMPAT_VOWELS = {
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
    "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
}


def is_hangul_syllable(ch: str) -> bool:
    return len(ch) == 1 and HANGUL_SYLLABLE_START <= ord(ch) <= HANGUL_SYLLABLE_END


def is_consonant(ch: str) -> bool:
    return ch in COMPAT_CONSONANTS


def is_vowel(ch: str) -> bool:
    return ch in COMPAT_VOWELS


def to_compat(ch: str) -> str:
    """초/중/종 자모(U+1100~U+11FF)를 호환자모(U+3130~U+318F)로 변환."""
    return j2hcj(ch)


def decompose_hangul(text: str) -> list[JamoToken]:
    """한글 문자열을 (자모, 위치, 음절번호) 토큰 목록으로 분해한다.

    음절이 아닌 문자(공백, 영문 등)는 건너뛴다. 음절번호는 한글 음절마다 증가한다.
    종성이 없는 음절은 (초성, 중성)만 생성된다.
    """
    tokens: list[JamoToken] = []
    syl = 0
    for ch in text:
        if not is_hangul_syllable(ch):
            continue
        decomposed = j2hcj(h2j(ch))
        if len(decomposed) == 2:
            onset, nucleus = decomposed
            tokens.append(JamoToken(onset, "onset", syl))
            tokens.append(JamoToken(nucleus, "nucleus", syl))
        elif len(decomposed) == 3:
            onset, nucleus, coda = decomposed
            tokens.append(JamoToken(onset, "onset", syl))
            tokens.append(JamoToken(nucleus, "nucleus", syl))
            tokens.append(JamoToken(coda, "coda", syl))
        else:
            # 안전장치: 분해 결과가 비정상이면 무시
            continue
        syl += 1
    return tokens


def syllabify_flat_jamo(seq: list[str], start_syl: int = 0) -> list[JamoToken]:
    """평탄한 자모 시퀀스(공백 제외, 호환자모만)를 (초/중/종) 위치를 부여해 토큰화한다.

    Wav2Vec2 phoneme 모델은 위치 정보 없이 호환자모만 평탄하게 출력하므로,
    한국어 음절 구조(C? V C?)에 맞춰 위치를 추론한다.

    규칙:
    - 자음 → 다음 토큰이 모음이면 onset, 아니면 직전 모음의 coda
    - 모음 → nucleus, 새 음절 시작
    - 자음만 떠 있으면(orphan) 가장 최근 음절의 coda로 기록
    """
    tokens: list[JamoToken] = []
    syl = start_syl
    i = 0
    n = len(seq)

    while i < n:
        ch = seq[i]
        # 자음 시작
        if is_consonant(ch):
            # 다음에 모음이 오면 onset, 아니면 (선행 음절이 있으면) coda로 처리
            if i + 1 < n and is_vowel(seq[i + 1]):
                tokens.append(JamoToken(ch, "onset", syl))
                tokens.append(JamoToken(seq[i + 1], "nucleus", syl))
                i += 2
                # 다음 자음을 보고 coda 여부 판단
                # - 다음 자음 다음이 자음이거나 끝이면 → 현재 자음은 coda (CVCCV 또는 CVC끝)
                # - 다음 자음 다음이 모음이면 → 현재 자음은 다음 음절의 onset (CVCV 패턴)
                if i < n and is_consonant(seq[i]):
                    j = i + 1
                    if j >= n or is_consonant(seq[j]):
                        tokens.append(JamoToken(seq[i], "coda", syl))
                        i += 1
                syl += 1
            else:
                # 모음이 안 따라옴: 직전 음절의 coda로 (가능하면)
                if tokens:
                    tokens.append(JamoToken(ch, "coda", syl - 1))
                else:
                    # 첫 토큰이 자음만 있고 모음이 없으면 onset으로라도 기록
                    tokens.append(JamoToken(ch, "onset", syl))
                    syl += 1
                i += 1
        elif is_vowel(ch):
            # onset 없이 모음 시작 (드문 경우)
            tokens.append(JamoToken(ch, "nucleus", syl))
            i += 1
            if i < n and is_consonant(seq[i]):
                if i + 1 >= n or is_consonant(seq[i + 1]):
                    tokens.append(JamoToken(seq[i], "coda", syl))
                    i += 1
                else:
                    tokens.append(JamoToken(seq[i], "coda", syl))
                    i += 1
            syl += 1
        else:
            # 알 수 없는 토큰은 건너뜀
            i += 1
    return tokens


def syllabify_with_spaces(seq: list[str]) -> list[JamoToken]:
    """공백을 음절 경계로 사용하지 않고 어절 경계로만 사용한다.

    공백은 음절번호를 리셋하지 않지만, 분리해서 어절을 인식할 수 있다.
    현재는 공백을 단순히 무시하고 음절번호를 연속해서 부여한다.
    """
    out: list[JamoToken] = []
    chunk: list[str] = []
    syl_offset = 0
    for ch in seq:
        if ch == " ":
            if chunk:
                tokens = syllabify_flat_jamo(chunk, start_syl=syl_offset)
                out.extend(tokens)
                if tokens:
                    syl_offset = tokens[-1].syl + 1
                chunk = []
        else:
            chunk.append(ch)
    if chunk:
        tokens = syllabify_flat_jamo(chunk, start_syl=syl_offset)
        out.extend(tokens)
    return out
