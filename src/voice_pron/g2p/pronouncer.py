"""g2pkk를 이용해 한글 텍스트를 표준 발음형으로 변환하고 자모 시퀀스를 만든다."""

from __future__ import annotations

import re

from g2pkk import G2p

from voice_pron.g2p.jamo_utils import JamoToken, decompose_hangul, is_hangul_syllable

_NON_HANGUL_RE = re.compile(r"[^가-힣\s]+")


class StandardPronouncer:
    """한글 → 표준 발음형 한글 → 자모 시퀀스 변환기."""

    def __init__(self) -> None:
        self._g2p = G2p()

    @staticmethod
    def normalize(text: str) -> str:
        """비한글(영문/숫자/구두점)을 공백으로 치환한다.

        공백을 보존해 어절 경계는 유지한다. g2pkk가 영문/숫자를 보존하지만
        본 시스템은 한국어 자모 비교만 수행하므로 미리 제거한다.
        """
        cleaned = _NON_HANGUL_RE.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def to_pronunciation(self, text: str) -> str:
        """한글 문자열을 표준 발음형 한글로 변환한다.

        예: "굳이" → "구지", "같이" → "가치", "꽃이" → "꼬치"
        비한글은 normalize 단계에서 제거된다.
        """
        norm = self.normalize(text)
        if not norm:
            return ""
        try:
            return self._g2p(norm)
        except Exception:
            # g2pkk 내부 오류 시 원본 텍스트로 fallback
            return norm

    def to_jamo_sequence(self, text: str) -> list[JamoToken]:
        """텍스트를 표준 발음형으로 변환한 뒤 자모 토큰 시퀀스로 분해한다."""
        pron = self.to_pronunciation(text)
        # 발음형에 비한글이 섞여 있을 수 있어 한 번 더 거른다
        pron = "".join(ch for ch in pron if is_hangul_syllable(ch) or ch == " ")
        return decompose_hangul(pron)
