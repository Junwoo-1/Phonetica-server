"""자모 거리(substitution/insertion/deletion) 비용 함수."""

from __future__ import annotations

from voice_pron.g2p.jamo_utils import JamoToken, is_consonant, is_vowel

# 평음 / 경음 / 격음 그룹: 같은 그룹 내 치환은 약한 페널티
CONSONANT_FAMILIES: list[set[str]] = [
    {"ㄱ", "ㄲ", "ㅋ"},        # 연구개음
    {"ㄷ", "ㄸ", "ㅌ"},        # 치경음(파열)
    {"ㅂ", "ㅃ", "ㅍ"},        # 양순음
    {"ㅈ", "ㅉ", "ㅊ"},        # 치경구개음(파찰)
    {"ㅅ", "ㅆ"},              # 치경음(마찰)
    {"ㄴ", "ㅁ", "ㅇ"},        # 비음
    {"ㄹ"},                    # 유음
    {"ㅎ"},                    # 후두음
]

# 같은 조음위치이지만 다른 방법(예: 비음↔파열음)은 중간 페널티
PLACE_GROUPS: list[set[str]] = [
    {"ㄱ", "ㄲ", "ㅋ", "ㅇ"},                        # 연구개
    {"ㄷ", "ㄸ", "ㅌ", "ㄴ", "ㄹ", "ㅅ", "ㅆ"},     # 치경
    {"ㅂ", "ㅃ", "ㅍ", "ㅁ"},                        # 양순
    {"ㅈ", "ㅉ", "ㅊ"},                              # 치경구개
    {"ㅎ"},
]

# 모음 인접 그룹(단모음 위주)
VOWEL_ADJACENCY: list[set[str]] = [
    {"ㅏ", "ㅓ"},
    {"ㅓ", "ㅗ"},
    {"ㅗ", "ㅜ"},
    {"ㅡ", "ㅜ"},
    {"ㅡ", "ㅣ"},
    {"ㅔ", "ㅐ"},
    {"ㅔ", "ㅖ"},
    {"ㅐ", "ㅒ"},
    {"ㅏ", "ㅑ"},
    {"ㅓ", "ㅕ"},
    {"ㅗ", "ㅛ"},
    {"ㅜ", "ㅠ"},
]

DIPHTHONGS = {"ㅑ", "ㅒ", "ㅕ", "ㅖ", "ㅛ", "ㅠ", "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ"}

COMPOUND_CODAS = {"ㄳ", "ㄵ", "ㄶ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅄ"}

# 위치별 삭제(reference에서 누락) 비용
DEL_COST = {"onset": 1.0, "nucleus": 1.2, "coda": 0.5}

# 삽입(reference에 없는 자모가 hyp에 추가됨) 비용
INS_COST = 1.0

# 자음↔모음 치환 비용 (큰 페널티)
CV_MISMATCH = 1.5


def _consonant_substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    # 겹받침 단순화: 겹받침 치환은 일관된 1.0
    if a in COMPOUND_CODAS or b in COMPOUND_CODAS:
        return 1.0
    for family in CONSONANT_FAMILIES:
        if a in family and b in family:
            return 0.3
    for place in PLACE_GROUPS:
        if a in place and b in place:
            return 0.6
    return 1.0


def _vowel_substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    for adj in VOWEL_ADJACENCY:
        if a in adj and b in adj:
            return 0.3
    a_is_di = a in DIPHTHONGS
    b_is_di = b in DIPHTHONGS
    if a_is_di != b_is_di:
        return 0.6
    return 1.0


def jamo_distance(a: JamoToken | None, b: JamoToken | None) -> float:
    """두 자모 토큰 간 거리. None이면 삭제/삽입 비용을 반환."""
    if a is None and b is None:
        return 0.0
    if a is None:
        return INS_COST
    if b is None:
        return DEL_COST.get(a.pos, 1.0)
    # 위치 불일치는 별도 패널티 없이 자모 비교에만 의존(시퀀스 정렬이 위치 정합을 책임)
    a_c, b_c = a.char, b.char
    a_is_v = is_vowel(a_c)
    b_is_v = is_vowel(b_c)
    a_is_c = is_consonant(a_c)
    b_is_c = is_consonant(b_c)
    if a_is_v and b_is_v:
        return _vowel_substitution_cost(a_c, b_c)
    if a_is_c and b_is_c:
        return _consonant_substitution_cost(a_c, b_c)
    return CV_MISMATCH


def reference_weight(tok: JamoToken) -> float:
    """가중 PER 계산을 위한 reference 자모의 가중치.

    삭제 비용과 동일하게 두면 weighted_PER가 정렬 비용 합과 자연스럽게 연결된다.
    """
    return DEL_COST.get(tok.pos, 1.0)
