from voice_pron.g2p.jamo_utils import (
    JamoToken,
    decompose_hangul,
    is_consonant,
    is_vowel,
    syllabify_flat_jamo,
    syllabify_with_spaces,
)


def test_decompose_simple_syllable():
    tokens = decompose_hangul("가")
    assert tokens == [
        JamoToken("ㄱ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
    ]


def test_decompose_with_coda():
    tokens = decompose_hangul("강")
    assert tokens == [
        JamoToken("ㄱ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
        JamoToken("ㅇ", "coda", 0),
    ]


def test_decompose_compound_coda():
    # "값" → ㄱ, ㅏ, ㅄ (compound coda preserved as single compat jamo)
    tokens = decompose_hangul("값")
    assert tokens == [
        JamoToken("ㄱ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
        JamoToken("ㅄ", "coda", 0),
    ]


def test_decompose_zero_onset_marker():
    # "아" → ㅇ(onset), ㅏ
    tokens = decompose_hangul("아")
    assert tokens == [
        JamoToken("ㅇ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
    ]


def test_decompose_skips_non_hangul():
    tokens = decompose_hangul("a 가1 나")
    syls = sorted({t.syl for t in tokens})
    assert syls == [0, 1]
    chars = [t.char for t in tokens]
    assert chars == ["ㄱ", "ㅏ", "ㄴ", "ㅏ"]


def test_decompose_multi_syllable():
    tokens = decompose_hangul("안녕")
    assert tokens == [
        JamoToken("ㅇ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
        JamoToken("ㄴ", "coda", 0),
        JamoToken("ㄴ", "onset", 1),
        JamoToken("ㅕ", "nucleus", 1),
        JamoToken("ㅇ", "coda", 1),
    ]


def test_consonant_vowel_sets():
    assert is_consonant("ㄱ")
    assert is_consonant("ㅄ")
    assert not is_consonant("ㅏ")
    assert is_vowel("ㅏ")
    assert is_vowel("ㅢ")
    assert not is_vowel("ㄱ")


def test_syllabify_flat_simple():
    # "가" 발음 → ㄱ ㅏ
    tokens = syllabify_flat_jamo(["ㄱ", "ㅏ"])
    assert tokens == [
        JamoToken("ㄱ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
    ]


def test_syllabify_flat_cvc():
    # "강" 발음 → ㄱ ㅏ ㅇ
    tokens = syllabify_flat_jamo(["ㄱ", "ㅏ", "ㅇ"])
    assert tokens == [
        JamoToken("ㄱ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
        JamoToken("ㅇ", "coda", 0),
    ]


def test_syllabify_flat_two_syllables():
    # "안녕" → ㅇ ㅏ ㄴ ㄴ ㅕ ㅇ
    tokens = syllabify_flat_jamo(["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"])
    assert tokens == [
        JamoToken("ㅇ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
        JamoToken("ㄴ", "coda", 0),
        JamoToken("ㄴ", "onset", 1),
        JamoToken("ㅕ", "nucleus", 1),
        JamoToken("ㅇ", "coda", 1),
    ]


def test_syllabify_flat_open_then_consonant():
    # "가나" → ㄱ ㅏ ㄴ ㅏ (no coda)
    tokens = syllabify_flat_jamo(["ㄱ", "ㅏ", "ㄴ", "ㅏ"])
    assert tokens == [
        JamoToken("ㄱ", "onset", 0),
        JamoToken("ㅏ", "nucleus", 0),
        JamoToken("ㄴ", "onset", 1),
        JamoToken("ㅏ", "nucleus", 1),
    ]


def test_syllabify_with_spaces():
    # "안녕 하세요" 자모 분해 후 syllabify
    seq = ["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ", " ", "ㅎ", "ㅏ", "ㅅ", "ㅔ", "ㅇ", "ㅛ"]
    tokens = syllabify_with_spaces(seq)
    syls = [t.syl for t in tokens]
    # 5 syllables: 안 녕 하 세 요
    assert max(syls) == 4
    chars = [t.char for t in tokens]
    assert "ㅎ" in chars and "ㅔ" in chars
