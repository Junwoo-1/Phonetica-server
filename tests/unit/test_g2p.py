import pytest

from voice_pron.g2p.pronouncer import StandardPronouncer


@pytest.fixture(scope="module")
def pron():
    return StandardPronouncer()


def test_normalize_strips_non_hangul():
    assert StandardPronouncer.normalize("hello 안녕 123!") == "안녕"
    assert StandardPronouncer.normalize("가-나 다") == "가 나 다"
    assert StandardPronouncer.normalize("") == ""


def test_pronunciation_basic_palatalization(pron):
    # 구개음화: "굳이" → "구지"
    assert pron.to_pronunciation("굳이") == "구지"


def test_pronunciation_palatalization_chi(pron):
    # 구개음화: "같이" → "가치"
    assert pron.to_pronunciation("같이") == "가치"


def test_pronunciation_liaison(pron):
    # 연음: "꽃이" → "꼬치"
    assert pron.to_pronunciation("꽃이") == "꼬치"


def test_pronunciation_keeps_simple(pron):
    # 변화 없는 발음
    assert pron.to_pronunciation("안녕") == "안녕"


def test_jamo_sequence_simple(pron):
    tokens = pron.to_jamo_sequence("가")
    chars = [t.char for t in tokens]
    assert chars == ["ㄱ", "ㅏ"]


def test_jamo_sequence_with_palatalization(pron):
    # "굳이" → "구지" → ㄱ ㅜ ㅈ ㅣ
    tokens = pron.to_jamo_sequence("굳이")
    chars = [t.char for t in tokens]
    assert chars == ["ㄱ", "ㅜ", "ㅈ", "ㅣ"]


def test_jamo_sequence_strips_english(pron):
    tokens = pron.to_jamo_sequence("hi 안")
    chars = [t.char for t in tokens]
    # 'hi'는 정규화에서 제거. '안' → ㅇ ㅏ ㄴ
    assert chars == ["ㅇ", "ㅏ", "ㄴ"]


def test_empty_input(pron):
    assert pron.to_pronunciation("") == ""
    assert pron.to_jamo_sequence("") == []
    assert pron.to_jamo_sequence("123 abc") == []
