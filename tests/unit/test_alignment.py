from voice_pron.align.cost import jamo_distance
from voice_pron.align.needleman import align
from voice_pron.g2p.jamo_utils import JamoToken, decompose_hangul


def _seq(text):
    return decompose_hangul(text)


def test_perfect_match():
    ref = _seq("안녕")
    hyp = _seq("안녕")
    res = align(ref, hyp, jamo_distance)
    assert res.total_cost == 0.0
    assert all(p.op == "match" for p in res.pairs)
    assert len(res.pairs) == len(ref)


def test_substitution_close_consonant():
    # 안녕 → 칸녕 (ㅇ→ㅋ, 다른 그룹이지만 자음 sub, 거리 ~1.0)
    ref = _seq("안녕")
    hyp = _seq("칸녕")
    res = align(ref, hyp, jamo_distance)
    sub_pairs = [p for p in res.pairs if p.op == "sub"]
    assert len(sub_pairs) == 1
    assert sub_pairs[0].ref.char == "ㅇ"
    assert sub_pairs[0].hyp.char == "ㅋ"


def test_substitution_same_family_low_cost():
    # 가 → 까 (ㄱ→ㄲ, 같은 평/경/격음 그룹)
    ref = _seq("가")
    hyp = _seq("까")
    res = align(ref, hyp, jamo_distance)
    assert res.total_cost == 0.3


def test_deletion_of_coda_cheap():
    # 강 → 가 (종성 ㅇ 누락, del coda = 0.5)
    ref = _seq("강")
    hyp = _seq("가")
    res = align(ref, hyp, jamo_distance)
    dels = [p for p in res.pairs if p.op == "del"]
    assert len(dels) == 1
    assert dels[0].ref.char == "ㅇ"
    assert dels[0].ref.pos == "coda"
    assert res.total_cost == 0.5


def test_insertion_extra_jamo():
    # 가 → 강 (hyp에 잉여 ㅇ 추가, ins=1.0)
    ref = _seq("가")
    hyp = _seq("강")
    res = align(ref, hyp, jamo_distance)
    ins = [p for p in res.pairs if p.op == "ins"]
    assert len(ins) == 1
    assert ins[0].hyp.char == "ㅇ"
    assert res.total_cost == 1.0


def test_complex_alignment():
    # 사과 → 타과 (ㅅ→ㅌ, 다른 자음, 거리 1.0)
    ref = _seq("사과")
    hyp = _seq("타과")
    res = align(ref, hyp, jamo_distance)
    subs = [p for p in res.pairs if p.op == "sub"]
    assert any(p.ref.char == "ㅅ" and p.hyp.char == "ㅌ" for p in subs)


def test_empty_ref_all_inserts():
    res = align([], _seq("가"), jamo_distance)
    assert all(p.op == "ins" for p in res.pairs)
    assert res.total_cost == 2.0  # ㄱ + ㅏ 각 1.0


def test_empty_hyp_all_deletes():
    res = align(_seq("강"), [], jamo_distance)
    assert all(p.op == "del" for p in res.pairs)
    # ㄱ(onset 1.0) + ㅏ(nucleus 1.2) + ㅇ(coda 0.5) = 2.7
    assert abs(res.total_cost - 2.7) < 1e-9


def test_alignment_preserves_order():
    ref = _seq("안녕하세요")
    hyp = _seq("안녕하세요")
    res = align(ref, hyp, jamo_distance)
    # ref 자모 순서가 보존되어야 함
    ref_chars_from_pairs = [p.ref.char for p in res.pairs if p.ref is not None]
    assert ref_chars_from_pairs == [t.char for t in ref]
