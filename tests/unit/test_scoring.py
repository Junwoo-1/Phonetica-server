from voice_pron.align.cost import jamo_distance
from voice_pron.align.needleman import align
from voice_pron.g2p.jamo_utils import decompose_hangul
from voice_pron.scoring.score import compute_scores


def test_perfect_score():
    ref = decompose_hangul("안녕하세요")
    res = align(ref, ref, jamo_distance)
    report = compute_scores(res, ref)
    assert report.overall_score == 100.0
    assert report.per == 0.0
    assert report.counts["match"] == len(ref)
    assert report.counts["sub"] == 0
    assert report.problem_jamos == []


def test_partial_score_with_coda_drop():
    # "강" → "가" (종성 ㅇ 누락)
    ref = decompose_hangul("강")
    hyp = decompose_hangul("가")
    res = align(ref, hyp, jamo_distance)
    report = compute_scores(res, ref)
    # total_cost=0.5, total_weight=1.0+1.2+0.5=2.7 → weighted_per ≈ 0.185 → score ≈ 81.5
    assert report.counts["del"] == 1
    assert report.per_position["coda"].matched == 0
    assert report.per_position["coda"].total == 1
    assert report.per_position["onset"].accuracy == 1.0
    assert report.overall_score < 100.0
    assert report.overall_score > 70.0


def test_problem_jamo_detection():
    # "다다" → "타타" : ㄷ 두 번 모두 ㅌ로 치환
    ref = decompose_hangul("다다")
    hyp = decompose_hangul("타타")
    res = align(ref, hyp, jamo_distance)
    report = compute_scores(res, ref)
    assert "ㄷ@onset" in report.problem_jamos
    assert report.per_jamo["ㄷ@onset"].ref_count == 2
    assert report.per_jamo["ㄷ@onset"].correct == 0


def test_per_position_accuracy():
    # "사과" → "사가" : 두 번째 음절의 onset이 ㄱ→ㄱ, nucleus가 ㅘ→ㅏ
    ref = decompose_hangul("사과")
    hyp = decompose_hangul("사가")
    res = align(ref, hyp, jamo_distance)
    report = compute_scores(res, ref)
    # onset 모두 일치 (ㅅ, ㄱ)
    assert report.per_position["onset"].accuracy == 1.0
    # nucleus 1/2 (ㅏ 일치, ㅘ→ㅏ 치환)
    assert report.per_position["nucleus"].matched == 1
    assert report.per_position["nucleus"].total == 2


def test_score_clamps_to_zero():
    # 완전히 다른 시퀀스
    ref = decompose_hangul("가")
    hyp = decompose_hangul("뜨드으응")
    res = align(ref, hyp, jamo_distance)
    report = compute_scores(res, ref)
    assert report.overall_score >= 0.0
