"""정렬 결과로부터 자모/위치별 정확도와 종합 점수를 산출한다."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from voice_pron.align.cost import reference_weight
from voice_pron.align.needleman import AlignmentResult
from voice_pron.g2p.jamo_utils import JamoToken

Position = Literal["onset", "nucleus", "coda"]


@dataclass
class PositionStat:
    matched: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 1.0


@dataclass
class JamoStat:
    ref_count: int = 0
    correct: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def add_error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1


@dataclass
class ScoreReport:
    overall_score: float
    per: float
    weighted_per: float
    counts: dict[str, int]
    per_position: dict[Position, PositionStat]
    per_jamo: dict[str, JamoStat]
    problem_jamos: list[str]
    detailed_jamos: list[dict]


def compute_scores(alignment: AlignmentResult, ref: list[JamoToken]) -> ScoreReport:
    counts = {"match": 0, "sub": 0, "ins": 0, "del": 0}
    per_position: dict[Position, PositionStat] = {
        "onset": PositionStat(),
        "nucleus": PositionStat(),
        "coda": PositionStat(),
    }
    per_jamo: dict[str, JamoStat] = defaultdict(JamoStat)
    detailed_jamos: list[dict] = []

    n_ref = len(ref)
    total_weight = sum(reference_weight(t) for t in ref) or 1.0

    for pair in alignment.pairs:
        op = pair.op
        counts[op] += 1
        r = pair.ref
        if r is not None:
            per_position[r.pos].total += 1
            stat = per_jamo[f"{r.char}@{r.pos}"]
            stat.ref_count += 1
            if op == "match":
                per_position[r.pos].matched += 1
                stat.correct += 1
                j_score = 100.0
            elif op == "sub":
                hyp_char = pair.hyp.char if pair.hyp else "?"
                stat.add_error(f"sub_to_{hyp_char}")
                ref_wt = reference_weight(r) or 1.0
                j_score = 100.0 * max(0.0, 1.0 - (pair.cost / ref_wt))
            elif op == "del":
                stat.add_error("del")
                j_score = 0.0
            else:
                j_score = 100.0

            detailed_jamos.append({
                "syl": r.syl,
                "pos": r.pos,
                "char": r.char,
                "score": round(j_score, 2),
            })

    per = (
        (counts["sub"] + counts["del"] + counts["ins"]) / n_ref
        if n_ref
        else 0.0
    )
    weighted_per = alignment.total_cost / total_weight
    overall_score = 100.0 * max(0.0, 1.0 - weighted_per)

    # problem jamos: ref_count >= 2 이면서 정확도가 낮은 자모 top 5
    candidates = [
        (key, stat)
        for key, stat in per_jamo.items()
        if stat.ref_count >= 2
    ]
    candidates.sort(key=lambda kv: kv[1].correct / kv[1].ref_count)
    problem_jamos = [k for k, _ in candidates[:5] if (
        per_jamo[k].correct / per_jamo[k].ref_count
    ) < 1.0]

    return ScoreReport(
        overall_score=round(overall_score, 2),
        per=round(per, 4),
        weighted_per=round(weighted_per, 4),
        counts=counts,
        per_position=per_position,
        per_jamo=dict(per_jamo),
        problem_jamos=problem_jamos,
        detailed_jamos=detailed_jamos,
    )
