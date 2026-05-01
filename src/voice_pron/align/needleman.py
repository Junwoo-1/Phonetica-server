"""가중 Needleman-Wunsch 글로벌 정렬 알고리즘."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from voice_pron.g2p.jamo_utils import JamoToken

Op = Literal["match", "sub", "ins", "del"]


@dataclass
class AlignedPair:
    ref: JamoToken | None
    hyp: JamoToken | None
    op: Op
    cost: float


@dataclass
class AlignmentResult:
    pairs: list[AlignedPair]
    total_cost: float


CostFn = Callable[[JamoToken | None, JamoToken | None], float]


def align(
    ref: list[JamoToken],
    hyp: list[JamoToken],
    cost_fn: CostFn,
) -> AlignmentResult:
    """ref와 hyp 자모 시퀀스를 가중 NW로 글로벌 정렬한다."""
    m = len(ref)
    n = len(hyp)

    # DP 테이블 + 백트랙용 포인터
    INF = float("inf")
    dp: list[list[float]] = [[INF] * (n + 1) for _ in range(m + 1)]
    bt: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    # bt 코드: 0=시작, 1=대각선(match/sub), 2=위→아래(del, ref소비), 3=왼→오른쪽(ins, hyp소비)

    dp[0][0] = 0.0
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + cost_fn(ref[i - 1], None)
        bt[i][0] = 2
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + cost_fn(None, hyp[j - 1])
        bt[0][j] = 3

    for i in range(1, m + 1):
        r = ref[i - 1]
        for j in range(1, n + 1):
            h = hyp[j - 1]
            diag = dp[i - 1][j - 1] + cost_fn(r, h)
            up = dp[i - 1][j] + cost_fn(r, None)
            left = dp[i][j - 1] + cost_fn(None, h)
            best = diag
            move = 1
            if up < best:
                best = up
                move = 2
            if left < best:
                best = left
                move = 3
            dp[i][j] = best
            bt[i][j] = move

    # 백트랙
    pairs: list[AlignedPair] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == 1:
            r = ref[i - 1]
            h = hyp[j - 1]
            c = cost_fn(r, h)
            op: Op = "match" if r.char == h.char else "sub"
            pairs.append(AlignedPair(ref=r, hyp=h, op=op, cost=c))
            i -= 1
            j -= 1
        elif move == 2:
            r = ref[i - 1]
            c = cost_fn(r, None)
            pairs.append(AlignedPair(ref=r, hyp=None, op="del", cost=c))
            i -= 1
        elif move == 3:
            h = hyp[j - 1]
            c = cost_fn(None, h)
            pairs.append(AlignedPair(ref=None, hyp=h, op="ins", cost=c))
            j -= 1
        else:
            break

    pairs.reverse()
    return AlignmentResult(pairs=pairs, total_cost=dp[m][n])
