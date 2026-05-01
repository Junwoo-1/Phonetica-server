"""SSE 이벤트 envelope 빌더."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import orjson


def event(name: str, data: dict[str, Any], request_id: str) -> dict[str, str]:
    """sse-starlette EventSourceResponse가 요구하는 dict 형식으로 빌드한다.

    sse-starlette은 `{"event": str, "data": str}`를 받으면 SSE 프로토콜에 맞게 방출한다.
    data는 JSON 직렬화된 문자열(envelope 포함)을 넣는다.
    """
    payload = {
        "event": name,
        "data": data,
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
    }
    return {"event": name, "data": orjson.dumps(payload).decode()}
