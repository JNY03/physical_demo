# -*- coding: utf-8 -*-
"""드론 엣지를 통합 제어 버스에 올리는 어댑터 노드.

엣지(server.py)는 말단과 HTTP 2계층으로 말한다 — 엣지가 프레임을 당기고 판정을
되민다. 그 구조는 그대로 두고, **관제가 쓰는 명령 규약만 로봇과 같게** 맞춘다.
그래야 한 화면에서 로봇과 드론을 같은 어휘로 제어할 수 있다.

    관제 ──terminal/{id}/downlink──▶ 이 노드 ──HTTP /api/command──▶ edge/server.py
         ◀──terminal/{id}/uplink───            ◀──HTTP /api/health──

**규약을 다시 구현하지 않는다.** hw/pi/common 의 BaseNode 를 그대로 상속한다 —
토픽·하트비트·스풀·명령 4단계(ACK→수행중→물리변화→완료)·OTel 이 전부 따라온다.
같은 규약을 두 곳에 구현하면 반드시 어긋나고, 이 저장소는 그 사고를 이미 한 번
겪었다(common/schema.py 의 background_classes 주석).

엣지 서버와는 **localhost HTTP** 로만 말한다. 그래서 이 노드는 사이드카로 띄울 수
있고, k3s 에서도 같은 파드 안의 컨테이너 둘로 나뉜다.

실행:
    HW_ENTITY_ID=edge1 HW_BROKER_HOST=<브로커 주소> python3 mqtt_node.py
    EDGE_API=http://127.0.0.1:8891 도 환경변수로 받는다.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# 공유 규약 패키지. 지금은 hw/pi 아래 있어 경로를 붙여 쓴다 — 배치 정리 단계에서
# 저장소 공용 위치로 옮길 예정이고, 그때 이 세 줄이 사라진다.
_HW_PI = Path(__file__).resolve().parent.parent / "hw" / "pi"
if str(_HW_PI) not in sys.path:
    sys.path.insert(0, str(_HW_PI))

from common.base_actions import BASE_ACTIONS          # noqa: E402
from common.node import BaseNode                      # noqa: E402
from common.physical_command import CommandError      # noqa: E402

EDGE_API = os.environ.get("EDGE_API", "http://127.0.0.1:8891")
HTTP_TIMEOUT = float(os.environ.get("EDGE_API_TIMEOUT", "5"))


def _api(path, payload=None):
    """엣지 서버의 HTTP 창구. payload 가 있으면 POST, 없으면 GET."""
    url = f"{EDGE_API}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            body = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise CommandError("FAILED_PRECONDITION",
                           f"edge_http_{e.code}: {e.reason}") from e
    except Exception as e:
        # 엣지 서버가 안 떠 있는 것은 설정 오류가 아니라 전제 미충족이다.
        raise CommandError("UNAVAILABLE", f"edge_unreachable: {e}") from e
    try:
        return json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        return {"raw": body[:400]}


def _cmd(name):
    """엣지의 /api/command 를 거는 핸들러를 만든다. (stage, detail) 제너레이터."""
    def handler(node, params):
        yield "executing", None
        res = _api("/api/command", {"command": name})
        yield "completed", {k: v for k, v in (res.get("gates") or {}).items()}
    return handler


def _act_snapshot(node, params):
    """말단에서 한 장 당겨 온다. '지금 뭐가 보이나'를 묻는 경로."""
    yield "executing", None
    res = _api("/api/command", {"command": "fetch_image"})
    ok = bool(res.get("ok", True))
    if not ok:
        raise CommandError("UNAVAILABLE", str(res.get("error", "fetch_failed"))[:200])
    yield "completed", {"ok": 1.0 if ok else 0.0}


class EdgeNode(BaseNode):
    """드론 엣지 — 인지 파이프라인의 게이트를 중립 어휘로 노출한다.

    어휘를 로봇과 **같은 모양**으로 맞췄다. 로봇의 drive_start 가 "구동을 쓸 수
    있게 한다"이듯, 여기서는 perceive/ingest/publish 가 각각 "추론·가져오기·
    보내기를 쓸 수 있게 한다"이다. 기종도 모델도 이름에 넣지 않는다 — YOLOE 든
    색 분할이든 관제는 그것을 몰라야 한다(edge/config.json 의 ovd.provider).
    """

    ENTITY_TYPE = "edge"

    ACTIONS = dict(BASE_ACTIONS, **{
        # 인지 — 당겨 온 프레임을 판정할지
        "perceive_start": _cmd("start_inference"),
        "perceive_stop": _cmd("stop_inference"),
        # 가져오기 — 말단에서 프레임을 계속 당길지
        "ingest_start": _cmd("start_pull"),
        "ingest_stop": _cmd("stop_pull"),
        # 보내기 — 판정을 말단에 반영할지
        "publish_start": _cmd("start_push"),
        "publish_stop": _cmd("stop_push"),
        # 한 장만
        "snapshot": _act_snapshot,
    })

    # 말단의 물리 상태를 바꾸는 것은 **보내기뿐**이다. 추론과 가져오기는 엣지
    # 안에서만 일어나고 말단 레코드를 건드리지 않는다. 물리 동작 분류를 넓게
    # 잡으면 승인 절차가 의미를 잃는다.
    PHYSICAL_ACTIONS = frozenset({"publish_start"})

    # 엣지 상태는 **따로 도는 스레드가 떠 온다.** on_sample 은 메인 루프에서
    # 불리는데(common/node.py run), 거기서 HTTP 를 직접 때리면 그 시간만큼
    # 루프가 멈춘다. SAMPLE_INTERVAL 기본값이 1초인데 타임아웃이 5초라, 느린
    # 응답 하나에 하트비트가 밀리고 브로커가 keepalive 초과로 끊어 버린다
    # (2026-10-23 실측: rc=141, 미전송 100건). 폴러는 자기 주기로 돌고
    # on_sample 은 **마지막으로 본 값을 그대로 싣기만** 한다.
    POLL_INTERVAL = float(os.environ.get("EDGE_POLL_INTERVAL", "2.0"))

    def __init__(self):
        super().__init__()
        self._snap = {"edge": "starting"}
        self._snap_at = 0.0
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    def _poll_loop(self):
        while True:
            try:
                st = _api("/api/command", {"command": "terminal_status"})
                c = st.get("counters") or {}
                snap = {"gates": st.get("gates"), "terminal": st.get("terminal"),
                        "pulled": c.get("pulled"), "inferred": c.get("inferred"),
                        "pushed": c.get("pushed"), "failed": c.get("failed"),
                        "dropped": c.get("inference_dropped")}
            except CommandError as e:
                snap = {"edge": "unreachable", "reason": str(e)[:120]}
            self._snap, self._snap_at = snap, time.time()
            time.sleep(self.POLL_INTERVAL)

    def device_status_extra(self):
        """등록·상태 메시지에 싣는 엣지 고유 정보. 캐시만 본다."""
        return dict(self._snap)

    def on_sample(self, now):
        """주기 표본 — 마지막으로 떠 온 상태를 uplink 로 올린다.

        나이를 숨기지 않는다 — 폴러가 멈추면 age_s 가 계속 커지고, 그게 보이는
        것이 '마지막 값이 영원히 신선해 보이는 것'보다 낫다.
        """
        self.publish(f"{self.base}/uplink", {
            "kind": "edge_state",
            **self._snap,
            "age_s": round(now - self._snap_at, 1) if self._snap_at else None,
            "at": round(now, 3),
        })


if __name__ == "__main__":
    EdgeNode().run()
