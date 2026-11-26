"""가시화가 읽는 두 스트림의 계약과 기록 억제 규칙 (2026-11-24).

가시화에 직접 보내지 않는다 — Redis 스트림을 만들기만 하고 가시화는 읽기만 한다.

orchestration_stream  (schema "orch/1")  비전 처리 오케스트레이션 선택 과정의 단계마다 한 건
    event    requested · cluster_checked · planned · applied · running · stopped
             (사이에 degraded · fallback · replanned · failed)
             applied = 띄웠다, running = 프로세스가 살아 있고 처리한 장이 있으며 입력 영상이 최근 것이다
    event_id     기록 한 건마다 고유 — 읽는 쪽은 이것으로 겹친 기록을 거른다
    request_id   런처에서 한 번 요청한 것
    execution_id 그 요청을 실제로 띄운 한 번 (다시 배치하면 새로) — 마지막 상태는 이것별로 다시 만든다
    mode     k3s | local          (정해지기 전이면 빈 칸)
    basis    live | saved         (자원 정보가 지금 잰 것인지, 기억해 둔 것인지)
    reason   사람이 읽는 한 줄     request · plan · detail  JSON
    code     이유 코드 (REASON_CODES, M11 에 더함) — 거절 · 실패 · 낮춤 · 대기에 붙는다. 읽는 쪽은 문구 대신 이것을 본다
             배치표의 refused · degraded · waiting 항목도 {item, code, reason} 이다
    ts       ISO 밀리초 +09:00     seq  이 프로세스에서 쓴 순번
    desired_version · converged   (선택, 감독자 — Docker 계획 §5) 원하는 상태 버전 · 맞음 "1" | 안 맞음 "0". 없으면 칸이 없다
    function (선택) 기능 이름 — 읽는 쪽이 기능 단위로 묶어 본다. 없으면 칸이 없다
    stage    (선택) device | server — 그 기록이 기능의 어느 단계인가. 없으면 칸이 없다
    quality  (선택) full | server_only | device_only | fallback — 두 단계 쌍이 깨졌을 때 지금 어떤 품질로 도는가.
             없으면 칸이 없다

node_resource_stream  (schema "node/1")  노드가 추가 · 제거될 때와 상태가 바뀔 때만 한 건
    node · role · source(k3s | local) · change(added · removed · ready · not_ready · baseline · service · usage · caps)
    caps  노드 특성(아키텍처 · 센서 · 가속기 · 온도 한계)이 바뀌었다 — 그 기억을 다루는 쪽이 한 건 남긴다
    service_mode   active(그 노드의 로봇 서비스가 실제로 돎 — 소스 영상이 온다) | idle | "" — 바뀌면 바로 남긴다(service)
    ready · cpu_pct · mem_pct · mem_mb · temp_c · gpu(JSON 목록) · baseline(JSON, added · baseline 일 때만) · ts
    잦거나 작은 변화는 남기지 않는다 — NodeChangeFilter.
"""
import json
import time
import uuid
from datetime import datetime, timedelta, timezone

ORCH_STREAM = "orchestration_stream"
NODE_STREAM = "node_resource_stream"
ORCH_SCHEMA = "orch/1"
NODE_SCHEMA = "node/1"
ORCH_EVENTS = ("requested", "cluster_checked", "planned", "applied", "running", "stopped",
               "degraded", "fallback", "replanned", "failed")
#: 이유 코드 — 한 곳에만 둔다. 새 이유는 여기에 한 줄 더하고, 모르는 코드는 reason_item · orch_record 가 거부한다
#: (문구 비교로 되돌아가지 않게). 사람이 읽는 문구(reason)는 따로 그대로 둔다.
#: 읽는 쪽(가시화)은 문구가 아니라 이 코드로 거절 · 실패 · 낮춤 · 대기의 까닭을 가른다 — 코드를 지우거나 뜻을 바꾸지 마라.
REASON_CODES = {
    "gpu_mem": "GPU 메모리 부족 (배치표 · 실행 직전 들이기 모두)",
    "gpu_throughput": "GPU 처리량 예산을 fps 하한까지 낮춰도 넘음",
    "gpu_exclusive": "단독 측정 실행 중이라 다른 GPU 작업을 들이지 않음",
    "gpu_unreadable": "nvidia-smi 를 못 읽어 들이지 않음",
    "gpu_preempted": "GPU 메모리가 모자라 우선순위가 더 높은 일에 자리를 내줌 (배치표에서 들이지 않음 · 적재된 단위를 멈춤)",
    "gpu_deferred": "GPU 메모리가 빌 때까지 미룸 — 메모리가 비면 우선순위순으로 다시 들인다 (요청 때 부르는 일 · 밀려난 단위)",
    "no_gpu_node": "쓸 수 있는 GPU 노드가 없음",
    "remote_gpu": "원격 GPU Executor 없음",
    "no_cost": "비용표에 그 모델 조합이 없음",
    "fps_lowered": "GPU 예산 때문에 fps 상한을 낮춤",
    "no_node": "CPU · 메모리 조건을 맞는 노드가 없음",
    "no_bandwidth": "대역폭이 모자람 (장당 전송이 장당 처리보다 김)",
    "unknown_bandwidth": "대역폭 아직 모름 (측정 전 · 실패 뒤 다시 재는 중)",
    "bandwidth_unsupported": "대역폭을 잴 수 없는 설치 (테일넷 주소 없음 등)",
    "no_cpu_capability": "k3s 노드는 읽히지만 CPU 작업을 올릴 수 없음 (클러스터 객체 없음 등)",
    "cpu_launch_failed": "CPU 작업을 띄우지 못함",
    "input_wait": "입력 영상 대기",
    "input_wait_unbudgeted": "입력 영상 대기 — 지금 예산으로는 미리 잡지 못함, 들어오면 다시 배치",
    "no_sensor": "그 작업이 필요한 센서가 노드에 없음",
    "arch_mismatch": "이미지 아키텍처가 노드와 다름 (arm64 · amd64)",
    "no_accel": "필요한 가속기(GPU · NPU)가 노드에 없음",
    "thermal_guard": "노드가 뜨겁거나 throttle 중이라 뺌",
    "deadline_miss": "전송 + 처리 시간이 작업 마감을 넘음",
    "pair_incomplete": "한 기능의 두 단계(기기 · 서버) 중 한쪽을 놓을 수 없어 기능을 거절",
    "preempted": "우선순위가 높은 일에 자리를 내주려 뺌 (한도 면제로 들어온 일이 노드를 넘치게 했다)",
    "manual_unavailable": "수동 배치로 고른 노드가 지금 없거나 그 작업을 받을 수 없음",
    "manual_excluded": "수동 배치에서 올리지 않기로 고름",
    # 작업 그래프 (work/1, 2026-11-27) — 배치표 work_warnings 의 코드. 거절이 아니라 알림이다(둘 다 돌린다)
    "dup_kind": "같은 일을 다른 모델로 두 번 함 — 한 소스에서 같은 종류의 결과(depth · 검출 …)를 다른 모델 둘이 낸다. "
                "하나로 맞추면 GPU 를 아낀다",
    "no_producer": "기능이 받을 결과물(예: 추적의 검출)을 내는 기능이 그 소스에 없음",
    "param_fallback": "기능 매개변수로 고른 입력 기능이 그 소스에 없어 기본 입력으로 바꿈",
}
MODES = ("k3s", "local")
BASES = ("live", "saved")
#: 한 기능의 단계 — 기기에서 도는 쪽과 서버에서 도는 쪽
STAGES = ("device", "server")
#: 두 단계 쌍이 깨졌을 때 지금 도는 품질 — full(둘 다) · server_only · device_only · fallback(대신 돌리는 것)
QUALITIES = ("full", "server_only", "device_only", "fallback")
NODE_CHANGES = ("added", "removed", "ready", "not_ready", "baseline", "service", "usage", "caps")
KST = timezone(timedelta(hours=9))


def iso_now(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")


def _j(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _num(v):
    return "" if v is None else ("%.1f" % v if isinstance(v, float) else str(v))


def check_code(code):
    """이유 코드가 REASON_CODES 에 있는지 — 없으면 ValueError"""
    if code not in REASON_CODES:
        raise ValueError("모르는 이유 코드 %r (가능: %s)" % (code, ", ".join(sorted(REASON_CODES))))
    return code


def reason_item(item, code, reason, **extra):
    """배치표 · 요청 기록의 거절 · 낮춤 · 대기 한 건 {item, code, reason, ...}"""
    return dict(extra, item=item, code=check_code(code), reason=str(reason))


def item_codes(items):
    """항목 목록의 코드 집합 (코드 없는 옛 항목은 건너뛴다)"""
    return {x.get("code") for x in items or [] if isinstance(x, dict) and x.get("code")}


def orch_record(event, request_id, mode=None, basis=None, reason="", request=None, plan=None, detail=None, t=None,
                execution_id=None, event_id=None, code=None, desired_version=None, converged=None,
                function=None, stage=None, quality=None):
    """orchestration_stream 한 건 (Redis 칸 = 글자). 틀린 값은 ValueError. code 는 REASON_CODES 중 하나(있을 때만 칸이 생김).
    function · stage · quality 도 값이 있을 때만 칸이 생긴다 — 칸 추가만이므로 schema 이름(orch/1)은 그대로다."""
    if code is not None:
        check_code(code)
    if event not in ORCH_EVENTS:
        raise ValueError("모르는 event %r (가능: %s)" % (event, ", ".join(ORCH_EVENTS)))
    if mode not in (None,) + MODES:
        raise ValueError("모르는 mode %r" % mode)
    if basis not in (None,) + BASES:
        raise ValueError("모르는 basis %r" % basis)
    if stage not in (None,) + STAGES:
        raise ValueError("모르는 stage %r (가능: %s)" % (stage, ", ".join(STAGES)))
    if quality not in (None,) + QUALITIES:
        raise ValueError("모르는 quality %r (가능: %s)" % (quality, ", ".join(QUALITIES)))
    if not request_id:
        raise ValueError("request_id 가 없다")
    rec = {"schema": ORCH_SCHEMA, "event_id": event_id or uuid.uuid4().hex, "event": event,
           "request_id": str(request_id), "execution_id": str(execution_id or ""), "mode": mode or "", "basis": basis or "",
           "reason": str(reason or ""), "ts": iso_now(t)}
    if code is not None:
        rec["code"] = code
    if request is not None:
        rec["request"] = _j(request)
    if plan is not None:
        rec["plan"] = _j(plan)
    if detail is not None:
        rec["detail"] = _j(detail)
    if desired_version is not None:
        rec["desired_version"] = str(int(desired_version))
    if converged is not None:
        rec["converged"] = "1" if converged else "0"
    if function is not None:
        rec["function"] = str(function)
    if stage is not None:
        rec["stage"] = stage
    if quality is not None:
        rec["quality"] = quality
    return rec


def node_record(node, change, sample, t=None):
    """node_resource_stream 한 건. sample: role · source · ready · cpu_pct · mem_pct · mem_mb · temp_c · gpu · baseline"""
    if change not in NODE_CHANGES:
        raise ValueError("모르는 change %r (가능: %s)" % (change, ", ".join(NODE_CHANGES)))
    s = sample or {}
    rec = {"schema": NODE_SCHEMA, "node": str(node), "change": change, "role": str(s.get("role") or ""),
           "source": str(s.get("source") or ""), "ready": "1" if s.get("ready") else "0",
           "cpu_pct": _num(s.get("cpu_pct")), "mem_pct": _num(s.get("mem_pct")), "mem_mb": _num(s.get("mem_mb")),
           "temp_c": _num(s.get("temp_c")), "gpu": _j(s.get("gpu") or []), "service_mode": str(s.get("service_mode") or ""),
           "ts": iso_now(t)}
    if change in ("added", "baseline") and s.get("baseline") is not None:
        rec["baseline"] = _j(s["baseline"])
    return rec


def latest_by_execution(entries):
    """읽는 쪽 도우미 — 스트림 항목(사전)들에서 event_id 로 겹친 것을 거르고 execution_id 별 마지막 기록을 낸다.
    소비자가 다시 시작해도 처음부터 다시 읽으면 같은 결과가 나온다."""
    seen, last = set(), {}
    for e in entries:
        eid = e.get("event_id")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        key = e.get("execution_id") or ("request:" + str(e.get("request_id")))
        last[key] = e
    return last


class NodeChangeFilter:
    """노드 표본을 받아 **남길 만한 변화만** node_resource_stream 기록으로 돌려준다.

    바로 남김   처음 본 노드(added) · 빠진 노드(removed) · Ready 변화 · 기본 자원 변화(baseline 해시)
    사용량      CPU · 메모리 · GPU 메모리를 band_pct(10%) 구간으로 본다. 지난 기록 구간의 경계에서
                hysteresis_pct(2%) 넘게 벗어나야 '바뀜' 이고, 그 구간이 confirm(2) 번 연속이어야 하며,
                같은 노드의 사용량 기록은 min_interval_s(30초) 넘게 떨어져야 한다.
    """

    def __init__(self, band_pct=10.0, hysteresis_pct=2.0, confirm=2, min_interval_s=30.0):
        self.band, self.hyst, self.confirm, self.min_gap = band_pct, hysteresis_pct, confirm, min_interval_s
        self.state = {}

    @staticmethod
    def _metrics(s):
        out = {"cpu": s.get("cpu_pct"), "mem": s.get("mem_pct")}
        for i, g in enumerate(s.get("gpu") or []):
            tot = g.get("mem_total_mb") or 0
            out["gpu%d" % i] = (100.0 * g.get("mem_used_mb", 0) / tot) if tot else None
        return out

    def _band(self, v):
        return None if v is None else int(min(max(v, 0.0), 100.0) // self.band)

    def _moved(self, recorded_band, v):
        if v is None or recorded_band is None:
            return (v is None) != (recorded_band is None)
        lo, hi = recorded_band * self.band, (recorded_band + 1) * self.band
        return v < lo - self.hyst or v >= hi + self.hyst

    def observe(self, node, sample, now=None):
        now = time.time() if now is None else now
        out = []
        st = self.state.get(node)
        bl = sample.get("baseline")
        bl_key = _j(bl) if bl is not None else None
        m = self._metrics(sample)
        bands = {k: self._band(v) for k, v in m.items()}
        if st is None:
            self.state[node] = {"ready": bool(sample.get("ready")), "baseline": bl_key, "bands": bands,
                                "pending": None, "count": 0, "last_usage": now, "mode": sample.get("service_mode")}
            return [node_record(node, "added", sample, now)]
        if bool(sample.get("ready")) != st["ready"]:
            st["ready"] = bool(sample.get("ready"))
            out.append(node_record(node, "ready" if st["ready"] else "not_ready", sample, now))
        if bl_key is not None and bl_key != st["baseline"]:
            st["baseline"] = bl_key
            out.append(node_record(node, "baseline", sample, now))
        if sample.get("service_mode") and sample.get("service_mode") != st.get("mode"):
            st["mode"] = sample["service_mode"]
            out.append(node_record(node, "service", sample, now))
        moved = any(self._moved(st["bands"].get(k), v) for k, v in m.items())
        if not moved:
            st["pending"], st["count"] = None, 0
            return out
        if bands == st["pending"]:
            st["count"] += 1
        else:
            st["pending"], st["count"] = bands, 1
        if st["count"] >= self.confirm and now - st["last_usage"] >= self.min_gap:
            st["bands"], st["pending"], st["count"], st["last_usage"] = bands, None, 0, now
            out.append(node_record(node, "usage", sample, now))
        return out

    def remove(self, node, sample=None, now=None):
        if self.state.pop(node, None) is None:
            return []
        return [node_record(node, "removed", dict(sample or {}, ready=False), now)]


class StreamWriter:
    """Redis 스트림에 쓴다. 못 써도 예외를 올리지 않는다(오케스트레이션이 기록 때문에 멈추지 않게)."""

    def __init__(self, redis_client, maxlen=1000):
        self.r, self.maxlen = redis_client, maxlen
        self.seq, self.fail, self.err = 0, 0, None

    def add(self, stream, rec):
        self.seq += 1
        fields = dict(rec, seq=str(self.seq))
        try:
            self.r.xadd(stream, fields, maxlen=self.maxlen, approximate=True)
            return True
        except Exception as e:   # noqa: BLE001 — 기록 실패는 세기만 한다
            self.fail += 1
            self.err = "%s: %s" % (type(e).__name__, e)
            return False

    def orch(self, *a, **kw):
        return self.add(ORCH_STREAM, orch_record(*a, **kw))

    def nodes(self, records):
        return [self.add(NODE_STREAM, r) for r in records]
