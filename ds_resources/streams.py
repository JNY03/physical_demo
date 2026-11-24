"""가시화가 읽는 두 스트림의 계약과 기록 억제 규칙 (2026-11-24).

가시화에 직접 보내지 않는다 — Redis 스트림을 만들기만 하고 가시화는 읽기만 한다.

orchestration_stream  (schema "orch/1")  비전 처리 오케스트레이션 선택 과정의 단계마다 한 건
    event    requested · cluster_checked · planned · applied · degraded · fallback · replanned · failed
    run_id   한 번의 선택 과정(런처에서 실행을 누른 한 번)을 묶는 ID
    mode     k3s | local          (정해지기 전이면 빈 칸)
    basis    live | saved         (자원 정보가 지금 잰 것인지, 기억해 둔 것인지)
    reason   사람이 읽는 한 줄     request · plan · detail  JSON
    ts       ISO 밀리초 +09:00     seq  이 프로세스에서 쓴 순번

node_resource_stream  (schema "node/1")  노드가 추가 · 제거될 때와 상태가 바뀔 때만 한 건
    node · role · source(k3s | local) · change(added · removed · ready · not_ready · baseline · usage)
    ready · cpu_pct · mem_pct · mem_mb · temp_c · gpu(JSON 목록) · baseline(JSON, added · baseline 일 때만) · ts
    잦거나 작은 변화는 남기지 않는다 — NodeChangeFilter.
"""
import json
import time
from datetime import datetime, timedelta, timezone

ORCH_STREAM = "orchestration_stream"
NODE_STREAM = "node_resource_stream"
ORCH_SCHEMA = "orch/1"
NODE_SCHEMA = "node/1"
ORCH_EVENTS = ("requested", "cluster_checked", "planned", "applied", "degraded", "fallback", "replanned", "failed")
MODES = ("k3s", "local")
BASES = ("live", "saved")
NODE_CHANGES = ("added", "removed", "ready", "not_ready", "baseline", "usage")
KST = timezone(timedelta(hours=9))


def iso_now(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")


def _j(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _num(v):
    return "" if v is None else ("%.1f" % v if isinstance(v, float) else str(v))


def orch_record(event, run_id, mode=None, basis=None, reason="", request=None, plan=None, detail=None, t=None):
    """orchestration_stream 한 건 (Redis 칸 = 글자). 틀린 값은 ValueError."""
    if event not in ORCH_EVENTS:
        raise ValueError("모르는 event %r (가능: %s)" % (event, ", ".join(ORCH_EVENTS)))
    if mode not in (None,) + MODES:
        raise ValueError("모르는 mode %r" % mode)
    if basis not in (None,) + BASES:
        raise ValueError("모르는 basis %r" % basis)
    if not run_id:
        raise ValueError("run_id 가 없다")
    rec = {"schema": ORCH_SCHEMA, "event": event, "run_id": str(run_id), "mode": mode or "", "basis": basis or "",
           "reason": str(reason or ""), "ts": iso_now(t)}
    if request is not None:
        rec["request"] = _j(request)
    if plan is not None:
        rec["plan"] = _j(plan)
    if detail is not None:
        rec["detail"] = _j(detail)
    return rec


def node_record(node, change, sample, t=None):
    """node_resource_stream 한 건. sample: role · source · ready · cpu_pct · mem_pct · mem_mb · temp_c · gpu · baseline"""
    if change not in NODE_CHANGES:
        raise ValueError("모르는 change %r (가능: %s)" % (change, ", ".join(NODE_CHANGES)))
    s = sample or {}
    rec = {"schema": NODE_SCHEMA, "node": str(node), "change": change, "role": str(s.get("role") or ""),
           "source": str(s.get("source") or ""), "ready": "1" if s.get("ready") else "0",
           "cpu_pct": _num(s.get("cpu_pct")), "mem_pct": _num(s.get("mem_pct")), "mem_mb": _num(s.get("mem_mb")),
           "temp_c": _num(s.get("temp_c")), "gpu": _j(s.get("gpu") or []), "ts": iso_now(t)}
    if change in ("added", "baseline") and s.get("baseline") is not None:
        rec["baseline"] = _j(s["baseline"])
    return rec


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
                                "pending": None, "count": 0, "last_usage": now}
            return [node_record(node, "added", sample, now)]
        if bool(sample.get("ready")) != st["ready"]:
            st["ready"] = bool(sample.get("ready"))
            out.append(node_record(node, "ready" if st["ready"] else "not_ready", sample, now))
        if bl_key is not None and bl_key != st["baseline"]:
            st["baseline"] = bl_key
            out.append(node_record(node, "baseline", sample, now))
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
