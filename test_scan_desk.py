"""스캔 탐지 창구(scan_desk.py) 상태 기계 시험 — 가짜 Redis 로, 비전 · 서버 없이.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 venv/bin/python -m pytest test_scan_desk.py -q
"""
import json
import threading
import time

import pytest

import scan_desk as sd


class FakeRedis:
    """scan_desk 가 쓰는 만큼만: get/set/delete/expire/incr/time · 스트림 xadd/xrevrange/xrange/xread."""

    def __init__(self):
        self.kv, self.streams, self.cv = {}, {}, threading.Condition()
        self.t_ms = int(time.time() * 1000)
        self.seq = 0

    def time(self):
        ms = int(time.time() * 1000)
        return ms // 1000, (ms % 1000) * 1000

    def get(self, k):
        v = self.kv.get(k)
        return None if v is None else (v if isinstance(v, bytes) else str(v).encode())

    def set(self, k, v, ex=None):
        self.kv[k] = v

    def delete(self, *ks):
        for k in ks:
            self.kv.pop(k, None)

    def expire(self, k, s):
        return k in self.kv

    def incr(self, k):
        self.kv[k] = int(self.kv.get(k, 0)) + 1
        return self.kv[k]

    def xadd(self, key, fields, ms=None):
        with self.cv:
            ms = int(time.time() * 1000) if ms is None else ms
            self.seq += 1
            eid = ("%d-%d" % (ms, self.seq)).encode()
            self.streams.setdefault(key, []).append((eid, {k.encode(): v for k, v in fields.items()}))
            self.cv.notify_all()
            return eid

    @staticmethod
    def _k(eid):
        a, b = (eid.decode() if isinstance(eid, bytes) else eid).split("-")
        return int(a), int(b)

    def xrevrange(self, key, mx, mn, count=None):
        return list(reversed(self.streams.get(key, [])))[:count]

    def xrange(self, key, a, b):
        return [e for e in self.streams.get(key, []) if self._k(a) <= self._k(e[0]) <= self._k(b)]

    def xread(self, ids, count=None, block=None):
        deadline = time.time() + (block or 0) / 1000.0
        with self.cv:
            while True:
                out = []
                for key, last in ids.items():
                    new = [e for e in self.streams.get(key, []) if self._k(e[0]) > self._k(last)][:count]
                    if new:
                        out.append((key.encode(), new))
                if out or time.time() >= deadline:
                    return out
                self.cv.wait(deadline - time.time())


def door(score=72.0, found=True, x=300.0):
    return {"label": "demo_door", "bbox": [x - 50, 50, x + 50, 350],
            "dist": {"value": 2.35, "space": "metric", "source": "measured", "age": 0},
            "bearing": {"offset_deg": 5.0, "f_px": 300.0, "f_source": "go1_box"},
            "ground_distance": {"distance_cm": 235.0, "reason": None},
            "assess": {"frame": {"score": score, "suspect": "demo_door" if found else None, "gates_failed": []},
                       "axes": {"global": found, "local": None, "local_declared": False, "geometry": None},
                       "cue_values": {"class": {"pct": 80.0, "role": "score", "passed": None, "w": 0.5}},
                       "min_score": 50.0, "enough": True}}


class FakeVision:
    """비전 대역 — 판 키를 보고 ack 를 쓰고, 결과를 profile.scan · mark 를 달아 낸다."""

    def __init__(self, r, src="robot1", version=12, dets=None, ack_error=None):
        self.r, self.src, self.version, self.ack_error = r, src, version, ack_error
        self.dets = dets or (lambda mark: [door()])
        self.n = 0

    def frame(self, mark=None, image=None, version=None):
        """원본 한 장 + 그 장의 targeted 결과 한 장."""
        spec = json.loads(self.r.kv["vision_scan"]) if "vision_scan" in self.r.kv else None
        img = image or ("img-%d" % self.n).encode()
        eid = self.r.xadd("camera_stream:" + self.src, {"image": img})
        self.n += 1
        prof = {"version": self.version if version is None else version}
        if spec:
            prof["scan"] = spec["scan_id"]
            prof["mark"] = int(self.r.kv.get("vision_scan_mark:" + self.src, 0)) or None if mark is None else mark
        res = {"n": self.n, "profile": prof, "provisional": ["demo_door"],
               "detections": self.dets(prof.get("mark"))}
        self.r.xadd("vision_stream:%s:targeted" % self.src,
                    {"result": json.dumps(res), "header": json.dumps({"stream_entry_id": eid.decode()}),
                     "overlay": b"ov-" + img})
        return eid

    def ack(self):
        spec = json.loads(self.r.kv["vision_scan"])
        self.r.set("vision_scan_ack:" + spec["scan_id"],
                   json.dumps({"ok": self.ack_error is None, "error": self.ack_error, "version": self.version}))


@pytest.fixture
def env(tmp_path):
    r = FakeRedis()
    desk = sd.ScanDesk("redis://fake", tmp_path, {"robot1": "robot1"}, log=lambda m: None, redis_client=r)
    v = FakeVision(r)
    v.frame()                                          # 소스 · 비전이 살아 있다
    return desk, r, v, tmp_path


def prep(desk, scan_id="S1", device="go1-001", port="robot1", **kw):
    body = dict({"scan_id": scan_id, "device": device, "target": "door", "steps": 8, "step_deg": 45}, **kw)
    return desk.handle("POST", "/scan/prepare", port, "", json.dumps(body).encode())


def ready(desk, v, scan_id="S1"):
    assert prep(desk, scan_id)[1]["state"] == "loading"
    v.ack()
    v.frame()
    code, st = desk.handle("GET", "/scan/status", "robot1", "scan_id=" + scan_id, None)
    assert st["state"] == "ready" and st["profile_version"] == 12
    return st


def mark(desk, v, rot, scan_id="S1", feed=True, image=None, delay=0.6):
    """mark 를 보내는 동안 비전이 0.5초 뒤 장을 낸다."""
    out = {}
    th = threading.Thread(target=lambda: out.update(r=desk.handle(
        "POST", "/scan/mark", "robot1", "", json.dumps({"scan_id": scan_id, "rotation_deg": rot}).encode())))
    th.start()
    if feed:
        time.sleep(0.05)
        v.frame(image=image)                           # 표시 직후 장 — 0.5초 전이라 버려져야 한다
        time.sleep(delay)
        v.frame(image=image)
    th.join(5)
    return out["r"]


def finish(desk, scan_id="S1"):
    desk.handle("POST", "/scan/finish", "robot1", "", json.dumps({"scan_id": scan_id}).encode())
    for _ in range(50):
        code, res = desk.handle("GET", "/scan/result", "robot1", "scan_id=" + scan_id, None)
        if res["state"] != "running":
            return res
        time.sleep(0.02)
    return res


# ---------------------------------------------------------------- prepare
def test_prepare_writes_scan_key_and_reports_loading(env):
    desk, r, v, _ = env
    code, out = prep(desk)
    assert code == 200 and out == {"scan_id": "S1", "state": "loading", "profile_version": None, "error": None,
                                   "stream": "robot1", "model": "targeted", "targets": ["demo_door"]}
    assert json.loads(r.kv["vision_scan"]) == {"scan_id": "S1", "devices": ["robot1"],
                                               "targets": [{"name": "demo_door"}]}


def test_prepare_object_uses_query(env):
    desk, r, v, _ = env
    code, out = prep(desk, target="object", query="fire extinguisher", query_ko="소화기")
    assert out["targets"] == ["fire extinguisher"]
    assert prep(desk, "S2", target="object")[0] == 400                      # query · query_ko 없음


def test_prepare_is_idempotent_and_busy_is_409(env):
    desk, r, v, _ = env
    prep(desk)
    assert prep(desk)[1]["state"] == "loading"
    code, out = prep(desk, "S2")
    assert code == 409 and out["busy_scan_id"] == "S1"


def test_prepare_wrong_port_and_unknown_device(env):
    desk, *_ = env
    code, out = prep(desk, port="a2")
    assert code == 400 and out["error"] == "wrong_port" and out["stream"] == "robot1"
    assert prep(desk, device="go1-999")[1]["error"] == "unknown_device"


def test_prepare_device_table():
    assert sd.DEVICE_IDS == {"go1-001": "robot1", "ep-001": "ep-001", "a2-001": "a2"}


def test_prepare_no_stream_no_vision(tmp_path):
    r = FakeRedis()
    desk = sd.ScanDesk("x", tmp_path, {}, log=lambda m: None, redis_client=r)
    assert prep(desk)[1]["error"] == "no_stream"
    r.xadd("camera_stream:robot1", {"image": b"x"})
    out = prep(desk, "S2")[1]
    assert out["state"] == "failed" and out["error"] == "no_vision"
    assert "vision_scan" not in r.kv


def test_ack_rejection_fails_with_reason(env):
    desk, r, v, _ = env
    prep(desk)
    v.ack_error = "RuntimeError: CUDA out of memory"
    v.ack()
    st = desk.handle("GET", "/scan/status", "robot1", "scan_id=S1", None)[1]
    assert st["state"] == "failed" and st["error"].startswith("gpu: ")
    assert "vision_scan" not in r.kv


def test_ready_needs_a_result_with_the_scan_tag(env):
    desk, r, v, _ = env
    prep(desk)
    v.ack()
    assert desk.status("S1")["state"] == "loading"          # ack 만으로는 아직
    v.frame()
    assert desk.status("S1")["state"] == "ready"


# ---------------------------------------------------------------- mark · finish
def test_full_scan_flow(env):
    desk, r, v, tmp = env
    ready(desk, v)
    m = mark(desk, v, 0)
    assert m[0] == 200 and m[1]["frame_captured"] and m[1]["captured_at"].endswith("Z")
    first_n = m[1]["frame_n"]
    v.dets = lambda mark: [door(score=30.0, found=False)] if mark == 1 else [door(score=80.0)]
    mark(desk, v, 90)
    st = desk.status("S1")
    assert st["state"] == "collecting" and st["angles_captured"] == [0, 90]
    res = finish(desk)
    assert res["state"] == "done" and res["score_basis"] == "frame" and res["profile_version"] == 12
    assert [f["rotation_deg"] for f in res["frames"]] == [0, 90]
    assert res["frames"][0]["frame_n"] == first_n
    assert res["best"]["rotation_deg"] == 90 and res["best"]["bearing_deg"] == 95.0
    assert res["best"]["distance_valid"] and res["best"]["provisional"]
    assert res["best"]["distance_source"] == "ground_line" and res["best"]["distance_m"] == 2.35
    assert "vision_scan" not in r.kv                        # 끝나면 판 키를 지운다 → 비전이 되돌린다
    assert json.loads((tmp / "scans" / "S1" / "result.json").read_text())["best"]["rotation_deg"] == 90
    assert (tmp / "scans" / "S1" / "90_original.jpg").is_file()


def test_mark_skips_frames_before_settle_and_wrong_mark(env):
    desk, r, v, _ = env
    ready(desk, v)
    m = mark(desk, v, 0)
    # 표시 직후(0.05초) 장이 아니라 0.6초 뒤 장이다 — 원본 번호로 가린다
    assert m[1]["frame_n"] == v.n
    f = desk.scans["S1"].frames[0]
    assert f["rec"]["profile"]["mark"] == 1


def test_mark_without_result_is_not_captured(env, monkeypatch):
    desk, r, v, _ = env
    monkeypatch.setattr(sd, "CAPTURE_S", 0.3)
    ready(desk, v)
    t = time.time()
    m = mark(desk, v, 45, feed=False)
    assert m[1]["frame_captured"] is False and time.time() - t < 1.0
    assert desk.status("S1")["angles_captured"] == []


def test_same_angle_twice_keeps_the_last(env):
    desk, r, v, _ = env
    ready(desk, v)
    a = mark(desk, v, 45)[1]["frame_n"]
    b = mark(desk, v, 45)[1]["frame_n"]
    assert b > a and desk.scans["S1"].frames[45]["frame_n"] == b


def test_frozen_camera_is_excluded(env):
    desk, r, v, _ = env
    ready(desk, v)
    mark(desk, v, 0, image=b"same")
    mark(desk, v, 45, image=b"same")
    res = finish(desk)
    assert res["frames"][1]["frozen"] and "frozen" in res["frames"][1]["reason"]
    assert res["best"]["rotation_deg"] == 0


def test_profile_change_during_scan_fails(env):
    desk, r, v, _ = env
    ready(desk, v)
    v.version = 13
    m = mark(desk, v, 0)
    assert m[1]["frame_captured"] is False and m[1]["error"] == "profile_changed"
    assert desk.status("S1")["state"] == "failed"


def test_finish_without_frames_fails(env):
    desk, r, v, _ = env
    ready(desk, v)
    res = finish(desk)
    assert res["state"] == "failed" and res["error"] == "no_frames"


def test_not_found_is_done_with_best_null(env):
    desk, r, v, _ = env
    v.dets = lambda mark: []
    ready(desk, v)
    mark(desk, v, 0)
    res = finish(desk)
    assert res["state"] == "done" and res["best"] is None and res["frames"][0]["reason"] == "no candidate"


def test_mark_before_ready_is_409(env):
    desk, r, v, _ = env
    prep(desk)
    code, out = desk.handle("POST", "/scan/mark", "robot1", "", b'{"scan_id": "S1", "rotation_deg": 0}')
    assert code == 409 and out["state"] == "loading"


# ---------------------------------------------------------------- cancel · 시간 · 그림 · health
def test_cancel_clears_key_and_allows_next_scan(env):
    desk, r, v, _ = env
    ready(desk, v)
    assert desk.cancel({"scan_id": "S1"})["state"] == "cancelled"
    assert "vision_scan" not in r.kv
    assert prep(desk, "S2")[0] == 200


def test_loading_and_idle_timeouts(env, monkeypatch):
    desk, r, v, _ = env
    now = [1000.0]
    desk.clock = lambda: now[0]
    prep(desk)
    now[0] += sd.LOADING_S + 1
    desk.tick()
    assert desk.scans["S1"].state == "failed" and desk.scans["S1"].error == "timeout"
    prep(desk, "S2")
    v.ack()
    v.frame()
    desk.tick()
    assert desk.scans["S2"].state == "ready"
    now[0] += sd.IDLE_S + 1
    desk.tick()
    assert desk.scans["S2"].error == "timeout" and "vision_scan" not in r.kv


def test_frame_images(env):
    desk, r, v, _ = env
    ready(desk, v)
    mark(desk, v, 90)
    code, (img, ctype) = desk.handle("GET", "/scan/frame", "robot1", "scan_id=S1&rotation_deg=90&kind=original", None)
    assert ctype == "image/jpeg" and img.startswith(b"img-")
    assert desk.handle("GET", "/scan/frame", "robot1", "scan_id=S1&rotation_deg=90&kind=overlay", None)[1][0].startswith(b"ov-")
    assert desk.handle("GET", "/scan/frame", "robot1", "scan_id=S1&rotation_deg=45&kind=original", None)[0] == 404


def test_health_and_unknown_paths(env):
    desk, r, v, _ = env
    prep(desk)
    assert desk.health() == {"busy_scan_id": "S1", "devices": sd.DEVICE_IDS}
    assert desk.handle("GET", "/scan/nope", "robot1", "", None)[0] == 404
    assert desk.handle("GET", "/scan/status", "robot1", "scan_id=ZZ", None)[0] == 404
    assert desk.handle("POST", "/scan/mark", "robot1", "", b"not json")[0] == 400


# ---------------------------------------------------------------- 질의 정리 (2026-11-24)
def test_prepare_object_uses_query_ko_table(env):
    desk, r, v, _ = env
    code, out = prep(desk, target="object", query="celedar or person orhumanbeingorhumanh", query_ko="사람")
    assert code == 200 and out["targets"] == ["person"] and out["query_source"] == "query_ko"
    assert json.loads(r.kv["vision_scan"])["targets"] == [{"name": "person"}]


def test_prepare_object_rejects_broken_query(env):
    desk, *_ = env
    code, out = prep(desk, target="object", query="a or b", query_ko="낯선 것")
    assert code == 400 and out["error"] == "bad_query"
    assert prep(desk, "S3", target="object", query_ko="의자")[1]["targets"] == ["chair"]
