import json

import cv2
import numpy as np

from ds_resources import controller as ctl
from ds_resources import gpu_executor as ge
from ds_resources import memory as mm
from ds_resources.state import StateFile

JPG = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))[1].tobytes()


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeProc:
    n = 7000

    def __init__(self, cmd, **kw):
        FakeProc.n += 1
        self.pid, self.cmd, self.rc = FakeProc.n, cmd, None

    def poll(self):
        return self.rc


class FakeRedis:
    def __init__(self, clock):
        self.clock, self.kv, self.rows, self.down, self.frames = clock, {}, [], False, {}

    def add_frames(self, src, n=50, age_s=0.2):
        now = int(self.clock() * 1000)
        self.frames[src] = [("%d-0" % (now - int(age_s * 1000) - i * 100), {b"image": JPG}) for i in range(n)]

    def xrevrange(self, key, hi, lo, count=None):
        src = key.split(":", 1)[1]
        rows = [(e.encode(), f) for e, f in self.frames.get(src, [])]
        if lo != "-":
            lo_ms = int(lo.split("-")[0])
            rows = [r for r in rows if int(r[0].split(b"-")[0]) >= lo_ms]
        return rows[:count]

    def xadd(self, stream, fields, maxlen=None, approximate=None):
        if self.down:
            raise ConnectionError("redis down")
        self.rows.append(fields)

    def scan_iter(self, pattern, count=None):
        suffix = pattern.split("*")[-1]
        return [k for k in self.kv if k.endswith(suffix)]

    def get(self, k):
        return self.kv.get(k)

    def events(self):
        return [r["event"] for r in self.rows]


def gpus(used=0.0):
    return lambda timeout=2: [{"name": "RTX", "mem_total_mb": 6141.0, "mem_used_mb": used}]


def make(tmp_path, used=0.0):
    c = Clock()
    r = FakeRedis(c)
    st = StateFile(tmp_path / "s.json")
    ex = ge.GpuExecutor(st, r, gpus=gpus(used), popen=FakeProc, clock=c, python="py", log=lambda *_: None)
    k = ctl.Controller(r, st, ex, mm.ResourceMemory(tmp_path / "m.json", clock=c), local_node="wsl", cluster=None,
                       cfg={}, clock=c, log=lambda *_: None, gpus=gpus(used))
    return k, r, c


def test_jpeg_size():
    assert ctl.jpeg_size(JPG) == (1280, 720)


def test_local_fallback_event_sequence_and_fps_cap(tmp_path):
    k, r, _ = make(tmp_path)
    r.add_frames("pi13")
    rid = k.submit(["pi13"], ["yoloe"])
    assert r.events() == ["requested", "cluster_checked", "fallback", "planned", "applied"]
    fb = [x for x in r.rows if x["event"] == "fallback"][0]
    assert fb["mode"] == "local" and fb["basis"] == "saved"
    plan = json.loads([x for x in r.rows if x["event"] == "planned"][0]["plan"])
    u = plan["gpu_units"][0]
    assert u["need_mb"] == round(1300 * 1.2) and plan["algorithm"] == "placement/1"
    unit = list(k.state.units().values())[0]
    assert unit["request_id"] == rid and unit["execution_id"].startswith("exe-")
    cmd = k.ex.procs[unit["unit_id"]].cmd
    assert "--max-fps" in cmd and cmd[cmd.index("--max-fps") + 1].startswith("pi13=")


def test_missing_input_is_failed_not_applied(tmp_path):
    k, r, _ = make(tmp_path)
    r.add_frames("pi13", age_s=30)                                    # 30초 멈춘 영상
    k.submit(["pi13"], ["yoloe"])
    assert "applied" not in r.events()
    f = [x for x in r.rows if x["event"] == "failed"][0]
    assert "입력 영상 없음" in f["reason"]


def test_gpu_refusal_is_failed_with_reason(tmp_path):
    k, r, _ = make(tmp_path, used=5000)
    r.add_frames("pi13")
    k.submit(["pi13"], ["river"])
    assert "applied" not in r.events() and "GPU 메모리 부족" in r.rows[-1]["reason"]


def test_running_needs_frames_and_fresh_input(tmp_path):
    k, r, c = make(tmp_path)
    r.add_frames("pi13")
    k.submit(["pi13"], ["yoloe"])
    u = list(k.state.units().values())[0]
    r.kv["vision_telemetry:h:%d" % u["pid"]] = json.dumps({"streams": {"pi13": {"frames": 30, "fps": 10.0}}})
    r.add_frames("pi13", age_s=20)                                    # 처리한 장은 있지만 입력이 멈췄다
    assert "running" not in [e["event"] for e in k.poll()]            # starting → loaded, running 아님
    assert "running" not in [e["event"] for e in k.poll()]
    r.add_frames("pi13")
    assert [e["event"] for e in k.poll()] == ["running"]
    assert [e["event"] for e in k.poll()] == []                         # 한 번만


def test_degraded_after_sustained_low_fps(tmp_path):
    k, r, c = make(tmp_path)
    r.add_frames("pi13", n=100)                                       # 10 fps
    k.submit(["pi13"], ["yoloe"], target_fps={"pi13": 5.0})
    u = list(k.state.units().values())[0]
    key = "vision_telemetry:h:%d" % u["pid"]
    r.kv[key] = json.dumps({"streams": {"pi13": {"frames": 30, "fps": 5.0}}})
    k.poll(), k.poll()                                                # loaded → running
    r.kv[key] = json.dumps({"streams": {"pi13": {"frames": 10, "fps": 2.0}}})
    assert k.poll() == []                                             # 낮아지기 시작
    c.t += 31
    r.add_frames("pi13", n=100)
    ev = k.poll()
    assert [e["event"] for e in ev] == ["degraded"] and "미달" in ev[0]["reason"]


def test_redis_down_events_go_to_outbox_then_flush(tmp_path):
    k, r, _ = make(tmp_path)
    r.add_frames("pi13")
    r.down = True
    k.submit(["pi13"], ["yoloe"])
    assert r.rows == [] and len(k.state.data["outbox"]) == 5
    assert len(StateFile(tmp_path / "s.json", writer=False).data["outbox"]) == 5    # 파일에 남아 있다
    r.down = False
    k.poll()
    assert r.events()[:5] == ["requested", "cluster_checked", "fallback", "planned", "applied"]
    assert k.state.data["outbox"] == []


def test_stop_emits_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(ge.os, "killpg", lambda *a: None)
    k, r, _ = make(tmp_path)
    r.add_frames("pi13")
    rid = k.submit(["pi13"], ["yoloe"])
    assert k.stop(rid) == 1 and r.events()[-1] == "stopped"


class Cluster:
    def __init__(self, ok):
        self.ok = ok

    def check(self):
        return self.ok, "" if self.ok else "API 응답 없음"

    def nodes(self):
        return [{"name": "pi7", "role": "robot", "ready": True}]


def test_cluster_ok_is_live_mode(tmp_path):
    k, r, _ = make(tmp_path)
    k.cluster = Cluster(True)
    r.add_frames("pi13")
    k.submit(["pi13"], ["yoloe"])
    assert "fallback" not in r.events()
    assert [x for x in r.rows if x["event"] == "planned"][0]["mode"] == "k3s"
