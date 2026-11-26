"""감독자 시험 — 가짜 popen · 시계 · Redis (진짜 프로세스 · GPU 없음)."""
import json
import signal

import pytest

from ds_resources import desired as ds
from ds_resources import supervisor as sv
from ds_resources.streams import orch_record


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class Proc:
    def __init__(self, pid, cmd, die_on=(signal.SIGINT,)):
        self.pid, self.cmd, self.rc, self.die_on = pid, cmd, None, die_on

    def poll(self):
        return self.rc


class Popen:
    def __init__(self):
        self.procs, self.next_pid, self.die_on = [], 100, (signal.SIGINT,)
        self.kills = []

    def __call__(self, cmd, **kw):
        assert kw["start_new_session"] is True and "stdout" in kw
        self.next_pid += 1
        p = Proc(self.next_pid, cmd, self.die_on)
        self.procs.append(p)
        return p

    def killpg(self, pid, sig):
        p = next(p for p in self.procs if p.pid == pid)
        self.kills.append((self.name_of(p), sig))
        if sig in p.die_on or sig == signal.SIGKILL:
            p.rc = -sig

    @staticmethod
    def name_of(p):
        s = " ".join(p.cmd)
        return "server" if sv.SERVER_SCRIPT in s else "slice" if sv.SLICE_SCRIPT in s else "vision"

    def started(self, name):
        return [p for p in self.procs if self.name_of(p) == name]


class Redis:
    def __init__(self, down=False):
        self.kv, self.ex, self.xs, self.down = {}, {}, [], down

    def get(self, k):
        if self.down:
            raise ConnectionError("down")
        return self.kv.get(k)

    def set(self, k, v, ex=None):
        if self.down:
            raise ConnectionError("down")
        self.kv[k], self.ex[k] = v, ex

    def xadd(self, stream, fields, **kw):
        if self.down:
            raise ConnectionError("down")
        self.xs.append((stream, fields))


def doc(**kw):
    d = {"schema": ds.SCHEMA, "running": True, "server": {"sources": ["pi13", "a2"], "robot_cam": 1},
         "slice": {"enabled": False}, "vision": {"enabled": True, "orchestrate": False, "models": ["yoloe"],
                                                 "profile": {"models": {"yoloe": {"overlay": {"count": True}}}}}}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = dict(d[k], **v)
        else:
            d[k] = v
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "STOP_WAITS", (6.0, 2.0, 1.0))
    clk, pp, r = Clock(), Popen(), Redis()
    busy, others = set(), {}

    def make(role="all", redis=r):
        return sv.Supervisor(role, redis, tmp_path, popen=pp, clock=clk, wall=clk, sleep=clk.sleep, killpg=pp.killpg,
                             port_in_use=lambda port: port in busy, find_procs=lambda s, ex: others.get(s, []),
                             python="/py", log=lambda m: None)

    def put(d, redis=r):
        out, errs = ds.write(redis, tmp_path / "desired.json", d)
        assert not errs, errs
        return out

    class E:
        pass
    e = E()
    e.clk, e.pp, e.r, e.busy, e.others, e.make, e.put, e.dir = clk, pp, r, busy, others, make, put, tmp_path
    return e


def run_until(s, clk, secs, step=1.0):
    end = clk.t + secs
    while clk.t < end:
        s.tick()
        clk.t += step
    s.tick()


def test_start_order_and_restart_rules(env):
    env.put(doc(server={"sources": ["cam360"]}, slice={"enabled": True, "u": 10}))
    s = env.make()
    s.tick()
    assert [env.pp.name_of(p) for p in env.pp.procs] == ["server"]                # 슬라이스 · 비전은 서버를 기다린다
    cmd = env.pp.procs[0].cmd
    assert cmd[:2] == ["/py", str(sv.REPO / sv.SERVER_SCRIPT)] and cmd[cmd.index("--on") + 1:cmd.index("--robot-cam")] \
        == ["cam360", "cam360_slice"]
    run_until(s, env.clk, 8)
    assert [env.pp.name_of(p) for p in env.pp.procs] == ["server", "slice", "vision"]
    assert "--u" in env.pp.started("slice")[0].cmd and "-30" in env.pp.started("slice")[0].cmd
    vis = env.pp.started("vision")[0].cmd
    assert vis[vis.index("--devices") + 1:vis.index("--models")] == ["cam360", "cam360_slice"]
    assert vis[vis.index("--profile") + 1] == str(env.dir / "profile.json")
    assert s.converged
    n = len(env.pp.procs)
    # 프로필만 바뀜 → 아무것도 다시 띄우지 않고 파일만
    env.put(doc(server={"sources": ["cam360"]}, slice={"enabled": True, "u": 10},
                vision={"profile": {"models": {"yoloe": {"overlay": {"count": False}}}}}))
    run_until(s, env.clk, 3)
    assert len(env.pp.procs) == n and json.loads((env.dir / "profile.json").read_text())["profile_version"] == 2
    # 슬라이스 각도만 → 슬라이스만
    env.put(doc(server={"sources": ["cam360"]}, slice={"enabled": True, "u": 20}))
    run_until(s, env.clk, 3)
    assert [env.pp.name_of(p) for p in env.pp.procs[n:]] == ["slice"]
    assert [k[0] for k in env.pp.kills] == ["slice"]
    n = len(env.pp.procs)
    # 로봇 캠 → 서버가 바뀌면 모두 다시 (끄는 순서: 비전 → 슬라이스 → 서버)
    env.pp.kills.clear()
    env.put(doc(server={"sources": ["cam360"], "robot_cam": 2}, slice={"enabled": True, "u": 20}))
    run_until(s, env.clk, 8)
    assert [k[0] for k in env.pp.kills] == ["vision", "slice", "server"]
    assert [env.pp.name_of(p) for p in env.pp.procs[n:]] == ["server", "slice", "vision"]
    # 비전 모델 → 비전만
    n, env.pp.kills[:] = len(env.pp.procs), []
    env.put(doc(server={"sources": ["cam360"], "robot_cam": 2}, slice={"enabled": True, "u": 20},
                vision={"models": ["yoloe", "river"]}))
    run_until(s, env.clk, 3)
    assert [env.pp.name_of(p) for p in env.pp.procs[n:]] == ["vision"] and [k[0] for k in env.pp.kills] == ["vision"]
    assert s.converged


def test_crash_backoff(env):
    env.put(doc(vision={"enabled": False}))
    s = env.make("core")
    s.tick()
    env.pp.procs[-1].rc = 1
    s.tick()
    c = s.comps["server"]
    assert c.status == "crashed" and c.restarts == 1 and "코드 1" in c.reason
    env.clk.t += 1.0
    s.tick()
    assert len(env.pp.procs) == 1                                                   # 2초 대기 중
    env.clk.t += 1.0
    s.tick()
    assert len(env.pp.procs) == 2
    env.pp.procs[-1].rc = 1
    s.tick()
    env.clk.t += 3.0
    s.tick()
    assert len(env.pp.procs) == 2 and c.restarts == 2                               # 둘째는 4초
    env.clk.t += 1.0
    s.tick()
    assert len(env.pp.procs) == 3
    for _ in range(10):                                                             # 대기는 BACKOFF_MAX_S 를 넘지 않는다
        env.pp.procs[-1].rc = 1
        s.tick()
        env.clk.t += sv.BACKOFF_MAX_S
        s.tick()
    assert c.next_try - env.clk.t <= sv.BACKOFF_MAX_S
    env.clk.t += sv.STABLE_S + 5                                                    # 오래 돌면 대기가 처음으로
    s.tick()
    s.tick()
    assert c.status == "running" and c.fails == 0


def test_never_exits_when_nothing_to_run(env, monkeypatch):
    env.put(doc(running=False, server={"sources": []}))
    s = env.make()
    calls = {"n": 0}

    def wait(t):
        calls["n"] += 1
        if calls["n"] >= 5:
            s.stop_event.set()
    monkeypatch.setattr(s.stop_event, "wait", wait)
    assert s.run() == 0 and calls["n"] == 5 and env.pp.procs == []
    rep = json.loads((env.dir / "reported_all.json").read_text())
    assert rep["converged"] is True and all(c["status"] == "stopped" for c in rep["components"].values())


def test_profile_version_only_on_content_change(tmp_path):
    p = tmp_path / "profile.json"
    assert sv.write_profile(p, {"models": {"a": 1}}) == (1, True)
    m = p.stat().st_mtime_ns
    assert sv.write_profile(p, {"models": {"a": 1}, "profile_version": 99}) == (1, False)  # 들어온 번호는 무시
    assert p.stat().st_mtime_ns == m
    assert sv.write_profile(p, {"models": {"a": 2}}) == (2, True)
    assert json.loads(p.read_text()) == {"models": {"a": 2}, "profile_version": 2}


def test_signal_stop_ordering(env):
    env.put(doc(vision={"enabled": False}))
    s = env.make("core")
    s.tick()
    env.pp.die_on = (signal.SIGTERM,)
    env.pp.procs[-1].die_on = (signal.SIGTERM,)
    t0 = env.clk.t
    s.shutdown()
    assert env.pp.kills == [("server", signal.SIGINT), ("server", signal.SIGTERM)]
    assert env.clk.t - t0 >= 6.0                                                    # SIGINT 뒤 6초 기다림
    env.pp.kills.clear()
    env.put(doc(vision={"enabled": False}))
    s = env.make("core")
    s.tick()
    env.pp.procs[-1].die_on = ()                                                    # 아무것도 안 들음 → KILL
    s.shutdown()
    assert [k[1] for k in env.pp.kills] == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    rep = json.loads((env.dir / "reported_core.json").read_text())
    assert rep["components"]["server"]["status"] == "stopped"
    assert env.r.xs[-1][1]["event"] == "stopped"


def test_blocked_port_and_foreign_process(env):
    env.put(doc(vision={"enabled": False}))
    env.busy.add(10011)                                                             # pi13
    s = env.make("core")
    run_until(s, env.clk, 5)
    c = s.comps["server"]
    assert env.pp.procs == [] and c.status == "blocked" and "pi13:10011" in c.reason and not s.converged
    env.busy.clear()
    env.others[sv.SERVER_SCRIPT] = [4242]                                           # 런처 창의 서버
    s.tick()
    assert c.status == "blocked" and "4242" in c.reason and c.restarts == 0
    env.others.clear()
    s.tick()
    assert len(env.pp.procs) == 1 and c.status == "starting"


def test_converged_transitions_and_stream(env):
    env.put(doc(vision={"enabled": False}))
    s = env.make("core")
    s.tick()
    assert s.converged is False
    run_until(s, env.clk, 3)
    assert s.converged is True
    env.pp.procs[-1].rc = 1
    s.tick()
    assert s.converged is False
    run_until(s, env.clk, 4)
    assert s.converged is True
    ev = [(f["event"], f["converged"], f["desired_version"], f["request_id"]) for _, f in env.r.xs]
    assert ev == [("requested", "0", "1", "desired-v1"), ("applied", "1", "1", "desired-v1"),
                  ("degraded", "0", "1", "desired-v1"), ("applied", "1", "1", "desired-v1")]    # 바뀔 때만
    rep = json.loads(env.r.kv["ds_reported:core"])
    assert rep["desired_version"] == 1 and rep["converged"] is True and rep["components"]["server"]["restarts"] == 1
    assert env.r.ex["ds_reported:core"] == 6
    env.put(doc(running=False, server={"sources": []}))
    run_until(s, env.clk, 2)
    assert env.r.xs[-1][1]["event"] == "stopped" and env.r.xs[-1][1]["converged"] == "1"


def test_desired_from_file_when_redis_down(env):
    env.put(doc(vision={"enabled": False}))
    down = Redis(down=True)
    s = env.make("core", redis=down)
    s.tick()
    assert s.source == "file" and s.version == 1 and len(env.pp.procs) == 1
    assert json.loads((env.dir / "reported_core.json").read_text())["desired_version"] == 1


def test_orch_record_optional_fields():
    a = orch_record("applied", "desired-v3")
    assert "desired_version" not in a and "converged" not in a
    b = orch_record("applied", "desired-v3", desired_version=3, converged=True)
    assert b["desired_version"] == "3" and b["converged"] == "1"


class FakeState:
    def __init__(self):
        self.data, self._units, self.closed = {"requests": {}}, {}, False

    def units(self):
        return dict(self._units)

    def close(self):
        self.closed = True


class FakeEx:
    def __init__(self, st):
        self.st = st

    def active_units(self):
        return {k: u for k, u in self.st._units.items() if u.get("kind") == "gpu" and u["status"] in ("starting", "loaded")}


class FakeCtl:
    def __init__(self):
        self.state = FakeState()
        self.ex = FakeEx(self.state)
        self.calls, self.refuse = [], False

    def cleanup_previous(self):
        self.calls.append("cleanup")
        return 0

    def pending_sources(self, rid):
        return []

    def replan_if_needed(self, rid):
        return None

    def submit(self, sources, models, profile, profile_path, target_fps=None, assign=None, manual=None):
        self.calls.append(("submit", tuple(sources), tuple(models), (profile or {}).get("profile_version")))
        self.last_submit = {"assign": assign, "manual": manual}      # 기기별 모델 매칭 · 수동 배치 (2026-11-26)
        rid = "req-%d" % len(self.calls)
        if self.refuse:
            self.state.data["requests"][rid] = {"plan": {"refused": [{"reason": "GPU 메모리 부족"}]}}
        else:
            self.state._units["u1"] = {"kind": "gpu", "status": "starting", "request_id": rid}
        return rid

    def poll(self):
        return []

    def stop(self, rid, reason):
        self.calls.append(("stop", rid))
        for u in self.state._units.values():
            u["status"] = "stopped"

    def flush_outbox(self):
        pass


def test_orchestrate_in_process(env):
    ctls, lock = [], {"held": True}

    def factory(sup):
        if lock["held"]:
            return None, "다른 오케스트레이터가 실행 상태 파일을 잡고 있음"
        k = FakeCtl()
        ctls.append(k)
        return k, ""
    env.put(doc(vision={"orchestrate": True}))
    s = env.make("vision")
    s.controller_factory = factory
    s.tick()
    c = s.comps["vision"]
    assert isinstance(c, sv.VisionOrch) and c.status == "blocked" and env.pp.procs == []
    lock["held"] = False
    s.tick()
    k = ctls[-1]
    assert k.calls[0] == "cleanup" and k.calls[1] == ("submit", ("pi13", "a2"), ("yoloe",), 1) and c.status == "starting"
    k.state._units["u1"]["status"] = "loaded"
    s.tick()
    assert c.status == "running" and s.converged
    k.state._units["u1"]["status"] = "stopped"                                     # Executor 가 다시 띄우기를 포기
    s.tick()
    assert c.status == "crashed" and "작업이 없음" in c.reason and k.state.closed
    env.clk.t += sv.BACKOFF_S
    s.tick()
    assert len(ctls) == 2 and c.status == "starting"
    # 직접 방식으로 바꾸면 컨트롤러를 멈추고 vision_infer 를 띄운다
    env.put(doc(vision={"orchestrate": False}))
    s.tick()
    assert ("stop", "req-2") in ctls[-1].calls and ctls[-1].state.closed
    assert isinstance(s.comps["vision"], sv.VisionProc) and len(env.pp.started("vision")) == 1


def test_orchestrate_passes_model_match_and_manual_place(env):
    """기기별 모델 매칭 · 수동 배치는 원하는 상태에서 컨트롤러로 그대로 간다 (2026-11-26)"""
    ctls = []

    def factory(sup):
        k = FakeCtl()
        ctls.append(k)
        return k, ""
    env.put(doc(vision={"orchestrate": True, "assign": {"pi13": ["yoloe"], "a2": []},
                        "manual": {"enabled": True, "place": {"object_detect.device": "pi1"}}}))
    s = env.make("vision")
    s.controller_factory = factory
    s.tick()
    assert ctls[-1].last_submit == {"assign": {"pi13": ["yoloe"], "a2": []},
                                    "manual": {"object_detect.device": "pi1"}}
    # 수동 배치를 끄면 자리를 넘기지 않는다 (적어 둔 것은 그대로 남는다)
    env.put(doc(vision={"orchestrate": True,
                        "manual": {"enabled": False, "place": {"object_detect.device": "pi1"}}}))
    s2 = env.make("vision")
    s2.controller_factory = factory
    s2.tick()
    assert ctls[-1].last_submit == {"assign": None, "manual": None}
