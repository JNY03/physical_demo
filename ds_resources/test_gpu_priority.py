"""GPU 를 우선순위로 판단 — 배치(placement) · 들이기(gpu_executor) · 밀어내기 · 미루기 · 다시 들이기(controller).

모두 가짜로 돈다(GPU 없음). 가짜 nvidia-smi 는 「바탕 사용량 + 적재된(loaded) 단위의 실제 몫」을 돌려준다 —
단위를 멈추면 메모리가 풀리는 것까지 흉내 낸다.
"""
import copy
import json

import pytest

from ds_resources import controller as ctl
from ds_resources import gpu_executor as ge
from ds_resources import memory as mm
from ds_resources import placement as pl
from ds_resources.state import StateFile
from ds_resources.streams import REASON_CODES

from ds_resources.test_controller import Clock, FakeProc, FakeRedis


# ---------------------------------------------------------------- 배치 (placement)
def gpu_node(name="wsl", used=0.0, total=6141.0, **kw):
    return dict({"name": name, "role": "gpu", "ready": True, "gpus": [{"mem_total_mb": total, "mem_used_mb": used}],
                 "cpu_alloc_m": 20000, "mem_alloc_mb": 15000}, **kw)


COSTS = {"yoloe": {"need_mb": 1560, "basis": "measured", "p90_ms": {"pi13": 100.0, "a2": 140.0}},
         "river": {"need_mb": 3360, "basis": "measured", "p90_ms": {"pi13": 600.0, "a2": 800.0}}}


def G(models, sources, crit=None, trigger=None):
    g = {"models": models, "sources": sources, "target_fps": {s: 5.0 for s in sources}}
    if crit:
        g["criticality"] = crit
    if trigger:
        g["trigger"] = trigger
    return g


def plan(*bundles, node=None, costs=COSTS):
    return pl.plan({"request_id": "r1", "gpu": list(bundles), "cpu": []},
                   {"nodes": [node or gpu_node()], "costs": costs}, "h", "wsl")


def test_new_codes_have_korean_text():
    for code in ("gpu_preempted", "gpu_deferred"):
        assert REASON_CODES[code]


def test_higher_priority_is_placed_first_and_lower_is_refused_preempted():
    # 여유 3541 — 둘 다는 안 들어간다. 이름순이면 river 가 먼저지만 safety 인 yoloe 가 먼저 자리를 잡는다
    out = plan(G(["river"], ["a2"], "best_effort"), G(["yoloe"], ["pi13"], "safety"), node=gpu_node(used=2000))
    assert [u["unit"] for u in out["gpu_units"]] == ["gpu:yoloe"]
    assert out["gpu_units"][0]["rank"] == 0 and out["gpu_units"][0]["criticality"] == "safety"
    (ref,) = out["refused"]
    assert ref["item"] == "gpu:river" and ref["code"] == "gpu_preempted" and ref["rank"] == 2


def test_lower_priority_on_request_is_deferred_not_refused():
    out = plan(G(["river"], ["a2"], "best_effort", "on_request"), G(["yoloe"], ["pi13"], "safety"),
               node=gpu_node(used=2000))
    assert not out["refused"]
    (w,) = [x for x in out.get("waiting") or [] if x["code"] == "gpu_deferred"]
    assert w["item"] == "gpu:river" and w["models"] == ["river"] and w["sources"] == ["a2"]
    assert w["need_mb"] == 3360 and w["trigger"] == "on_request" and w["rank"] == 2


def test_legacy_requests_keep_name_order_and_gpu_mem():
    # 기능 없이 온 옛 요청 — 모두 mission 이라 예전처럼 이름순 · 같은 우선순위끼리는 gpu_mem
    out = plan(G(["yoloe"], ["pi13"]), G(["river"], ["a2"]), node=gpu_node(used=1500))
    assert [u["unit"] for u in out["gpu_units"]] == ["gpu:river"]
    assert out["refused"][0]["code"] == "gpu_mem"
    assert out["gpu_units"][0]["criticality"] == "mission" and "evict" not in out["gpu_units"][0]


def test_everything_fits_is_unchanged_apart_from_added_fields():
    out = plan(G(["yoloe"], ["pi13"]), G(["river"], ["a2"]))
    assert not out["refused"] and not out.get("waiting")
    assert [u["unit"] for u in out["gpu_units"]] == ["gpu:river", "gpu:yoloe"]
    for u in out["gpu_units"]:
        assert "evict" not in u and u["rank"] == 1
    assert "_gpu_held" not in out and "_gpu_evicted" not in out


LOADED = [{"unit_id": "gpu-a", "rank": 2, "mem_mb": 1000, "gpu_util": 0.1},
          {"unit_id": "gpu-b", "rank": 2, "mem_mb": 2000, "gpu_util": 0.2},
          {"unit_id": "gpu-c", "rank": 1, "mem_mb": 5000, "gpu_util": 0.3}]


def big(used=8000.0):
    return gpu_node(total=10000.0, used=used, gpu_loaded=copy.deepcopy(LOADED))   # 여유 10000-8000-300 = 1700


def one(need, crit):
    return plan(G(["river"], ["a2"], crit), node=big(), costs={"river": {"need_mb": need, "basis": "measured",
                                                                       "p90_ms": {"a2": 100.0}}})


def test_eviction_picks_lowest_priority_then_largest_memory():
    out = one(3360, "mission")                                        # 모자람 1660 → best_effort 중 큰 것(b) 하나로 충분
    assert out["gpu_units"][0]["evict"] == ["gpu-b"]
    out = one(4000, "mission")                                        # 모자람 2300 → b(2000) + a(1000)
    assert out["gpu_units"][0]["evict"] == ["gpu-a", "gpu-b"]
    assert one(4000, "mission") == one(4000, "mission")               # 되풀이 가능


def test_equal_or_higher_priority_is_never_evicted():
    out = one(6000, "mission")                 # 모자람 4300 — 낮은 것(a+b 3000)으론 모자라고 c 는 같은 mission
    assert not out["gpu_units"] and out["refused"][0]["code"] == "gpu_mem"
    out = one(6000, "best_effort")             # 더 낮은 것이 없다 — 아무것도 비우지 않는다. 높은 일이 쥐고 있으니 preempted
    assert not out["gpu_units"] and out["refused"][0]["code"] == "gpu_preempted"
    out = one(6000, "safety")                  # safety 는 mission 까지 비울 수 있다 (낮은 것부터)
    assert out["gpu_units"][0]["evict"] == ["gpu-a", "gpu-b", "gpu-c"]


def test_two_bundles_do_not_evict_the_same_unit_twice():
    node = big()
    costs = {"river": {"need_mb": 3000, "basis": "measured", "p90_ms": {"a2": 100.0}},
             "yoloe": {"need_mb": 1700, "basis": "measured", "p90_ms": {"pi13": 100.0}}}
    out = plan(G(["river"], ["a2"], "safety"), G(["yoloe"], ["pi13"], "mission"), node=node, costs=costs)
    ev = [x for u in out["gpu_units"] for x in u.get("evict") or []]
    assert len(ev) == len(set(ev))
    assert out["gpu_units"][0]["unit"] == "gpu:river" and out["gpu_units"][0]["evict"] == ["gpu-b"]
    # 남은 여유 1700 - 3000 + 2000 = 700 → yoloe 1700 은 a(1000) 를 비우면 들어간다 (b 는 이미 비우기로 했다)
    assert out["gpu_units"][1]["evict"] == ["gpu-a"]


# ---------------------------------------------------------------- 들이기 (gpu_executor)
class Gpu:
    """가짜 nvidia-smi — 바탕 사용량 + loaded 단위의 실제 몫(예약 ÷ MARGIN)"""

    def __init__(self, state, base=0.0, total=6141.0):
        self.state, self.base, self.total = state, base, total

    def __call__(self, timeout=2):
        used = self.base + sum(float(u.get("reserved_mb") or 0) / ge.MARGIN for u in self.state.units().values()
                               if u.get("kind") == "gpu" and u.get("status") == "loaded")
        return [{"name": "RTX", "mem_total_mb": self.total, "mem_used_mb": used}]


@pytest.fixture
def quick_stop(monkeypatch):
    """killpg 가짜 — 그 pid 의 가짜 프로세스를 끝낸다 (stop 이 실제로 기다리지 않게)"""
    procs = []

    def killpg(pid, sig):
        for p in procs:
            if p.pid == pid:
                p.rc = -sig

    class P(FakeProc):
        def __init__(self, cmd, **kw):
            super().__init__(cmd, **kw)
            procs.append(self)

    monkeypatch.setattr(ge.os, "killpg", killpg)
    return P


def executor(tmp_path, base, popen):
    c = Clock()
    st = StateFile(tmp_path / "s.json")
    g = Gpu(st, base)
    ex = ge.GpuExecutor(st, None, gpus=g, popen=popen, clock=c, python="py", log=lambda *_: None)
    return ex, g, c


def loaded(ex, models, need, crit=None, **kw):
    uid = ex.launch(dict({"devices": ["pi13"], "models": models, "request_id": "r-" + models[0],
                          **({"criticality": crit} if crit else {})}, **kw), need, {"mode": "normal"})
    ex.state.update_unit(uid, status="loaded")
    return uid


def test_admit_proposes_only_lower_priority_victims(tmp_path, quick_stop):
    ex, g, _c = executor(tmp_path, 1500, quick_stop)
    lo = loaded(ex, ["yoloe"], 2400, "best_effort")                   # 실제 2000
    mid = loaded(ex, ["river"], 1800, "mission")                      # 실제 1500 → 사용 5000, 여유 1141
    ok, _why, calc = ex.admit(2400, "measured", rank=1)               # mission: best_effort 만 비울 수 있다
    assert not ok and calc["code"] == "gpu_mem" and calc["evict"] == [lo] and calc["evict_mode"] == "normal"
    assert calc["evict_mb"] == 2000
    ok, _why, calc = ex.admit(2400, "measured", rank=2)               # best_effort: 같은 것은 비우지 않는다
    assert not ok and "evict" not in calc
    ok, _why, calc = ex.admit(2400, "measured")                       # 옛 호출(rank 없음) — 예전 그대로
    assert not ok and "evict" not in calc
    ok, _why, calc = ex.admit(4000, "measured", rank=0)               # safety: 낮은 것부터 (best_effort → mission)
    assert calc["evict"] == [lo, mid]
    ok, _why, calc = ex.admit(9000, "measured", rank=0)               # 다 비워도 안 되면 아무것도 제안하지 않는다
    assert "evict" not in calc


def test_admit_evicts_a_lower_exclusive_measurement(tmp_path, quick_stop):
    ex, _g, _c = executor(tmp_path, 0, quick_stop)
    uid = ex.launch({"devices": ["pi13"], "models": ["yoloe"], "criticality": "best_effort"}, 1560,
                    {"mode": "exclusive"})
    ok, _why, calc = ex.admit(1000, "measured", rank=1)
    assert not ok and calc["code"] == "gpu_exclusive" and calc["evict"] == [uid]


def test_preempt_stops_and_defers_with_hold(tmp_path, quick_stop):
    ex, _g, c = executor(tmp_path, 0, quick_stop)
    uid = loaded(ex, ["yoloe"], 1560, "best_effort", trigger="on_activity")
    gone = ex.preempt([uid], "밀려남", by={"unit": "gpu:river"})
    assert [u["unit_id"] for u in gone] == [uid]
    u = ex.state.units()[uid]
    assert u["status"] == "preempted" and u["preempted_by"] == {"unit": "gpu:river"} and not ex.active_units()
    e = ex.deferred()[uid]
    assert e["code"] == "gpu_preempted" and e["rank"] == 2 and e["hold_until"] == c.t + ge.READMIT_HOLD_S
    assert e["spec"]["models"] == ["yoloe"] and e["spec"]["criticality"] == "best_effort"
    assert ex.preempt([uid], "또") == []                               # 이미 멈춘 것은 다시 멈추지 않는다


# ---------------------------------------------------------------- 밀어내기 · 미루기 · 다시 들이기 (controller)
PRIO = {"yoloe": {"criticality": "best_effort", "trigger": "on_activity"},
        "river": {"criticality": "safety", "trigger": "always"}}


def controller(tmp_path, base, popen, prio=PRIO):
    c = Clock()
    r = FakeRedis(c)
    st = StateFile(tmp_path / "s.json")
    g = Gpu(st, base)
    ex = ge.GpuExecutor(st, r, gpus=g, popen=popen, clock=c, python="py", log=lambda *_: None)
    k = ctl.Controller(r, st, ex, mm.ResourceMemory(tmp_path / "m.json", clock=c), local_node="wsl", cluster=None,
                       cfg={}, clock=c, log=lambda *_: None, gpus=g)
    k.gpu_priority = lambda models, sources, functions: dict(prio.get(models[0]) or {})
    return k, r, c, g


def mark_loaded(k, r, uid):
    pid = k.state.units()[uid]["pid"]
    r.kv["vision_telemetry:h:%d" % pid] = json.dumps({"streams": {}})
    k.poll()
    assert k.state.units()[uid]["status"] == "loaded"


def gpu_units(k, status=("starting", "loaded")):
    return {u["models"][0]: uid for uid, u in k.state.units().items()
            if u.get("kind") == "gpu" and u.get("status") in status}


def test_higher_priority_evicts_lower_then_lower_resumes_after_hold(tmp_path, quick_stop):
    # 바탕 1700 MB 를 다른 것이 쓴다. yoloe(best_effort, 1560 예약 · 실제 1300) 가 돌고 있으면 river(safety 3360) 는
    # 여유 6141-1700-1300-300 = 2841 이라 안 들어간다 → yoloe 를 비우고 들인다
    k, r, c, _g = controller(tmp_path, 1700, quick_stop)
    r.add_frames("pi13")
    r.add_frames("a2")
    rid1 = k.submit(["pi13"], ["yoloe"])
    lo = gpu_units(k)["yoloe"]
    mark_loaded(k, r, lo)
    r.rows.clear()
    rid2 = k.submit(["a2"], ["river"])
    plan = json.loads([x for x in r.rows if x["event"] == "planned"][0]["plan"])
    assert plan["gpu_units"][0]["evict"] == [lo] and plan["gpu_units"][0]["rank"] == 0
    st = [x for x in r.rows if x["event"] == "stopped"]
    assert len(st) == 1 and st[0]["code"] == "gpu_preempted" and st[0]["request_id"] == rid1
    det = json.loads(st[0]["detail"])
    assert det["unit_id"] == lo and det["preempted"] and det["by"]["request_id"] == rid2
    assert "우선순위" in st[0]["reason"]
    assert r.events()[-1] == "applied" and "failed" not in r.events()
    assert k.state.units()[lo]["status"] == "preempted" and set(gpu_units(k)) == {"river"}
    assert k.pending_sources(rid1) == ["pi13"]                         # 밀려난 단위도 기다릴 이유다
    # river 가 끝났다 — 메모리는 풀렸지만 hold 안이라 아직 들이지 않는다 (되풀이 방지)
    k.stop(rid2)
    r.rows.clear()
    k.poll()
    assert "applied" not in r.events() and k.state.units()[lo]["status"] == "preempted"
    c.t += ge.READMIT_HOLD_S + 1
    k.poll()
    (ap,) = [x for x in r.rows if x["event"] == "applied"]
    det = json.loads(ap["detail"])
    assert ap["request_id"] == rid1 and det["resumed"] and det["unit_id"] == lo
    assert det["resumed_from"] == "gpu_preempted" and "다시 들임" in ap["reason"]
    u = k.state.units()[lo]
    assert u["status"] == "starting" and u["criticality"] == "best_effort" and not k.ex.deferred()


def test_equal_priority_is_not_evicted(tmp_path, quick_stop):
    prio = {"yoloe": {"criticality": "mission", "trigger": "always"},
            "river": {"criticality": "mission", "trigger": "always"}}
    k, r, _c, _g = controller(tmp_path, 1700, quick_stop, prio)
    r.add_frames("pi13")
    r.add_frames("a2")
    k.submit(["pi13"], ["yoloe"])
    lo = gpu_units(k)["yoloe"]
    mark_loaded(k, r, lo)
    r.rows.clear()
    k.submit(["a2"], ["river"])
    assert "stopped" not in r.events() and "applied" not in r.events()
    assert [x["code"] for x in r.rows if x["event"] == "failed"] == ["gpu_mem"]   # 예전과 같은 코드 (cli 재시도)
    assert k.state.units()[lo]["status"] == "loaded" and not k.ex.deferred()


def test_on_request_is_deferred_then_admitted_when_memory_frees(tmp_path, quick_stop):
    prio = {"yoloe": {"criticality": "safety", "trigger": "always"},
            "river": {"criticality": "best_effort", "trigger": "on_request"}}
    k, r, c, _g = controller(tmp_path, 1700, quick_stop, prio)
    r.add_frames("pi13")
    r.add_frames("a2")
    rid1 = k.submit(["pi13"], ["yoloe"])
    hi = gpu_units(k)["yoloe"]
    mark_loaded(k, r, hi)
    r.rows.clear()
    rid2 = k.submit(["a2"], ["river"])
    dg = [x for x in r.rows if x["event"] == "degraded" and x.get("code") == "gpu_deferred"]
    assert len(dg) == 1 and "failed" not in r.events() and "stopped" not in r.events()
    assert k.state.units()[hi]["status"] == "loaded"                  # 높은 것은 그대로
    (e,) = k.ex.deferred().values()
    assert e["code"] == "gpu_deferred" and e["request_id"] == rid2 and e["spec"]["devices"] == ["a2"]
    assert k.pending_sources(rid2) == ["a2"]
    r.rows.clear()
    k.poll()                                                           # 아직 메모리가 없다 — 조용히 기다린다
    assert r.events() == []
    k.stop(rid1)                                                       # 높은 일이 끝났다 → 다음 poll 에서 바로
    r.rows.clear()
    k.poll()
    (ap,) = [x for x in r.rows if x["event"] == "applied"]
    assert ap["request_id"] == rid2 and json.loads(ap["detail"])["resumed_from"] == "gpu_deferred"
    assert set(gpu_units(k)) == {"river"} and not k.ex.deferred()


def test_deferred_retry_goes_in_priority_order(tmp_path, quick_stop):
    k, _r, c, g = controller(tmp_path, 6000, quick_stop)               # 처음에는 아무것도 안 들어간다
    for name, crit in (("a-low", "best_effort"), ("b-high", "safety"), ("c-mid", "mission")):
        k.ex.defer(name, {"unit_id": None, "unit": "gpu:" + name, "spec": {"devices": ["pi13"], "models": ["yoloe"],
                                                                            "criticality": crit, "request_id": "r"},
                          "need_mb": 2000, "basis": "measured", "estimate_mb": None, "rank": pl.PRIORITY_RANK[crit],
                          "request_id": "r", "since": c.t, "hold_until": 0.0, "code": "gpu_deferred"})
    assert k.retry_deferred(force=True) == []
    g.base = 6141 - 300 - 2000 * 1.5                                   # 하나 반 만큼 비었다 — 하나만 들어간다
    ev = k.retry_deferred(force=True)
    assert len(ev) == 1 and sorted(k.ex.deferred()) == ["a-low", "c-mid"]   # safety 가 먼저
    assert k.retry_deferred() == []                                    # 간격 안 — 다시 보지 않는다


def test_stop_drops_deferred_units_of_that_request(tmp_path, quick_stop):
    k, r, c, _g = controller(tmp_path, 1700, quick_stop)
    r.add_frames("pi13")
    r.add_frames("a2")
    rid1 = k.submit(["pi13"], ["yoloe"])
    lo = gpu_units(k)["yoloe"]
    mark_loaded(k, r, lo)
    k.submit(["a2"], ["river"])
    assert lo in k.ex.deferred()
    k.stop(rid1)
    assert not k.ex.deferred() and k.state.units()[lo]["status"] == "stopped"
    c.t += ge.READMIT_HOLD_S + 1
    r.rows.clear()
    k.poll()
    assert "applied" not in r.events()


def test_gpu_priority_from_functions(tmp_path):
    from ds_resources import functions as fnc
    k, _r, _c, _g = controller(tmp_path, 0, FakeProc)
    del k.gpu_priority                                                 # 진짜 계산으로
    base = {"pattern": "server_only", "note": "x",
            "stages": {"server": {"cpu_m": 100, "mem_mb": 100, "roles": ["gpu"], "note": "s"}}}
    k._funcs = fnc.load({"functions": {
        "a": dict(base, criticality="best_effort", trigger="always", options=["yoloe"]),
        "b": dict(base, criticality="mission", trigger="on_request", options=["yoloe", "river"]),
        "c": dict(base, criticality="mission", trigger="on_activity", options=["river"])}})
    assert k.gpu_priority(["yoloe"], ["pi13"], None) == {}                         # 옛 요청 — mission 으로 본다
    assert k.gpu_priority(["yoloe"], ["pi13"], ["a", "b"]) == {
        "criticality": "mission", "trigger": "on_request", "functions": ["a", "b"]}
    assert k.gpu_priority(["river"], ["pi13"], ["b", "c"])["trigger"] == "on_activity"   # 같은 중요도면 급한 쪽
    assert k.gpu_priority(["yoloe"], ["pi13"], {"pi13": ["a"], "a2": ["b"]})["criticality"] == "best_effort"
    assert k.gpu_priority(["yoloe"], ["pi13"], ["c"]) == {}                        # 이 모델을 쓰는 기능이 없다
