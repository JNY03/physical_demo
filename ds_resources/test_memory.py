import json

from ds_resources import memory as mm
from ds_resources import probe
from ds_resources.recorder import LocalRecorder, telemetry_records
from ds_resources.streams import NODE_STREAM


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- 노드 기본 자원
def test_baseline_recheck_interval_grows_and_resets(tmp_path):
    c = Clock()
    m = mm.ResourceMemory(tmp_path / "m.json", clock=c)
    ident = {"boot_id": "a", "kernel": "k", "os": "o"}
    assert m.baseline_due("pi1", ident)
    assert m.record_baseline("pi1", ident, {"cpu_count": 4, "jpeg_720p_fps": 100.0}) is True
    assert not m.baseline_due("pi1", ident)
    c.t += mm.DAY + 1
    assert m.baseline_due("pi1", ident)
    assert m.record_baseline("pi1", ident, {"cpu_count": 4, "jpeg_720p_fps": 108.0}) is False   # 벤치 ±15% → 같음
    assert m.node("pi1")["interval"] == 2 * mm.DAY
    for _ in range(10):
        c.t += m.node("pi1")["interval"] + 1
        m.record_baseline("pi1", ident, {"cpu_count": 4, "jpeg_720p_fps": 100.0})
    assert m.node("pi1")["interval"] == mm.RECHECK_MAX_S
    assert m.record_baseline("pi1", ident, {"cpu_count": 8, "jpeg_720p_fps": 100.0}) is True
    assert m.node("pi1")["interval"] == mm.RECHECK_MIN_S


def test_identity_change_forces_recheck(tmp_path):
    m = mm.ResourceMemory(tmp_path / "m.json", clock=Clock())
    m.record_baseline("pi1", {"boot_id": "a"}, {"cpu_count": 4})
    assert not m.baseline_due("pi1", {"boot_id": "a"})
    assert m.baseline_due("pi1", {"boot_id": "b"})                  # 재부팅


def test_save_load_roundtrip_is_atomic(tmp_path):
    p = tmp_path / "sub" / "m.json"
    m = mm.ResourceMemory(p, clock=Clock())
    m.record_baseline("pi1", {"boot_id": "a"}, {"cpu_count": 4})
    assert m.save() and not m.save()                                 # 바뀐 것 없으면 안 쓴다
    assert not list(p.parent.glob(".mem-*"))
    m2 = mm.ResourceMemory(p)
    assert m2.node("pi1")["baseline"] == {"cpu_count": 4}
    p.write_text("{broken", encoding="utf-8")
    assert mm.ResourceMemory(p).data["nodes"] == {}                   # 깨진 파일 → 빈 기억으로 시작


# ---------------------------------------------------------------- 로봇 서비스 몫
def test_service_share_uses_p95_after_enough_samples(tmp_path):
    m = mm.ResourceMemory(tmp_path / "m.json", clock=Clock())
    m.update_usage("pi7", 10, 500)
    s = m.service_share("pi7")
    assert (s["cpu_pct"], s["mem_mb"], s["active_measured"]) == (10, 500, False)   # 아직 p95 없음 → EWMA
    for i in range(100):
        m.update_usage("pi7", 10 + (40 if i % 20 == 0 else 0), 500)
    s = m.service_share("pi7")
    assert s["cpu_pct"] >= 10 and s["mem_mb"] == 500


def test_service_share_prefers_active_robot_load(tmp_path):
    m = mm.ResourceMemory(tmp_path / "m.json", clock=Clock())
    for _ in range(20):
        m.update_usage("pi7", 5, 400, mode="idle")                        # 로봇이 쉬는 동안
    s = m.service_share("pi7")
    assert s["active_measured"] is False and s["cpu_pct"] == 5 and s["mode"] == "idle"
    for _ in range(9):
        m.update_usage("pi7", 60, 1500, mode="active")                   # 로봇이 돌기 시작 — 아직 표본 9개
    assert m.service_share("pi7")["active_measured"] is False
    m.update_usage("pi7", 60, 1500, mode="active")
    s = m.service_share("pi7")
    assert s["active_measured"] and s["cpu_pct"] == 60 and s["mem_mb"] == 1500 and s["mode"] == "active"
    for _ in range(5):
        m.update_usage("pi7", 5, 400, mode="idle")                        # 다시 쉬어도 active 몫을 기억한다
    assert m.service_share("pi7")["cpu_pct"] == 60


# ---------------------------------------------------------------- 모델 조합
def test_combo_estimate_learning_measured(tmp_path):
    m = mm.ResourceMemory(tmp_path / "m.json", clock=Clock())
    key = mm.combo_key("cost", ["yoloe"], "h1", "gpu:RTX", "m")
    v, basis = m.combo(key, {"infer_ms_p50": 100.0})
    assert basis == "estimate" and v["infer_ms_p50"] == 120.0
    assert m.record_combo(key, {"infer_ms_p50": 40.0}, frames=50, seconds=10) == "learning"
    assert m.record_combo(key, {"infer_ms_p50": 60.0}, frames=50, seconds=10) == "learning"
    assert abs(m.combo(key)[0]["infer_ms_p50"] - 50.0) < 1e-6           # 학습 중에는 장 수 가중 평균
    assert m.record_combo(key, {"infer_ms_p50": 50.0}, frames=120, seconds=20) == "measured"
    m.record_combo(key, {"infer_ms_p50": 100.0}, frames=10, seconds=10)
    assert abs(m.combo(key)[0]["infer_ms_p50"] - 60.0) < 1e-6           # 실측 뒤 EWMA: 0.8 × 50 + 0.2 × 100
    assert m.combo(key)[1] == "measured"


def test_gpu_peak_only_decays_slowly(tmp_path):
    m = mm.ResourceMemory(tmp_path / "m.json", clock=Clock())
    key = mm.combo_key("gpu", ["river"], "h", "gpu:RTX")
    m.record_combo(key, {"gpu_mb_peak": 3000.0}, frames=300, seconds=70)
    m.record_combo(key, {"gpu_mb_peak": 1000.0}, frames=10, seconds=10)
    assert m.combo(key)[0]["gpu_mb_peak"] == 2400.0                  # max(3000 × 0.8, 1000)
    m.record_combo(key, {"gpu_mb_peak": 3500.0}, frames=10, seconds=10)
    assert m.combo(key)[0]["gpu_mb_peak"] == 3500.0


def test_combo_key_and_buckets():
    assert mm.res_bucket(464, 400) == "s" and mm.res_bucket(1280, 720) == "m"
    assert mm.res_bucket(1920, 1080) == "l" and mm.res_bucket(3840, 1920) == "xl"
    assert mm.combo_key("gpu", ["b", "a"], "h", "d") == "gpu|a+b|h|d"


def test_prune_keeps_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MAX_COMBOS", 3)
    c = Clock()
    m = mm.ResourceMemory(tmp_path / "m.json", clock=c)
    for i in range(5):
        c.t += 1
        m.record_combo("k%d" % i, {"x": 1.0}, 1, 1)
    assert sorted(m.data["combos"]) == ["k2", "k3", "k4"]


# ---------------------------------------------------------------- 주기
def test_poll_schedule():
    p = mm.PollSchedule()
    assert p.next(False, [10, 20]) == 60
    assert p.next(True, [10, 20]) == 10                                # 배치 중
    assert p.next(True, [11, 21]) == 20                                # 안 변함 → 두 배
    assert p.next(True, [11, 21]) == 40
    assert p.next(True, [11, 21]) == 60 and p.next(True, [11, 21]) == 60
    assert p.next(True, [30, 21]) == 10                                # 크게 변함
    assert p.next(True, [85, 21]) == 5                                 # 80% 넘음


# ---------------------------------------------------------------- 비전 보고 → 기억
TEL = {"schema": "vtel/1", "seq": 3, "window_s": 10.0, "models": ["yoloe"], "cfg_hash": "h", "device_class": "gpu:RTX",
       "gpu": {"proc_mb": None, "torch_peak_mb": 1350.0, "device_used_mb": 1600.0, "device_total_mb": 6141.0},
       "streams": {"ep-001": {"frames": 100, "infer_ms_p50": 35.0, "infer_ms_p90": 60.0, "fps": 10.0, "w": 1280, "h": 720},
                   "robot1": {"frames": 0}}}


def test_telemetry_records():
    rows = telemetry_records(TEL)
    assert rows[0][0] == "gpu|yoloe|h|gpu:RTX" and rows[0][2] == 100
    assert rows[0][1]["gpu_mb_peak"] == 1350.0 + 150.0 and rows[0][1]["gpu_basis"] == "torch+ctx"   # WSL: 프로세스별 값 없음
    assert rows[1][0] == "cost|yoloe|h|gpu:RTX|m" and rows[1][1]["infer_ms_p90"] == 60.0
    assert len(rows) == 2                                               # 장이 없는 기기는 빠진다


def test_process_gpu_prefers_process_value():
    from ds_resources.recorder import process_gpu_mb
    assert process_gpu_mb({"proc_mb": 900.0, "torch_peak_mb": 500.0}) == (900.0, "process")
    assert process_gpu_mb({"torch_peak_mb": 500.0}, 150) == (650.0, "torch+ctx")
    assert process_gpu_mb({}) == (None, None)


class FixedUsage:
    """실제 기기 부하를 읽지 않는다 — 부하가 80% 를 넘는 순간 주기가 5초(hot)가 되어 시험이 흔들렸다(2026-11-24)"""

    def sample(self, with_gpu=True):
        return {"cpu_pct": 20.0, "mem_pct": 40.0, "mem_mb": 6000.0, "temp_c": None, "gpu": []}


class FakeRedis:
    def __init__(self, kv):
        self.kv, self.rows = kv, []

    def scan_iter(self, pattern, count=None):
        return [k.encode() for k in self.kv if k.startswith(pattern.rstrip("*"))]

    def get(self, k):
        k = k.decode() if isinstance(k, bytes) else k
        return self.kv.get(k)

    def xadd(self, stream, fields, maxlen=None, approximate=None):
        self.rows.append((stream, fields))

    def xrevrange(self, key, hi, lo, count=None):
        return []                                                         # 소스 영상 없음


def test_recorder_tick_records_once_per_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "baseline", lambda bench=True: {"cpu_count": 4, "gpus": []})
    monkeypatch.setattr(probe, "gpus", lambda timeout=2: [])
    r = FakeRedis({"vision_telemetry:h:1": json.dumps(TEL)})
    rec = LocalRecorder(r, mm.ResourceMemory(tmp_path / "m.json"), node="wsl", log=lambda *_: None, bench=False)
    rec.usage = FixedUsage()
    wait = rec.tick()
    assert wait == 10                                                   # 비전이 돌고 있다 → 배치 중 주기
    assert [x[1]["change"] for x in r.rows if x[0] == NODE_STREAM] == ["added"]
    assert rec.mem.data["combos"]["cost|yoloe|h|gpu:RTX|m"]["frames"] == 100
    rec.tick()                                                          # 같은 seq — 다시 세지 않는다
    assert rec.mem.data["combos"]["cost|yoloe|h|gpu:RTX|m"]["frames"] == 100
    assert (tmp_path / "m.json").exists()


# ---------------------------------------------------------------- 단일 Writer · 노드 원천 (M7)
def test_memory_single_writer(tmp_path):
    a = mm.ResourceMemory(tmp_path / "m.json")
    b = mm.ResourceMemory(tmp_path / "m.json")
    assert a.writer and not b.writer
    a.record_baseline("pi1", {"boot_id": "x"}, {"cpu_count": 4})
    assert a.save() and b.save() is False                               # 읽기 전용은 쓰지 않는다
    assert b.node("pi1")["baseline"] == {"cpu_count": 4}                # 읽을 때 다시 읽는다


class FakeSource:
    def __init__(self, memory=None, log=None):
        self.rows, self.nets, self.want, self.fail = [], {}, [], False

    def samples(self):
        if self.fail:
            raise RuntimeError("k3s down")
        return [(n, dict(s)) for n, s in self.rows]

    def want_net(self, node, ip):
        self.want.append(node)

    def net_updates(self):
        d, self.nets = self.nets, {}
        return d


def S7(boot="a", ready=True, cpu=10.0):
    return {"role": "robot", "source": "k3s", "ready": ready, "cpu_pct": cpu, "mem_pct": 10.0, "mem_mb": 800.0,
            "identity": {"boot_id": boot}, "baseline": {"cpu_count": 4}, "ip": "100.0.0.1"}


def test_recorder_node_source_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "baseline", lambda bench=True: {"cpu_count": 20, "gpus": []})
    monkeypatch.setattr(probe, "gpus", lambda timeout=2: [])
    import sys, types
    mod = types.ModuleType("fake_src_mod")
    mod.FakeSource = FakeSource
    sys.modules["fake_src_mod"] = mod
    r = FakeRedis({})
    rec = LocalRecorder(r, mm.ResourceMemory(tmp_path / "m.json"), node="wsl", log=lambda *_: None, bench=False,
                        node_sources=["fake_src_mod:FakeSource"])
    src = rec.sources[0]
    src.rows = [("pi7", dict(S7(), source_name="robot1"))]
    rec.usage = FixedUsage()
    rec.tick()
    assert rec.mem.node("pi7")["usage"]["mode"] == "idle"                # 영상 없음 → 로봇 서비스 쉼
    changes = [(x[1]["node"], x[1]["change"]) for x in r.rows]
    assert ("pi7", "added") in changes and src.want == ["pi7"]
    assert rec.mem.node("pi7")["baseline"] == {"cpu_count": 4}
    rec.tick()
    assert src.want == ["pi7"]                                          # 같은 신원 — 다시 재지 않는다
    src.rows = [("pi7", S7(boot="b", ready=False))]                     # 재부팅 중
    src.nets = {"pi7": {"latency_ms": 6.0, "relay": False, "bw_mbps": 80.0}}
    rec.tick()
    assert rec.mem.node("pi7")["identity"] == {"boot_id": "b"} and src.want == ["pi7", "pi7"]
    assert rec.mem.node("pi7")["net"]["bw_mbps"] == 80.0
    assert ("pi7", "not_ready") in [(x[1]["node"], x[1]["change"]) for x in r.rows]
    src.rows = []
    rec.tick()
    assert ("pi7", "removed") in [(x[1]["node"], x[1]["change"]) for x in r.rows]
    src.fail = True
    rec.tick()                                                          # 원천이 죽어도 로컬은 돈다
    assert "k3s down" in rec.last_error


def test_recorder_bad_source_path_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "gpus", lambda timeout=2: [])
    rec = LocalRecorder(FakeRedis({}), mm.ResourceMemory(tmp_path / "m.json"), node="wsl", log=lambda *_: None,
                        bench=False, node_sources=["no_such_module:X"])
    assert rec.sources == []


def test_robot_becomes_active_when_its_video_arrives(tmp_path, monkeypatch):
    import time as _t
    monkeypatch.setattr(probe, "gpus", lambda timeout=2: [])
    import sys, types
    mod = types.ModuleType("fake_src_mod2")
    mod.FakeSource = FakeSource
    sys.modules["fake_src_mod2"] = mod

    class R(FakeRedis):
        def __init__(self):
            super().__init__({})
            self.last = None

        def xrevrange(self, key, hi, lo, count=None):
            return [] if self.last is None else [(("%d-0" % self.last).encode(), {})]
    r = R()
    rec = LocalRecorder(r, mm.ResourceMemory(tmp_path / "m.json"), node="wsl", log=lambda *_: None, bench=False,
                        node_sources=["fake_src_mod2:FakeSource"])
    rec.sources[0].rows = [("pi7", dict(S7(), source_name="robot1"))]
    rec.usage = FixedUsage()
    assert rec.tick() == 60                                              # 영상 없음 — 쉼 주기
    r.last = int(_t.time() * 1000)                                       # 로봇 영상이 들어온다
    assert rec.tick() == 10                                              # 바로 배치 중 주기로
    assert rec.mem.node("pi7")["usage"]["mode"] == "active"
    assert ("pi7", "service") in [(x[1]["node"], x[1]["change"]) for x in r.rows]
