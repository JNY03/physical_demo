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
    assert m.service_share("pi7") == (10, 500)                       # 아직 p95 없음 → EWMA
    for i in range(100):
        m.update_usage("pi7", 10 + (40 if i % 20 == 0 else 0), 500)
    cpu, mem = m.service_share("pi7")
    assert cpu >= 10 and mem == 500


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


def test_recorder_tick_records_once_per_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "baseline", lambda bench=True: {"cpu_count": 4, "gpus": []})
    monkeypatch.setattr(probe, "gpus", lambda timeout=2: [])
    r = FakeRedis({"vision_telemetry:h:1": json.dumps(TEL)})
    rec = LocalRecorder(r, mm.ResourceMemory(tmp_path / "m.json"), node="wsl", log=lambda *_: None, bench=False)
    rec.usage.sample()
    wait = rec.tick()
    assert wait == 10                                                   # 비전이 돌고 있다 → 배치 중 주기
    assert [x[1]["change"] for x in r.rows if x[0] == NODE_STREAM] == ["added"]
    assert rec.mem.data["combos"]["cost|yoloe|h|gpu:RTX|m"]["frames"] == 100
    rec.tick()                                                          # 같은 seq — 다시 세지 않는다
    assert rec.mem.data["combos"]["cost|yoloe|h|gpu:RTX|m"]["frames"] == 100
    assert (tmp_path / "m.json").exists()
