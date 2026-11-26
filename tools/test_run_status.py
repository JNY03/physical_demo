"""run_status — 런처 「실행 상태」 칸이 읽는 한 장 (2026-11-27). 가짜 /proc · /health · Redis · 상태 파일로 돈다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_status as rs  # noqa: E402

LOCAL = "book-test"
SMI = "NVIDIA GeForce RTX 4050 Laptop GPU, 3277, 6141, 45\n"


class FakeRedis:
    def __init__(self, kv=None, orch=None):
        self.kv = kv or {}
        self.orch = orch or []            # [(id, fields)] 오래된 것 → 새것

    def scan_iter(self, pat, count=100):
        pre = pat.rstrip("*")
        return [k for k in self.kv if k.startswith(pre)]

    def get(self, k):
        return self.kv.get(k)

    def xrevrange(self, name, count=10):
        return list(reversed(self.orch))[:count] if name == rs.ORCH_STREAM else []


def rec(i, event, **f):
    d = {"schema": "orch/1", "event": event, "request_id": f.pop("rid", "req-1"), "reason": f.pop("reason", ""),
         "ts": "2026-11-27T10:00:%02d.000+09:00" % i}
    for k in ("detail", "plan", "request"):
        if k in f:
            d[k] = json.dumps(f.pop(k), ensure_ascii=False)
    d.update(f)
    return ("17956900000%02d-0" % i, d)


def fake(monkeypatch, tmp_path, procs=(), health=None, redis=None, units=None, smi=SMI, ports=None):
    monkeypatch.setattr(rs, "_procs", lambda: list(procs))
    monkeypatch.setattr(rs, "_health", lambda port, timeout: (health or {}).get(port))
    monkeypatch.setattr(rs, "_ports", lambda: dict(ports or {"robot1": 10003, "pi12": 10009, "a2": 10007}))
    monkeypatch.setattr(rs, "_redis", lambda url: redis)
    monkeypatch.setattr(rs, "_state_units", lambda: dict(units or {}))
    monkeypatch.setattr(rs, "_nvidia_smi", lambda: smi)
    monkeypatch.setattr(rs, "_local_node", lambda: LOCAL)
    monkeypatch.setattr(rs, "_is_wsl", lambda: True)
    return tmp_path / "prev.json"


SERVER = (101, ["./venv/bin/python", "server_stream_multi_source.py", "--on", "robot1", "pi12"])
ORCH = (102, ["./venv/bin/python", "-m", "ds_resources.cli", "run", "--sources", "robot1"])
VISION = (103, ["./venv/bin/python", "vision/vision_infer.py", "--devices", "robot1"])
WRAPPER = (100, ["bash", "tools/ds_run.sh", "ds_server", "./venv/bin/python", "server_stream_multi_source.py"])


def test_nothing_running_gives_empty_state(monkeypatch, tmp_path):
    prev = fake(monkeypatch, tmp_path, smi=None)
    s = rs.snapshot(prev_path=prev, now=1000.0)
    assert s["active"] is False and s["sources"] == [] and s["processes"] == []
    assert s["orchestrator"]["running"] is False and s["summary"] == "실행 중인 것 없음"
    assert "nvidia-smi" in s["gpu_text"]
    json.dumps(s, ensure_ascii=False)                       # 런처가 읽는다 — 직렬화가 되어야 한다


def test_processes_skip_shell_wrappers():
    p = rs.processes([WRAPPER, SERVER, ORCH, VISION, (104, ["python3", "other.py"])])
    assert [(x["role"], x["pid"]) for x in p] == [("data_server", 101), ("orchestrator", 102), ("vision", 103)]
    assert p[0]["role_ko"] == "데이터 서버"


def test_live_source_with_fps_model_where_and_gpu(monkeypatch, tmp_path):
    tel = {"schema": "vtel/1", "models": ["targeted"], "streams": {"robot1": {"frames": 21, "fps": 2.07}}}
    r = FakeRedis(kv={"vision_telemetry:%s:103" % LOCAL: json.dumps(tel)},
                  orch=[rec(1, "requested", request={"sources": ["robot1", "a2"]}),
                        rec(2, "planned", plan={"waiting": [{"item": "input:a2", "code": "input_wait",
                                                             "reason": "입력 대기 — 영상 없음"}]}),
                        rec(3, "running", detail={"sources_up": ["robot1"], "unit_id": "gpu-1"},
                            reason="처리 중: targeted")])
    units = {"gpu-1": {"kind": "gpu", "status": "loaded", "devices": ["robot1"], "models": ["targeted"], "node": LOCAL}}
    prev = fake(monkeypatch, tmp_path, procs=[SERVER, ORCH, VISION], redis=r, units=units,
                health={10003: {"frames": 100, "vision": {"targeted": {"age_s": 1.0}}}})
    first = rs.snapshot(prev_path=prev, now=1000.0)
    row = {x["source"]: x for x in first["sources"]}
    assert set(row) == {"robot1", "a2"}                     # a2 는 요청에 있지만 서버가 안 받는다
    assert row["robot1"]["state"] == "live" and row["robot1"]["in_fps"] is None   # 첫 호출은 셈할 지난 값이 없다
    assert row["robot1"]["proc_fps"] == 2.07 and row["robot1"]["models"] == ["targeted"]
    assert row["robot1"]["where"] == "book-test GPU (NVIDIA GeForce RTX 4050 Laptop GPU) — 이 기기 · WSL"
    assert row["robot1"]["reason_kind"] == "ok"
    assert row["a2"]["state"] == "down" and row["a2"]["reason"].startswith("대기: 입력 대기")
    assert first["gpu"][0]["used_mb"] == 3277 and "3.2 / 6.0 GB" in first["gpu_text"]
    assert first["orchestrator"]["running"] and first["orchestrator"]["request_id"] == "req-1"

    monkeypatch.setattr(rs, "_health", lambda port, timeout: {10003: {"frames": 130}}.get(port))
    second = rs.snapshot(prev_path=prev, now=1003.0)
    row = {x["source"]: x for x in second["sources"]}
    assert row["robot1"]["in_fps"] == 10.0 and row["robot1"]["in_fps_text"] == "10.0"

    monkeypatch.setattr(rs, "_health", lambda port, timeout: {10003: {"frames": 130}}.get(port))
    third = rs.snapshot(prev_path=prev, now=1006.0)          # 새 장이 없다 → 연결은 됐지만 안 들어옴
    assert {x["source"]: x for x in third["sources"]}["robot1"]["state"] == "ready"


def test_last_reasons_refused_degraded_failed_in_plain_korean():
    units = {"cpu-x": {"kind": "cpu", "task": "targeted_detect.device@pi12"}}
    recs = [r for _, r in [
        rec(1, "requested"),
        rec(2, "planned", plan={"refused": [{"item": "gpu:targeted:drone", "code": "gpu_throughput", "reason": ""}],
                                "degraded": [{"item": "gpu:yoloe:robot1", "code": "fps_lowered", "reason": "fps 상한 5 로"}]}),
        rec(3, "failed", code="cpu_launch_failed", reason="CPU 작업을 못 띄움", detail={"unit_id": "cpu-x"}),
        rec(4, "degraded", reason="입력 끊김: a2 — 들어오면 바로 다시 처리", detail={"source_down": "a2"}),
    ]]
    out = rs.last_reasons(recs, units)
    assert out["drone"]["kind"] == "refused" and out["drone"]["text"] == "거절: " + rs.REASON_CODES["gpu_throughput"]
    assert out["robot1"]["text"] == "낮춤: fps 상한 5 로"
    assert out["pi12"]["kind"] == "failed" and out["pi12"]["code"] == "cpu_launch_failed"
    assert out["a2"]["kind"] == "degraded" and "입력 끊김" in out["a2"]["text"]


def test_stale_units_hidden_when_orchestrator_is_off(monkeypatch, tmp_path):
    # 지난 실행이 남긴 「loaded」 단위 · 기록은 오케스트레이터가 꺼져 있으면 보이지 않는다
    units = {"gpu-old": {"kind": "gpu", "status": "loaded", "devices": ["pi12"], "models": ["yoloe"], "node": LOCAL}}
    r = FakeRedis(orch=[rec(1, "requested", request={"sources": ["pi12"]}), rec(2, "stopped", reason="오케스트레이터 끝남")])
    prev = fake(monkeypatch, tmp_path, redis=r, units=units)
    s = rs.snapshot(prev_path=prev, now=1000.0)
    assert s["sources"] == [] and s["orchestrator"]["last_event"] == ""


def test_item_source_and_main_out(monkeypatch, tmp_path, capsys):
    assert rs.item_source("input:a2") == "a2" and rs.item_source("gpu:targeted:robot1") == "robot1"
    assert rs.item_source("targeted_detect.device@pi12") == "pi12" and rs.item_source("") == ""
    fake(monkeypatch, tmp_path)
    monkeypatch.setattr(rs, "PREV", tmp_path / "p.json")
    out = tmp_path / "o" / "run_status.json"
    assert rs.main(["--json", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text(encoding="utf-8")) == printed and printed["schema"] == "run_status/1"
