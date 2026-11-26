"""vision/device_stage.py — 기기 단계 실행체 (가짜 Redis · 가짜 실행기 · 로컬 MJPEG 서버로, 모델 없이)"""
import json
import sys
import threading
import time
import types
from http import server
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "docker" / "device-runner"))
import device_stage as ds  # noqa: E402

JPG = cv2.imencode(".jpg", np.full((240, 320, 3), 90, np.uint8))[1].tobytes()


class FakeRedis:
    def __init__(self):
        self.streams, self.kv, self.ttl = {}, {}, {}

    def put(self, key, fields):
        rows = self.streams.setdefault(key, [])
        eid = "%d-%d" % (int(time.time() * 1000), len(rows))
        rows.append((eid.encode(), fields))
        return eid

    def xread(self, streams, count=None, block=None):
        (key, last), = streams.items()
        rows = self.streams.get(key) or []
        if last == "$":                                   # 시험: 「다음에 오는 것」 = 아직 안 읽은 가장 오래된 것
            rows = [r for r in rows if not r[1].get(b"_read")]
        if not rows:
            return []
        rows[0][1][b"_read"] = True
        return [(key.encode(), [rows[0]])]

    def xadd(self, key, fields, maxlen=None, approximate=None):
        return self.put(key, fields)

    def set(self, k, v, ex=None):
        self.kv[k], self.ttl[k] = v, ex


class FakeDetector:
    kind = "detect"

    def __init__(self, n=30):
        self.n, self.calls = n, 0

    def __call__(self, bgr):
        self.calls += 1
        dets = [{"cls": i % 3, "name": "c%d" % (i % 3), "conf": i / 100.0, "xyxy": [i, i, i + 10, i + 10],
                 "_poly": np.zeros((5, 2))} for i in range(self.n)]
        return {"infer_ms": 1.0}, dets


def args(**kw):
    a = dict(source="cam1", function="object_detect", fps=9, top_k=5, max_age=0, maxlen=100, unit_id=None,
             max_frames=0)
    a.update(kw)
    return types.SimpleNamespace(**a)


# ---------------------------------------------------------------- 작은 함수
def test_fps_is_clamped_to_the_device_contract():
    assert ds.clamp_fps(9) == 5.0 and ds.clamp_fps(0.2) == 1.0 and ds.clamp_fps("x") == 2.0 and ds.clamp_fps(2) == 2.0


def test_threads_follow_cpu_m():
    assert ds.threads_for(1600) == 2 and ds.threads_for(400) == 1 and ds.threads_for(9000) == 4 and ds.threads_for(None) == 2


def test_resolve_paths_only_touches_models_and_external(tmp_path):
    cfg = {"model": "models/a/b.onnx", "src": "box_height", "l": ["external/x/y", "/abs/p"], "n": 3}
    out = ds.resolve_paths(cfg, tmp_path)
    assert out == {"model": str(tmp_path / "models/a/b.onnx"), "src": "box_height",
                   "l": [str(tmp_path / "external/x/y"), "/abs/p"], "n": 3}


def test_camera_url_auto_only_on_the_cameras_own_node():
    assert ds.pick_camera_url("auto", "pi13", {"NODE_NAME": "pi13", "HOST_IP": "10.0.0.5"}) == \
        "http://10.0.0.5:8090/stream/1"
    assert ds.pick_camera_url("auto", "robot1", {"NODE_NAME": "pi7", "DS_SOURCE_NODE": "pi7"}) == \
        "http://127.0.0.1:8090/stream/1"
    assert ds.pick_camera_url("auto", "pi13", {"NODE_NAME": "pi6", "DS_SOURCE_NODE": "pi13"}) is None   # 남의 카메라
    assert ds.pick_camera_url("auto", "pi13", {}) is None
    assert ds.pick_camera_url("none", "pi13", {"NODE_NAME": "pi13"}) is None
    assert ds.pick_camera_url("http://x/s", "pi13", {}) == "http://x/s"
    assert ds.pick_camera_url("auto", "pi13", {"DS_CAMERA_URL": "http://y/s"}) == "http://y/s"


def test_compact_dets_keeps_top_k_and_drops_private_fields():
    _, dets = FakeDetector()(None)
    out = ds.compact_dets(dets, 3, "detect")
    assert [d["conf"] for d in out] == [0.29, 0.28, 0.27]
    assert set(out[0]) == {"cls", "name", "conf", "xyxy"}
    poly = [{"cls": 0, "name": "a", "conf": 0.9, "xyxy": [0, 0, 1, 1], "_poly": np.zeros((100, 2))}]
    seg = ds.compact_dets(poly, 3, "segment")
    assert len(seg[0]["polygon"]) <= ds.POLY_MAX


def test_depth_summary_is_small_and_ignores_invalid_pixels():
    d = np.linspace(1, 5, 60 * 80, dtype=np.float32).reshape(60, 80)
    d[:10] = 0
    s = ds.depth_summary(d)
    assert 0.8 < s["valid_ratio"] < 0.85 and s["min"] > 1.0 and len(s["grid"]) == 6 and len(s["grid"][0]) == 8
    assert s["grid"][0][0] is None                                     # 무효 픽셀만 있는 칸
    assert ds.depth_summary(np.zeros((4, 4))) == {"valid_ratio": 0.0}


def test_unknown_model_key_falls_back_to_runner_file_or_stops(tmp_path):
    reg = tmp_path / "r.yaml"
    reg.write_text("models:\n  m1: {runner: 'x:Y', config: {a: 1}}\n", encoding="utf-8")
    vc = tmp_path / "v.yaml"
    vc.write_text("models: {}\n", encoding="utf-8")
    assert ds.model_spec("m1", reg, vc) == ({"runner": "x:Y", "config": {"a": 1}}, {"a": 1})
    with pytest.raises(SystemExit):
        ds.model_spec("no_such_model_zz", reg, vc)


def test_no_spin_wraps_onnxruntime_sessions(monkeypatch):
    """onnxruntime 세션에 spin 끄기 설정이 들어간다 — 실행기 코드는 그대로 (SessionOptions 를 넘겨도, 안 넘겨도)"""
    ort = pytest.importorskip("onnxruntime")
    seen = []

    class FakeSession:
        def __init__(self, path, so=None, providers=None):
            seen.append(so)

    class FakeOpts:
        def __init__(self):
            self.kv = {}

        def add_session_config_entry(self, k, v):
            self.kv[k] = v
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(ort, "SessionOptions", FakeOpts)
    assert ds.no_spin_onnxruntime() is True
    ort.InferenceSession("m.onnx", FakeOpts(), providers=["CPUExecutionProvider"])
    ort.InferenceSession("m.onnx")
    assert [s.kv["session.intra_op.allow_spinning"] for s in seen] == ["0", "0"]
    assert ds.no_spin_onnxruntime() is True and ort.InferenceSession.__mro__[1] is FakeSession   # 두 번 감싸지 않는다


# ---------------------------------------------------------------- 돌리기
def test_one_frame_from_redis_becomes_one_compact_record_and_a_heartbeat():
    r = FakeRedis()
    r.put("camera_stream:cam1", {b"image": JPG, b"header": json.dumps({"n": 41, "sender": {"sequence_id": 7}}).encode()})
    det = FakeDetector()
    st = ds.DeviceStage(r, det, args(max_frames=1), "tiny", "detect", None)
    st.run()
    rows = r.streams["device_stage:cam1:object_detect"]
    assert len(rows) == 1 and det.calls == 1
    rec = json.loads(rows[0][1]["result"])
    assert rec["schema"] == "dstage/1" and rec["frame_seq"] == 7 and rec["via"] == "redis"
    assert rec["n"] == 30 and len(rec["dets"]) == 5 and rec["width"] == 320 and rec["fps_target"] == 5.0
    assert "image" not in rows[0][1] and len(rows[0][1]["result"]) < 2000            # 프레임은 내지 않는다
    hb = json.loads(r.kv["cpu_telemetry:object_detect.device@cam1"])
    assert hb["frames"] == 1 and hb["model"] == "tiny" and abs(hb["ts"] - time.time()) < 5
    assert r.ttl["cpu_telemetry:object_detect.device@cam1"] == ds.TELEMETRY_TTL_S


def test_unit_id_from_k3s_names_the_heartbeat(monkeypatch):
    monkeypatch.setenv("UNIT_ID", "cpu-object_detect.device@cam1-ab12")
    r = FakeRedis()
    st = ds.DeviceStage(r, FakeDetector(), args(), "tiny", "detect", None)
    st.heartbeat(force=True)
    assert "cpu_telemetry:cpu-object_detect.device@cam1-ab12" in r.kv


def test_depth_runner_publishes_stats_not_the_map():
    class Depth:
        kind = "depth"

        def __call__(self, bgr):
            return {"infer_ms": 2.0}, np.ones(bgr.shape[:2], np.float32), np.eye(3)
    r = FakeRedis()
    r.put("camera_stream:cam1", {b"image": JPG, b"header": b"{}"})
    ds.DeviceStage(r, Depth(), args(function="depth_estimate", max_frames=1), "d", "depth", None).run()
    rec = json.loads(r.streams["device_stage:cam1:depth_estimate"][0][1]["result"])
    assert rec["depth"]["median"] == 1.0 and rec["dets"] == []


# ---------------------------------------------------------------- 카메라 HTTP (MJPEG)
def mjpeg_server(frames):
    class H(server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for i in range(frames):
                    self.wfile.write(("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n"
                                      "X-Frame-Seq: %d\r\nX-Timestamp: 1700000000.5\r\n\r\n" % (len(JPG), i + 1)).encode())
                    self.wfile.write(JPG + b"\r\n")
                    time.sleep(0.02)
                time.sleep(3)
            except (BrokenPipeError, ConnectionResetError):
                pass
    s = server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, "http://127.0.0.1:%d/stream/1" % s.server_address[1]


def test_mjpeg_reader_keeps_only_the_latest_frame():
    s, url = mjpeg_server(5)
    try:
        rd = ds.MjpegReader(url)
        deadline = time.time() + 3
        while rd.got < 5 and time.time() < deadline:
            time.sleep(0.02)
        jpeg, seq, ts, _ = rd.take()
        assert jpeg == JPG and seq == "5" and ts == "1700000000.5" and rd.take() is None
        rd.close()
    finally:
        s.shutdown()


def test_http_frames_are_used_on_the_cameras_node():
    s, url = mjpeg_server(3)
    try:
        r = FakeRedis()
        st = ds.DeviceStage(r, FakeDetector(), args(max_frames=1), "tiny", "detect", url)
        st.run()
        rec = json.loads(r.streams["device_stage:cam1:object_detect"][0][1]["result"])
        assert rec["via"] == "http" and isinstance(rec["frame_seq"], int) and rec["frame_ts"] == 1700000000.5
    finally:
        s.shutdown()


def test_dead_camera_url_falls_back_to_redis():
    r = FakeRedis()
    r.put("camera_stream:cam1", {b"image": JPG, b"header": b"{}"})
    inp = ds.FrameInput(r, "cam1", "http://127.0.0.1:9/stream/1")
    deadline = time.time() + 5
    fr = None
    while fr is None and time.time() < deadline:
        fr = inp.next()
        time.sleep(0.05)
    assert fr is not None and fr["via"] == "redis" and inp.http is None


# ---------------------------------------------------------------- 이미지 스테이징 (docker/device-runner/stage.py)
def test_stage_model_paths_from_registry_files_and_config():
    import stage
    reg = {"models": {"m": {"files": ["models/m/w.bin"], "config": {"model": "models/m/dir", "x": "y"}}}}
    assert stage.model_paths("m", reg, {}) == ["models/m/w.bin", "models/m/dir"]
    assert stage.model_paths("m", reg, {"models": {"m": {"weight": "models/m/other.onnx"}}}) == \
        ["models/m/w.bin", "models/m/other.onnx"]                     # vision_config 가 매개변수를 이긴다
    assert stage.model_paths("models/raw/f.onnx", reg, {}) == ["models/raw/f.onnx"]
    with pytest.raises(KeyError):
        stage.model_paths("nope", reg, {})


def test_stage_builds_a_context_with_runners_and_listed_weights(tmp_path, monkeypatch):
    import stage
    home = tmp_path / "home"
    (home / "models" / "m").mkdir(parents=True)
    (home / "models" / "m" / "w.bin").write_bytes(b"x")
    monkeypatch.setenv("DS_HOME", str(home))
    man = tmp_path / "models.txt"
    man.write_text("# 주석\nmodels/m/w.bin\nmodels/m/missing.bin\n", encoding="utf-8")
    out = tmp_path / "ctx"
    added, missing = stage.stage(out, man, log=lambda *_: None)
    assert added == ["models/m/w.bin"] and len(missing) == 1
    assert (out / "models/m/w.bin").exists() and (out / "vision/device_stage.py").exists()
    assert (out / "vision/runners/__init__.py").exists() and (out / "requirements.txt").exists()
    assert not list(out.rglob("__pycache__"))
