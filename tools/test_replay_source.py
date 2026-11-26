"""replay_source — 녹화 장으로 소스를 흉내 낸다 (2026-11-26). 가짜 Redis + 진짜 로컬 Redis(있을 때만) 짧은 실행."""
import http.client
import json
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import replay_source as rs  # noqa: E402
import server_port_override as spo  # noqa: E402

JPG = b"\xff\xd8\xff\xe0" + b"x" * 60 + b"\xff\xd9"


class FakeRedis:
    def __init__(self):
        self.rows, self.deleted = [], []

    def xadd(self, key, fields, maxlen=None, approximate=None):
        self.rows.append((key, fields, maxlen))

    def xlen(self, key):
        return sum(1 for k, _, _ in self.rows if k == key)

    def delete(self, key):
        self.deleted.append(key)
        return 1


def make_run(root, src="pi12", run="20261126_120000", n=3, sender=None, states=()):
    d = root / src / run
    (d / "frames").mkdir(parents=True)
    with open(d / "frames.jsonl", "w") as f:
        for i in range(1, n + 1):
            (d / "frames" / ("%09d.jpg" % i)).write_bytes(JPG + bytes([i]))
            s = dict(sender or {"Content-Type": "image/jpeg", "Content-Length": "1", "X-Frame-Seq": "900",
                                "X-Timestamp": "1795690912.059698"})
            f.write(json.dumps({"stream": src, "n": i, "frame_file": "%s/%s/frames/%09d.jpg" % (src, run, i),
                                "received_at": "2026-11-26T12:00:0%d.000+09:00" % i, "sender": s}) + "\n")
    with open(d / "states.jsonl", "w") as f:
        for k, (at, topic, payload) in enumerate(states, 1):
            f.write(json.dumps({"stream": src, "n": k, "topic": topic, "qos": 1, "retain": False,
                                "received_at": at, "payload": payload, "enc": "json"}) + "\n")
    return d


# ---------------------------------------------------------------- sender 고치기
def test_refresh_sender_picam_and_robot_forms():
    now = 1795700000.25
    pic = rs.refresh_sender({"Content-Type": "image/jpeg", "Content-Length": "9", "X-Frame-Seq": "158252",
                             "X-Timestamp": "1795690912.059698"}, 7, now, JPG)
    assert pic["X-Frame-Seq"] == "7" and pic["X-Timestamp"] == "1795700000.250000"
    assert pic["Content-Length"] == str(len(JPG))
    rob = rs.refresh_sender({"X-Frame-Seq": "1", "X-Timestamp": "1795690912108", "X-Timestamp-Iso": "x",
                             "X-Image-Bytes": "1", "X-Sha1": "f07490257fc7d95c", "X-Node-Id": "pi7"}, 3, now, JPG)
    assert rob["X-Timestamp"] == "1795700000250"                      # 밀리초 꼴은 밀리초로
    assert rob["X-Timestamp-Iso"].startswith("2026-") and rob["X-Timestamp-Iso"].endswith("+09:00")
    assert rob["X-Image-Bytes"] == str(len(JPG)) and len(rob["X-Sha1"]) == 16 and rob["X-Node-Id"] == "pi7"
    v2 = rs.refresh_sender({"camera_id": "d455", "sequence_id": 5, "timestamp": "2026-11-16T14:07:27.000+09:00",
                            "width": 1280, "height": 720}, 11, now, JPG, size=(640, 360))
    assert v2["sequence_id"] == 11 and v2["timestamp"] != "2026-11-16T14:07:27.000+09:00"
    assert (v2["width"], v2["height"]) == (640, 360) and v2["camera_id"] == "d455"
    assert rs.refresh_sender({}, 2, now, JPG)["X-Frame-Seq"] == "2"   # 옛 폴더 — picam 꼴로


def test_frame_fields_matches_server():
    srv = pytest.importorskip("server_stream_multi_source")
    h = {"stream": "pi12", "n": 1, "sender": {"X-Frame-Seq": "1", "image": "x", "meta": {"a": 1}, "num": 3}}
    assert rs.frame_fields(JPG, h) == srv.frame_fields(JPG, h)


# ---------------------------------------------------------------- 녹화 읽기 · redis 모드
def test_find_run_latest_and_recording(tmp_path):
    make_run(tmp_path, run="20261126_100000", n=2)
    make_run(tmp_path, run="20261126_120000", n=3)
    (tmp_path / "pi12" / "sample").mkdir()
    (tmp_path / "pi12" / "sample" / "a.jpg").write_bytes(JPG)         # 손으로 모은 폴더는 뒤로
    assert rs.find_run("pi12", "latest", tmp_path).name == "20261126_120000"
    assert rs.find_run("pi12", "20261126_100000", tmp_path).name == "20261126_100000"
    old = rs.Recording(tmp_path / "pi12" / "sample", tmp_path)          # frames.jsonl 없는 폴더도 읽는다
    assert len(old.frames) == 1 and old.frames[0][1] == {}
    with pytest.raises(SystemExit):
        rs.find_run("nope", "latest", tmp_path)


def test_redis_mode_writes_server_fields_and_states_in_order(tmp_path):
    states = [("2026-11-26T12:00:00.500+09:00", "zoneA/robot/go1-001/status",
               {"status": "online", "timestamp": "2026-11-26T12:00:00.400+09:00"}),
              ("2026-11-26T12:00:02.500+09:00", "zoneA/robot/go1-001/state", {"x": 1})]
    d = make_run(tmp_path, src="robot1", n=3, states=states)
    rec = rs.Recording(d, tmp_path)
    assert [len(x) for x in rec.states_before] == [1, 0, 1, 0]
    r = FakeRedis()
    p = rs.Player("robot1", "replay-test-robot1", rec, "redis", fps=50, loop=True, redis=r, maxlen=300,
                  run_label="robot1/" + d.name)
    for i in (0, 1, 2, 0):                                              # 돌아와도 번호는 이어 늘어난다
        p.emit(i)
    cams = [f for k, f, _ in r.rows if k == "camera_stream:replay-test-robot1"]
    sts = [f for k, f, _ in r.rows if k == "state_stream:replay-test-robot1"]
    assert len(cams) == 4 and len(sts) == 3 and all(m == 300 for _, _, m in r.rows)
    h = json.loads(cams[-1]["header"])
    assert h["n"] == 4 and h["stream"] == "replay-test-robot1" and h["sender"]["X-Frame-Seq"] == "4"
    assert set(cams[0]) == {"image", "header", "Content-Type", "Content-Length", "X-Frame-Seq", "X-Timestamp"}
    assert cams[0]["image"] == JPG + b"\x01" and h["frame_file"].startswith("robot1/")
    assert h["received_at"].endswith("+09:00") and h["via"].startswith("replay:")
    sp = json.loads(sts[0]["payload"])
    assert sp["status"] == "online" and sp["timestamp"] != "2026-11-26T12:00:00.400+09:00"   # 지금 시각
    assert json.loads(sts[0]["header"])["topic"] == "zoneA/robot/go1-001/status"


def test_trim_only_test_names():
    r = FakeRedis()
    gone = rs.trim(r, ["replay-test-pi12", "pi12"], {"pi12", "robot1"})
    assert gone == ["camera_stream:replay-test-pi12", "state_stream:replay-test-pi12"]


# ---------------------------------------------------------------- mjpeg 모드
def test_mjpeg_part_keeps_part_headers_or_frame_meta():
    a = rs.mjpeg_part(JPG, {"Content-Type": "image/jpeg", "X-Frame-Seq": "3", "X-Timestamp": "1.5"})
    assert b"X-Frame-Seq: 3\r\n" in a and b"Frame-Meta" not in a and a.endswith(JPG + b"\r\n")
    b = rs.mjpeg_part(JPG, {"camera_id": "d455", "sequence_id": 3})
    assert b'Frame-Meta: {"camera_id":"d455","sequence_id":3}\r\n' in b


def test_mjpeg_endpoint_is_read_by_server_reader(tmp_path):
    srv = pytest.importorskip("server_stream_multi_source")
    d = make_run(tmp_path, n=3)
    latest, stop = rs.Latest(), threading.Event()
    httpd = rs.start_http("127.0.0.1", 0, rs.make_handler("pi12", latest, stop))
    port = httpd.server_address[1]
    p = rs.Player("pi12", "pi12", rs.Recording(d, tmp_path), "mjpeg", fps=20, loop=True, latest=latest, stop=stop)
    p.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/stream/1")
        resp = conn.getresponse()
        boundary = srv.content_type_param(resp.getheader("Content-Type"), "boundary")
        got = []
        for headers, data in srv.read_mjpeg(resp, boundary):            # 서버의 robot_video_loop 가 읽는 그대로
            got.append((headers, data))
            if len(got) == 2:
                break
        conn.close()
        assert got[0][1].startswith(b"\xff\xd8") and int(got[1][0]["X-Frame-Seq"]) > int(got[0][0]["X-Frame-Seq"])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        h = json.loads(conn.getresponse().read())
        assert h["source"] == "pi12" and h["frames"] >= 2
    finally:
        stop.set()
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------- 서버 포트 바꾸기
def test_server_port_override_split(monkeypatch):
    monkeypatch.setenv("PD_VIDEO_PORT_A2", "18093")
    monkeypatch.setenv("PD_VIDEO_PORT_PI12", "1")                       # 주소 칸의 :포트 가 이긴다
    argv, ports = spo.split_overrides(
        ["--on", "pi12", "robot1", "--pi12", "127.0.0.1:18090", "--robot1=127.0.0.1:18091",
         "--video-port", "pi13=18092", "--save", "/tmp/x"], ["pi12", "robot1", "pi13", "a2"])
    assert argv == ["--on", "pi12", "robot1", "--pi12", "127.0.0.1", "--robot1", "127.0.0.1", "--save", "/tmp/x"]
    assert ports == {"pi12": 18090, "robot1": 18091, "pi13": 18092, "a2": 18093}


def test_server_port_override_patch_routes_by_source():
    calls = []

    class Mod:
        ROBOT_VIDEO_PORT = 8090

        @staticmethod
        def robot_video_loop(src, host, cam, port=8090, path=None):
            calls.append((src.name, host, port, path))

        @staticmethod
        def log(msg):
            pass

    spo.patch(Mod, {"pi12": 18090})
    Src = type("S", (), {})
    a, b = Src(), Src()
    a.name, b.name = "pi12", "drone"
    Mod.robot_video_loop(a, "127.0.0.1", 1)
    Mod.robot_video_loop(b, "h", 0, 8080, "/cam0.mjpg")                 # 다른 포트를 준 부름은 그대로
    assert calls == [("pi12", "127.0.0.1", 18090, None), ("drone", "h", 8080, "/cam0.mjpg")]


# ---------------------------------------------------------------- 진짜 로컬 Redis (있을 때만) — 시험 이름 · 끝나면 지움
def test_real_local_redis_short_run(tmp_path, capsys):
    redis = pytest.importorskip("redis")
    r = redis.Redis.from_url("redis://127.0.0.1:6380/0", socket_connect_timeout=1, socket_timeout=2)
    try:
        r.ping()
    except Exception:  # noqa: BLE001
        pytest.skip("로컬 Redis 6380 없음")
    name = "replay-test-pi12"
    r.delete("camera_stream:" + name, "state_stream:" + name)
    make_run(tmp_path, n=5)
    seen = {}

    def watch():                                                        # 도는 동안 스트림이 자라는지 본다
        t0 = time.time()
        while time.time() - t0 < 3 and not seen:
            if r.xlen("camera_stream:" + name) >= 3:
                seen["entry"] = r.xrevrange("camera_stream:" + name, count=1)[0]
            time.sleep(0.05)
    th = threading.Thread(target=watch)
    th.start()
    assert rs.main(["--source", "pi12", "--as", name, "--data", str(tmp_path), "--fps", "20", "--loop",
                    "--seconds", "1", "--trim", "--json"]) == 0
    th.join()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    row = out["sources"][0]
    assert row["name"] == name and row["sent"] >= 10 and row["xlen_after"] >= 10 and row["errors"] == 0
    assert "camera_stream:" + name in out["trimmed"] and r.exists("camera_stream:" + name) == 0
    eid, f = seen["entry"]
    assert f[b"image"].startswith(b"\xff\xd8") and json.loads(f[b"header"])["stream"] == name
    assert time.time() - int(eid.decode().split("-")[0]) / 1000.0 < 10                 # 지금 시각으로 들어갔다
