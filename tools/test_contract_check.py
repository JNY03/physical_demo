# -*- coding: utf-8 -*-
"""tools/contract_check.py 시험 — 모양 뽑기 · 「칸 추가만」 비교 규칙 · 걸러 내기.

가짜 HTTP 서버(소스 포트 흉내)와 가짜 Redis 로 돈다. 실제 data_stream · Redis 는 필요 없다.
    venv/bin/python -m pytest tools/test_contract_check.py -q
"""
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_check as cc  # noqa: E402


# ---------------------------------------------------------------- 모양
def test_schema_of_records_keys_and_types_not_values():
    s = cc.schema_of({"n": 3, "ok": True, "name": "secret-value", "meta": None, "xs": [1, 2.5]}, "/doc")
    assert s["types"] == ["object"]
    assert s["keys"]["n"] == {"types": ["num"]}
    assert s["keys"]["ok"] == {"types": ["bool"]}
    assert s["keys"]["meta"] == {"types": ["null"]}
    assert s["keys"]["xs"] == {"types": ["list"], "items": {"types": ["num"]}}
    assert "secret-value" not in json.dumps(s)


def test_merge_marks_keys_missing_from_some_samples_as_optional():
    a = cc.schema_of({"x": 1, "y": "a"})
    b = cc.schema_of({"x": 2, "z": [1]})
    m = cc.merge(a, b)
    assert "optional" not in m["keys"]["x"]
    assert m["keys"]["y"]["optional"] and m["keys"]["z"]["optional"]


def test_merge_unions_types_and_list_items():
    m = cc.merge(cc.schema_of([{"a": 1}]), cc.schema_of([{"a": None, "b": 2}]))
    assert m["items"]["keys"]["a"]["types"] == ["null", "num"]
    assert m["items"]["keys"]["b"]["optional"]


def test_dynamic_maps_by_path_and_by_key_shape():
    # by_topic: 토픽 이름이 키 → 키 집합이 아니라 값 모양만
    s = cc.schema_of({"by_topic": {"zoneA/robot/x/state": {"n": 1}}}, "/state")
    assert s["keys"]["by_topic"]["types"] == ["map"]
    assert s["keys"]["by_topic"]["values"]["keys"]["n"] == {"types": ["num"]}
    # 경로 목록에 없어도 / · 빈칸 · IP 가 든 키면 사전으로 본다
    s = cc.schema_of({"m": {"computer chair": 1, "night sky": 2}}, "/x")
    assert s["keys"]["m"]["types"] == ["map"]
    s = cc.schema_of({"m": {"10.0.0.1": 1}}, "/x")
    assert s["keys"]["m"]["types"] == ["map"]
    assert "10.0.0.1" not in json.dumps(s)


def test_text_schema_for_redis_fields():
    assert cc.text_schema(b"\xff\xd8\xff\xe0binary", "f") == {"types": ["bytes"]}
    assert cc.text_schema(b"123", "f") == {"types": ["num"]}
    assert cc.text_schema(b"", "f") == {"types": ["null"]}
    assert cc.text_schema(b"hello", "f") == {"types": ["str"]}
    assert cc.text_schema(b'{"a": [1]}', "f")["keys"]["a"]["items"] == {"types": ["num"]}
    assert cc.text_schema("null", "f") == {"types": ["null"]}


def test_key_pattern_hides_source_but_keeps_model():
    assert cc.key_pattern("camera_stream:robot1") == "camera_stream:*"
    assert cc.key_pattern("vision_stream:ep-001:yoloe") == "vision_stream:*:yoloe"
    assert cc.key_pattern("orchestration_stream") == "orchestration_stream"


# ---------------------------------------------------------------- 비교 규칙 (칸 추가만)
def _cmp(base, new):
    rep = cc.Report()
    cc.compare_schema(cc.schema_of(base, "/d"), cc.schema_of(new, "/d"), "d", rep)
    return rep


def test_added_key_is_warning_only():
    rep = _cmp({"a": 1}, {"a": 1, "b": 2})
    assert not rep.fails and any("새 키" in w for w in rep.warns)


def test_removed_key_fails():
    rep = _cmp({"a": 1, "b": 2}, {"a": 1})
    assert any("d.b" in f and "없어짐" in f for f in rep.fails)


def test_renamed_key_fails_and_new_name_warns():
    rep = _cmp({"frame_file": "x"}, {"file": "x"})
    assert rep.fails and rep.warns


def test_type_change_fails():
    rep = _cmp({"n": 1}, {"n": "1"})
    assert any("타입 바뀜" in f for f in rep.fails)
    rep = _cmp({"o": {"x": 1}}, {"o": [1]})
    assert any("타입 바뀜" in f for f in rep.fails)


def test_null_is_wildcard():
    assert not _cmp({"state": None}, {"state": {"gps": 1}}).fails
    rep = _cmp({"state": {"gps": 1}}, {"state": None})
    assert not rep.fails and rep.warns


def test_nested_removal_fails():
    rep = _cmp({"meta": {"sender": {"camera_id": "a"}}}, {"meta": {"sender": {}}})
    assert any("camera_id" in f for f in rep.fails)


def test_optional_key_missing_is_warning():
    base = cc.merge(cc.schema_of({"a": 1, "scan": {}}), cc.schema_of({"a": 1}))
    rep = cc.Report()
    cc.compare_schema(base, cc.schema_of({"a": 1}), "d", rep)
    assert not rep.fails and any("늘 오지는" in w for w in rep.warns)


# ---------------------------------------------------------------- 가짜 소스 포트
class FakeSource(BaseHTTPRequestHandler):
    """data_stream 한 포트 흉내. 클래스 속성으로 「바뀐 판」을 만든다."""
    source = "robot1"
    drop_x_state = False
    extra_header = False
    health_extra = {}
    state_payload = {"battery": {"level_pct": 70}}
    frame_status = 200

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra=()):
        self.send_response(code)
        if ctype:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path, _, q = self.path.partition("?")
        if path == "/health":
            return self._json(200, dict({"source": self.source, "frames": 2, "host": "10.1.2.3",
                                         "status": {"video": "x"}}, **self.health_extra))
        if path == "/state":
            return self._json(200, {"source": self.source, "latest": {"payload": self.state_payload},
                                    "by_topic": {"zoneA/robot/x/state": {"n": 1}}})
        if path == "/frame":
            if "since=" in q or self.frame_status == 204:
                return self._send(204, b"", None)
            if "format=json" in q:
                return self._json(200, {"n": 2, "meta": {"camera_id": "c"}, "state": self.state_payload,
                                        "image_b64": "AAAA", "source": self.source})
            extra = [("X-Frame-N", "2"), ("X-Frame-Meta", json.dumps({"camera_id": "c"}))]
            if not self.drop_x_state:
                extra.append(("X-State", json.dumps(self.state_payload)))
            if self.extra_header:
                extra.append(("X-New-Thing", "1"))
            return self._send(200, b"\xff\xd8jpeg", "image/jpeg", extra)
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\n\xff\xd8ab\r\n")
            self.wfile.flush()
            return
        return self._json(404, {"status": "error", "detail": "none"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        return self._json(400, {"error": "bad_request", "detail": "x"})

    def do_OPTIONS(self):
        self._send(204, b"", None, [("Access-Control-Allow-Methods", "GET")])


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def port_factory():
    servers = []

    def make(**attrs):
        h = type("H", (FakeSource,), attrs)
        srv = _serve(h)
        servers.append(srv)
        return srv.server_address[1]
    yield make
    for s in servers:
        s.shutdown()
        s.server_close()


def _snap(port, as_port="10003"):
    """다른 포트에서 찍어도 같은 포트 이름으로 비교되게 바꾼다."""
    snap = cc.capture("127.0.0.1", [port], no_redis=True)
    snap["http"] = {as_port: snap["http"][str(port)]}
    snap["ports"] = [int(as_port)]
    return snap


def test_capture_http_records_contract_shape(port_factory):
    snap = _snap(port_factory())
    p = snap["http"]["10003"]
    assert p["answers"] and p["source"] == "robot1"
    fr = p["endpoints"]["GET /frame"]
    assert fr["status"] == 200 and fr["content_type"] == "image/jpeg"
    assert {"x-state", "x-frame-n", "x-frame-meta"} <= set(fr["headers"])
    assert fr["header_json"]["x-state"]["keys"]["battery"]["keys"]["level_pct"] == {"types": ["num"]}
    assert p["endpoints"]["GET /frame?since=<n>"]["status"] == 204
    assert p["endpoints"]["GET /stream"]["first_part_headers"] == ["content-length", "content-type"]
    assert p["endpoints"]["GET /state"]["json"]["keys"]["by_topic"]["types"] == ["map"]
    # 값(IP · 그림)은 남지 않는다
    text = json.dumps(snap)
    assert "10.1.2.3" not in text and "AAAA" not in text


def test_same_server_passes(port_factory):
    base = _snap(port_factory())
    rep = cc.compare(base, _snap(port_factory()))
    assert not rep.fails and not rep.warns


def test_missing_x_state_header_fails(port_factory):
    base = _snap(port_factory())
    rep = cc.compare(base, _snap(port_factory(drop_x_state=True)))
    assert any("x-state" in f for f in rep.fails)


def test_new_header_and_new_health_key_only_warn(port_factory):
    base = _snap(port_factory())
    rep = cc.compare(base, _snap(port_factory(extra_header=True, health_extra={"desired_version": 3})))
    assert not rep.fails
    assert any("x-new-thing" in w for w in rep.warns) and any("desired_version" in w for w in rep.warns)


def test_state_payload_type_change_fails(port_factory):
    base = _snap(port_factory())
    rep = cc.compare(base, _snap(port_factory(state_payload={"battery": "70%"})))
    assert any("battery" in f and "타입" in f for f in rep.fails)


def test_source_change_and_dead_port_fail(port_factory):
    base = _snap(port_factory())
    assert any("소스가 바뀜" in f for f in cc.compare(base, _snap(port_factory(source="drone"))).fails)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead = s.getsockname()[1]
    s.close()
    dead_snap = _snap(dead)
    assert any("답하지 않음" in f for f in cc.compare(base, dead_snap).fails)


# ---------------------------------------------------------------- 가짜 Redis
class FakeRedis:
    def __init__(self, data):
        self.data = data            # 키 → ("stream", [필드 사전, ...]) | ("string", 값)

    def scan_iter(self, count=None):
        return iter([k.encode() for k in self.data])

    def type(self, k):
        return self.data[k.decode()][0].encode()

    def xrevrange(self, k, mx="+", mn="-", count=None):
        entries = self.data[k.decode()][1]
        return [(b"%d-0" % i, {a.encode(): (b if isinstance(b, bytes) else b.encode()) for a, b in e.items()})
                for i, e in enumerate(reversed(entries))][:count]

    def get(self, k):
        v = self.data[k.decode()][1]
        return v if isinstance(v, bytes) else v.encode()


def _cam(**extra):
    e = {"image": b"\xff\xd8\xff\x00", "header": json.dumps({"stream": "robot1", "n": 1, "frame_file": "f",
                                                            "received_by": "host-a", "sender": {"camera_id": "c"}})}
    e.update(extra)
    return e


def _redis_snap(data):
    return {"format": cc.FORMAT, "http": {}, "redis": cc.capture_redis(FakeRedis(data)), "redis_error": None}


def test_capture_redis_patterns_fields_and_optional():
    snap = cc.capture_redis(FakeRedis({
        "camera_stream:robot1": ("stream", [_cam(), _cam(timestamp="2026-01-01T00:00:00+09:00")]),
        "camera_stream:drone": ("stream", [_cam()]),
        "vision_reid": ("string", '{"version": 1, "items": []}'),
    }))
    cam = snap["camera_stream:*"]
    assert cam["type"] == "stream" and cam["instances"] == ["drone", "robot1"]
    assert cam["fields"]["image"] == {"types": ["bytes"]}
    assert cam["fields"]["header"]["keys"]["sender"]["keys"]["camera_id"] == {"types": ["str"]}
    assert cam["fields"]["timestamp"]["optional"]
    assert "optional" not in cam["fields"]["header"]
    assert snap["vision_reid"]["value"]["keys"]["version"] == {"types": ["num"]}


def test_redis_field_removed_fails_added_warns():
    base = _redis_snap({"state_stream:robot1": ("stream", [{"topic": "t", "payload": "{}", "header": "{}"}])})
    new = _redis_snap({"state_stream:robot1": ("stream", [{"topic": "t", "header": "{}", "desired_version": "1"}])})
    rep = cc.compare(base, new)
    assert any("#payload" in f for f in rep.fails)
    assert any("desired_version" in w for w in rep.warns)


def test_redis_stream_pattern_missing_fails_but_string_key_only_warns():
    base = _redis_snap({"orchestration_stream": ("stream", [{"event": "planned"}]),
                        "vision_scan_mark:robot1": ("string", "3")})
    rep = cc.compare(base, _redis_snap({}))
    assert any("orchestration_stream" in f for f in rep.fails)
    assert not any("vision_scan_mark" in f for f in rep.fails)
    assert any("vision_scan_mark" in w for w in rep.warns)


def test_redis_missing_instance_only_warns():
    base = _redis_snap({"camera_stream:robot1": ("stream", [_cam()]), "camera_stream:drone": ("stream", [_cam()])})
    rep = cc.compare(base, _redis_snap({"camera_stream:robot1": ("stream", [_cam()])}))
    assert not rep.fails and any("drone" in w for w in rep.warns)


def test_redis_header_key_rename_fails():
    base = _redis_snap({"camera_stream:robot1": ("stream", [_cam()])})
    renamed = _cam(header=json.dumps({"stream": "robot1", "n": 1, "file": "f", "received_by": "x",
                                      "sender": {"camera_id": "c"}}))
    rep = cc.compare(base, _redis_snap({"camera_stream:robot1": ("stream", [renamed])}))
    assert any("frame_file" in f for f in rep.fails)


# ---------------------------------------------------------------- 걸러 내기
def test_scrub_replaces_ip_and_hostname():
    host = socket.gethostname()
    out = cc.scrub({"k": ["100.64.1.2", "x-%s-y" % host], "map": {"10.0.0.5": 1}})
    text = json.dumps(out)
    assert "100.64.1.2" not in text and "10.0.0.5" not in text and host.lower() not in text.lower()
    assert "<ip>" in text and "<host>" in text
    assert cc.leaks(out) == []


def test_capture_never_stores_values(port_factory):
    snap = cc.capture("127.0.0.1", [port_factory()], no_redis=True)
    assert cc.leaks(snap) == []
    assert "10.1.2.3" not in json.dumps(snap)


# ---------------------------------------------------------------- 명령행
def test_main_compare_exit_codes(tmp_path, port_factory, capsys):
    good = port_factory()
    base = tmp_path / "base.json"
    assert cc.main(["capture", "--ports", str(good), "--no-redis", "--out", str(base)]) == 0
    assert cc.main(["compare", str(base), "--ports", str(good), "--no-redis"]) == 0
    # 같은 포트 번호에서 X-State 를 뺀 판 — 기준 파일의 포트 이름을 바꿔 비교한다
    bad = port_factory(drop_x_state=True)
    b = json.loads(base.read_text())
    b["http"] = {str(bad): b["http"][str(good)]}
    base.write_text(json.dumps(b))
    assert cc.main(["compare", str(base), "--ports", str(bad), "--no-redis"]) == 1
    assert "실패" in capsys.readouterr().out


def test_main_feed_requires_yes(capsys):
    assert cc.main(["feed"]) == 2


def test_seed_copies_only_vision_and_orchestration_streams():
    class Dst:
        def __init__(self):
            self.added = {}

        def xadd(self, k, f):
            self.added.setdefault(k, []).append(f)
    src = FakeRedis({"vision_stream:robot1:yoloe": ("stream", [{"result": "{}"}] * 5),
                     "orchestration_stream": ("stream", [{"event": "planned"}] * 2),
                     "camera_stream:robot1": ("stream", [_cam()]),
                     "vision_reid": ("string", "{}")})
    dst = Dst()
    out = cc.seed(src, dst, per_key=3)
    assert out == {"vision_stream:robot1:yoloe": 3, "orchestration_stream": 2}
    assert set(dst.added) == {"vision_stream:robot1:yoloe", "orchestration_stream"}


def test_main_seed_refuses_same_redis():
    assert cc.main(["seed", "--to", cc.DEFAULT_REDIS, "--yes"]) == 2


def test_ports_parser_skips_10010():
    assert cc._ports("10008-10012") == [10008, 10009, 10011, 10012]
    assert cc._ports("10002,10003") == [10002, 10003]
