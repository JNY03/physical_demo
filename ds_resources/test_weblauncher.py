"""웹 런처 (D7) 시험 — 임시 포트에 서버를 띄우고 가짜 Redis · 임시 state 로 API 를 부른다.

프로필 모양 시험의 기대값은 tools/launcher/launcher.ps1 Build-Profile 을 읽고 손으로 옮긴 것이다
(Self-Test 시나리오: yoloe+targeted · person/door/chair + 직접 입력 traffic cone 300x300x700 orange ·
 door 근거 직접 고르기에서 종횡비 끔 · 오버레이 targeted 개수/카드 항상/탐지 영역 끔 · yoloe 개수 · 거리 끔 · reid).
"""
import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ds_resources import desired as ds
from ds_resources.streams import ORCH_STREAM
from ds_resources.weblauncher import profile as pf
from ds_resources.weblauncher import server as sv

HERE = Path(__file__).resolve().parent
CAT = json.loads((HERE / "weblauncher" / "test_catalog.json").read_text(encoding="utf-8"))
DOOR_DEFAULT = ["class", "free_arm", "semantic", "dict_features", "ovd_pair_iou", "clip_shape", "saturation", "parts",
                "aspect", "size_veto"]


class FakeRedis:
    def __init__(self):
        self.kv, self.streams = {}, {}

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v):
        self.kv[k] = v

    def scan_iter(self, match="*"):
        rx = re.compile("^" + re.escape(match).replace("\\*", ".*") + "$")
        return [k for k in list(self.kv) if rx.match(k)]

    def xadd(self, name, fields):
        s = self.streams.setdefault(name, [])
        i = "%d-0" % (len(s) + 1)
        s.append((i, dict(fields)))
        return i

    def xrevrange(self, name, count=None):
        return list(reversed(self.streams.get(name, [])))[:count]


@pytest.fixture
def srv(tmp_path):
    r = FakeRedis()
    calls = []

    def planner(sources, models, profile):
        calls.append((sources, models, profile))
        return "방식: local (기억한 자원)", {"gpu_units": [], "mode": "local"}

    health = {}
    (tmp_path / "targets").mkdir()
    app = sv.App(r, tmp_path / "state", catalog_fn=lambda: CAT, planner=planner,
                 health_fn=lambda port: health.get(port, {"ok": False}), target_dir=tmp_path / "targets")
    s = sv.make_server(app, "127.0.0.1", 0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    base = "http://127.0.0.1:%d" % s.server_address[1]
    yield {"base": base, "r": r, "app": app, "calls": calls, "health": health, "tmp": tmp_path}
    s.shutdown()
    s.server_close()


def call(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw, code, ctype = resp.read(), resp.status, resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raw, code, ctype = e.read(), e.code, e.headers.get("Content-Type", "")
    return code, (json.loads(raw) if "json" in ctype else raw.decode("utf-8"))


def ui(**kw):
    u = pf.default_ui(CAT)
    u.update(sources=["pi13"], vision=True, models=["yoloe"])
    u.update(kw)
    return u


# ------------------------------------------------------------ 프로필 모양 (Build-Profile 과 같음)
def test_profile_parity_launcher_selftest_scenario():
    u = ui(models=["targeted", "yoloe"],
           targets=["person", "door", "chair", "traffic cone"],
           direct=[{"name": "traffic cone", "size": "300x300x700", "colour": "orange"}],
           distance=False,
           cueMode={"door": "custom"}, cues={"door": [c for c in DOOR_DEFAULT if c != "aspect"]},
           overlay={"targeted": {"count": True, "cards": "always", "roi": False}, "yoloe": {"count": True}},
           features={"reid": True})
    door = ["class", "free_arm", "semantic", "dict_features", "ovd_pair_iou", "clip_shape", "saturation", "parts"]
    assert pf.build_profile(u, CAT) == {
        "models": {"yoloe": {"overlay": {"count": True}},
                   "targeted": {"targets": [{"name": "person"}, {"name": "door", "cues": door}, {"name": "chair"},
                                            {"name": "traffic cone", "size_mm": [300.0, 300.0, 700.0],
                                             "colours": ["orange"]}],
                                "distance": False,
                                "overlay": {"count": True, "cards": "always", "roi": False}}},
        "features": {"reid": True}}
    d = pf.to_desired(u, True, CAT)
    assert d["vision"]["models"] == ["yoloe", "targeted"]                      # 카탈로그 순서 (Chosen-Models)
    assert ds.validate(d) == []


def test_profile_direct_custom_cues_follow_distance():
    base = dict(models=["targeted"], targets=["traffic cone"], cueMode={"traffic cone": "custom"},
                direct=[{"name": "traffic cone", "size": "300x300x700", "colour": "orange"}],
                cues={"traffic cone": ["class", "free_arm", "shape", "size_veto", "colour"]})
    on = pf.build_profile(ui(distance=True, **base), CAT)["models"]["targeted"]
    assert on["targets"][0]["cues"] == ["class", "free_arm", "shape", "size_veto", "colour"] and on["distance"] is True
    off = pf.build_profile(ui(distance=False, **base), CAT)["models"]["targeted"]
    assert off["targets"][0]["cues"] == ["class", "free_arm", "shape", "colour"]          # 크기 거부는 거리 측정이 있어야
    nosize = dict(base, direct=[{"name": "traffic cone", "size": "", "colour": ""}])
    t = pf.build_profile(ui(distance=True, **nosize), CAT)["models"]["targeted"]["targets"][0]
    assert t == {"name": "traffic cone", "cues": ["class", "free_arm"]}                    # 크기 · 색이 없으면 그 근거도 없다


def test_profile_overlay_label_tokens_and_untouched_targets():
    ov = {"river": {"label": ["lead", "dist"]}, "yoloe": {"label": ["name", "dist", "gps"]},
          "targeted": {"label": ["name", "dist", "gps"], "show_rejected": True}}
    p = pf.build_profile(ui(models=["yoloe", "river", "targeted"], overlay=ov, distance=True), CAT)["models"]
    assert p["river"] == {"overlay": {"label": [["lead", "dist"]]}}
    assert p["yoloe"] == {"overlay": {"label": [["name"]]}}                                 # 깊이 모델이 없으면 거리 · GPS 빠짐
    # 대상을 건드리지 않음 → targeted 는 오버레이만 (기본 대상에 demo_box 가 있어 show_rejected 유지)
    assert p["targeted"] == {"overlay": {"label": [["name", "dist", "gps"]], "show_rejected": True}}
    p = pf.build_profile(ui(models=["yoloe", "unidepth", "targeted"], overlay=ov, distance=False,
                            targets=["person"]), CAT)["models"]
    assert p["yoloe"] == {"overlay": {"label": [["name", "dist", "gps"]]}}
    assert p["targeted"] == {"targets": [{"name": "person"}], "distance": False, "overlay": {"label": [["name"]]}}
    assert "unidepth" not in p and "features" not in pf.build_profile(ui(), CAT)
    # targeted 를 안 고르면 대상을 건드렸어도 targeted 칸이 없다
    assert pf.build_profile(ui(models=["yoloe"], targets=["person"]), CAT) == {"models": {}}


def test_feature_unavailable_and_distance_unavailable():
    cat = json.loads(json.dumps(CAT))
    cat["features"][0]["available"] = False
    cat["distance_available"] = False
    p = pf.build_profile(ui(models=["targeted"], targets=["person"], distance=True, features={"reid": True}), cat)
    assert "features" not in p and p["models"]["targeted"]["distance"] is False


def test_ports_match_data_server():
    txt = (HERE.parent / "server_stream_multi_source.py").read_text(encoding="utf-8")
    block = re.search(r"^SOURCES = \{(.*?)\}", txt, re.S | re.M).group(1)
    ports = {k: int(v) for k, v in re.findall(r'"([\w\-]+)":\s*(\d+)', block)}
    assert ports == pf.PORTS
    assert set(pf.SOURCES) <= set(ports)


# ------------------------------------------------------------ API
def test_health_catalog_and_static(srv):
    assert call(srv["base"], "GET", "/health") == (200, {"ok": True})
    code, cat = call(srv["base"], "GET", "/api/catalog")
    assert code == 200 and cat["sources"][0] == "cam360" and cat["robot_cams"] == [1, 2, 3, 4, 5]
    assert cat["targets"][0]["name"] == "person" and cat["slice_v"] == -30 and cat["label_items"][0] == ["name", "클래스명"]
    code, html = call(srv["base"], "GET", "/")
    assert code == 200 and "데이터 스트림 실행" in html and "app.js" in html
    code, js = call(srv["base"], "GET", "/app.js")
    assert code == 200 and "/api/desired" in js and "cdn" not in js.lower()
    assert call(srv["base"], "GET", "/../server.py")[0] == 404
    assert call(srv["base"], "GET", "/nope.css")[0] == 404
    assert (srv["tmp"] / "state" / "catalog_cache.json").exists()          # 카탈로그를 못 읽을 때 쓸 사본


def test_catalog_falls_back_to_cache(srv):
    app = srv["app"]
    app.catalog(fresh=True)

    def boom():
        raise RuntimeError("x")
    app.catalog_fn, app._cat = boom, None
    assert app.catalog(fresh=True)["targets"][0]["name"] == "person"


def test_put_desired_validates_and_bumps_version(srv):
    b, r = srv["base"], srv["r"]
    code, body = call(b, "PUT", "/api/desired", {"ui": ui(sources=[]), "running": True})
    assert code == 400 and "소스를 하나 이상 고르세요." in body["errors"]
    assert ds.REDIS_KEY not in r.kv and not (srv["tmp"] / "state" / "desired.json").exists()
    code, body = call(b, "PUT", "/api/desired", {"ui": ui(vision=True, models=[]), "running": True})
    assert code == 400 and "비전 모델을 하나 이상 고르세요." in body["errors"]
    code, body = call(b, "PUT", "/api/desired", {"ui": ui(models=["targeted"], targets=[]), "running": True})
    assert code == 400 and any("대상" in e for e in body["errors"])
    code, body = call(b, "PUT", "/api/desired", {"ui": ui(slice=True), "running": True})
    assert code == 400 and any("cam360" in e for e in body["errors"])
    bad = ui(models=["targeted"], targets=["cone"], direct=[{"name": "cone", "size": "30x30", "colour": ""}])
    code, body = call(b, "PUT", "/api/desired", {"ui": bad, "running": True})
    assert code == 400 and any("가로x세로x높이" in e for e in body["errors"])

    code, body = call(b, "PUT", "/api/desired", {"ui": ui(sources=["pi13", "cam360"], slice=True, sliceU="12.5"),
                                                 "running": True})
    assert code == 200 and body["desired"]["version"] == 1 and body["desired"]["by"] == "web"
    d = body["desired"]
    assert d["server"] == {"sources": ["cam360", "pi13"], "robot_cam": 1}               # 런처 $Sources 순서
    assert d["slice"] == {"enabled": True, "u": 12.5, "v": -30}
    assert json.loads(r.kv[ds.REDIS_KEY])["version"] == 1
    # 감독자 쪽 max_fps 는 이어받는다
    doc = json.loads(r.kv[ds.REDIS_KEY])
    doc["vision"]["max_fps"] = {"pi13": 5}
    r.kv[ds.REDIS_KEY] = json.dumps(dict(doc, version=5))
    code, body = call(b, "PUT", "/api/desired", {"ui": ui(), "running": False})
    assert code == 200 and body["desired"]["version"] == 6 and body["desired"]["running"] is False
    assert body["desired"]["vision"]["max_fps"] == {"pi13": 5}
    # 끄는 문서는 소스가 없어도 된다 (화면 상태를 남기려고 꺼진 채로도 쓴다)
    assert call(b, "PUT", "/api/desired", {"ui": ui(sources=[]), "running": False})[0] == 200
    code, got = call(b, "GET", "/api/desired")
    assert code == 200 and got["desired"]["version"] == 7 and got["ui"]["sources"] == [] and got["source"] in ("redis", "file")


def test_ui_round_trip_restores_selection(srv):
    b = srv["base"]
    u = ui(models=["targeted"], targets=["door"], cueMode={"door": "custom"}, cues={"door": ["class"]},
           overlay={"targeted": {"count": True}}, robotCam=3, orchestrate=False)
    assert call(b, "PUT", "/api/desired", {"ui": u, "running": False})[0] == 200
    code, got = call(b, "GET", "/api/desired")
    for k in ("models", "targets", "cueMode", "cues", "overlay", "robotCam", "orchestrate"):
        assert got["ui"][k] == u[k]
    assert got["desired"]["vision"]["profile"]["models"]["targeted"]["targets"] == [{"name": "door", "cues": ["class"]}]


def test_desired_without_ui_is_read_back(srv, tmp_path):
    ds.write(srv["r"], srv["tmp"] / "state" / "desired.json",
             {"schema": ds.SCHEMA, "running": True, "server": {"sources": ["a2"], "robot_cam": 2},
              "vision": {"enabled": True, "models": ["river"]}})
    got = call(srv["base"], "GET", "/api/desired")[1]
    assert got["ui"]["sources"] == ["a2"] and got["ui"]["robotCam"] == 2 and got["ui"]["models"] == ["river"]


def test_plan_endpoint(srv):
    b = srv["base"]
    code, body = call(b, "POST", "/api/plan", {"ui": ui(models=[])})
    assert code == 400 and body["errors"] == ["소스와 비전 모델을 고르세요."]
    code, body = call(b, "POST", "/api/plan", {"ui": ui(sources=["cam360"], slice=True,
                                                         models=["targeted"], targets=["person"])})
    assert code == 200 and body["text"].startswith("방식:") and body["json"]["mode"] == "local"
    srcs, models, prof = srv["calls"][-1]
    assert srcs == ["cam360", "cam360_slice"] and models == ["targeted"]
    assert prof["models"]["targeted"]["targets"] == [{"name": "person"}]
    assert ds.REDIS_KEY not in srv["r"].kv                                     # 배치 보기는 쓰지 않는다

    def broken(*a):
        raise RuntimeError("redis 없음")
    srv["app"].planner = broken
    code, body = call(b, "POST", "/api/plan", {"ui": ui()})
    assert code == 500 and "배치를 계산하지 못했습니다" in body["errors"][0]


def test_status_aggregation(srv):
    b, r, app = srv["base"], srv["r"], srv["app"]
    code, st = call(b, "GET", "/api/status")
    assert code == 200 and st["line"] == "꺼짐" and st["converged"] is False
    call(b, "PUT", "/api/desired", {"ui": ui(sources=["pi13", "a2"]), "running": True})
    st = call(b, "GET", "/api/status")[1]
    assert st["line"].startswith("● 실행 요청됨 (v1)") and set(st["health"]) == {"pi13", "a2"}
    srv["health"][10011] = {"ok": True}
    now = app.clock()
    r.set("ds_reported:core", json.dumps({"desired_version": 1, "converged": True, "ts": now,
                                          "components": {"server": {"status": "running", "pid": 11, "restarts": 0}}}))
    r.set("ds_reported:vision", json.dumps({"desired_version": 1, "converged": True, "ts": now,
                                            "components": {"vision": {"status": "running", "pid": 12}}}))
    r.xadd(ORCH_STREAM, {"event_id": "e1", "event": "planned", "request_id": "req-1", "execution_id": "exe-1",
                         "reason": "", "ts": "2026-11-25T10:00:00.000+09:00"})
    r.xadd(ORCH_STREAM, {"event_id": "e2", "event": "running", "request_id": "req-1", "execution_id": "exe-1",
                         "reason": "돎", "ts": "2026-11-25T10:00:05.000+09:00", "desired_version": "1", "converged": "1"})
    r.xadd(ORCH_STREAM, {"event_id": "e2", "event": "running", "request_id": "req-1", "execution_id": "exe-1",
                         "reason": "겹친 기록", "ts": "2026-11-25T10:00:06.000+09:00"})
    st = call(b, "GET", "/api/status")[1]
    assert st["converged"] is True and st["line"].startswith("● 실행 중 · 반영됨 (v1)")
    assert "server 실행 중" in st["line"] and st["health"]["pi13"] == {"ok": True, "port": 10011}
    assert st["health"]["a2"]["ok"] is False
    assert [(e["execution_id"], e["event"], e["reason"]) for e in st["orchestration"]] == [("exe-1", "running", "돎")]
    assert set(st["reported"]) == {"core", "vision"}
    # 새 원하는 상태 → 감독자가 아직 옛 버전 · 비전이 이유를 남김
    call(b, "PUT", "/api/desired", {"ui": ui(sources=["pi13"]), "running": True})
    r.set("ds_reported:vision", json.dumps({"desired_version": 2, "converged": False, "ts": now,
                                            "components": {"vision": {"status": "failed", "reason": "GPU 메모리 부족",
                                                                      "restarts": 3}}}))
    st = call(b, "GET", "/api/status")[1]
    assert st["converged"] is False and st["line"].startswith("● 반영 중 (v2)")
    assert "반영 안 됨: vision — GPU 메모리 부족" in st["line"] and "vision 실패" in st["line"]
    # 감독자 보고가 오래되면 응답 없음
    app.clock = lambda: now + 120
    st = call(b, "GET", "/api/status")[1]
    assert "응답 없음: core, vision" in st["line"]


def test_status_reads_vision_profile_file(srv):
    call(srv["base"], "PUT", "/api/desired", {"ui": ui(), "running": True})
    (srv["tmp"] / "state" / "status_yoloe_pi13.json").write_text(
        json.dumps({"ok": False, "message": "모르는 근거 x"}, ensure_ascii=False), encoding="utf-8")
    st = call(srv["base"], "GET", "/api/status")[1]
    assert "반영 안 됨: 비전 — 모르는 근거 x" in st["line"]


def test_summarize_off_states():
    d = ds.normalize({"version": 4, "running": False})
    rep = {"core": {"desired_version": 4, "converged": True, "components": {"server": {"status": "stopped"}}}}
    assert sv.summarize(d, rep, 0)["line"] == "꺼짐 (v4) · server 꺼짐"
    rep["core"]["converged"] = False
    assert sv.summarize(d, rep, 0)["line"].startswith("끄는 중 (v4)")


def test_bad_json_and_unknown(srv):
    req = urllib.request.Request(srv["base"] + "/api/desired", data=b"{x", method="PUT")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert call(srv["base"], "POST", "/api/desired", {})[0] == 405


def test_register_target(srv):
    code, body = call(srv["base"], "POST", "/api/register_target", {"name": "traffic cone", "size": "300x300x700",
                                                                     "colour": "orange"})
    if code == 500:                                                              # cue_catalog 을 못 불러오는 환경
        pytest.skip(body)
    assert code == 200 and body["ok"] and (srv["tmp"] / "targets" / "traffic cone.json").exists()
    code, body = call(srv["base"], "POST", "/api/register_target", {"name": "traffic cone"})
    assert code == 400 and "이미 등록된 대상" in body["errors"][0]


# ------------------------------------------------------------ Host 검사 (인증이 없으므로)
def test_host_header_parsing():
    assert sv.host_of("127.0.0.1:10080") == "127.0.0.1"
    assert sv.host_of("LocalHost") == "localhost"
    assert sv.host_of("[::1]:10080") == "::1"
    assert sv.host_of(None) == "" and sv.host_of("") == ""


def test_host_check_rejects_other_names_and_allows_env(srv, monkeypatch):
    b, port = srv["base"], srv["base"].rsplit(":", 1)[1]

    def with_host(host, method="GET", path="/api/desired", body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(b + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", "Host": host})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    # 허용 — localhost · 127.0.0.1 · ::1 · 이 기기 이름 (포트는 무시)
    from ds_resources import settings
    for host in ("localhost", "localhost:%s" % port, "127.0.0.1:%s" % port, "[::1]:%s" % port, settings.node_name()):
        code, _ = with_host(host)
        assert code == 200, host

    # 거부 — 아무 웹페이지 · DNS rebinding (파일을 만드는 register_target 도 막힌다)
    for host in ("evil.example", "evil.example:%s" % port, "127.0.0.1.evil.example", ""):
        code, body = with_host(host)
        assert code == 403 and body["errors"]
    code, body = with_host("evil.example", "POST", "/api/register_target", {"name": "x"})
    assert code == 403 and not (srv["tmp"] / "targets" / "x.json").exists()
    code, body = with_host("evil.example", "POST", "/api/plan", {"ui": ui()})
    assert code == 403 and srv["calls"] == []
    code, body = with_host("evil.example", "GET", "/")                       # 정적 파일도 같은 규칙
    assert code == 403

    # 환경변수에 넣으면 그 이름(테일넷 이름 등)으로도 열린다
    monkeypatch.setenv("DS_WEB_ALLOWED_HOSTS", "wsl.tail1234.ts.net, evil.example")
    assert with_host("wsl.tail1234.ts.net:%s" % port)[0] == 200
    assert with_host("evil.example")[0] == 200
    assert with_host("other.example")[0] == 403


# ------------------------------------------------------------ 모델 매칭 · 수동 배치 (2026-11-26)
def test_assign_map_drops_the_default_and_keeps_an_empty_device():
    """기본값(고른 모델 전부)은 적지 않는다 — 적어 두면 소스를 켜고 끌 때마다 비전이 괜히 다시 뜬다"""
    u = ui(models=["yoloe", "river"], sources=["pi13", "a2"],
           assign={"pi13": ["yoloe", "river"], "a2": ["yoloe"], "pi11": ["yoloe"]})
    m = pf.assign_map(u, CAT)
    assert m == {"a2": ["yoloe"]}                                  # pi13 은 전부(기본값) · pi11 은 꺼진 소스
    u2 = ui(models=["yoloe"], sources=["pi13", "a2"], assign={"a2": []})
    assert pf.assign_map(u2, CAT) == {"a2": []}                    # 비워 둔 기기는 남는다 (비전 없음)


def test_to_desired_carries_match_and_manual():
    u = ui(models=["yoloe"], sources=["pi13", "a2"], assign={"a2": []},
           manual=True, manualPlace={"object_detect.device": "pi1", "x": ""})
    d = pf.to_desired(u, True, CAT)
    assert d["vision"]["assign"] == {"a2": []}
    assert d["vision"]["manual"] == {"enabled": True, "place": {"object_detect.device": "pi1"}}
    # 비워 둔 자리("x")는 화면 상태를 다듬을 때 사라진다 — 「자동」과 같은 뜻이라 적을 것이 없다
    assert pf.ui_from_desired(d, CAT)["manualPlace"] == {"object_detect.device": "pi1"}


def test_manual_needs_orchestration_and_all_devices_cannot_be_empty():
    assert any("자원 배분" in e for e in pf.validate_ui(ui(manual=True, orchestrate=False), CAT))
    assert any("모든 기기를 비웠" in e
               for e in pf.validate_ui(ui(sources=["pi13"], models=["yoloe"], assign={"pi13": []}), CAT))


def test_plan_passes_match_and_manual_to_the_planner(srv):
    """배치 보기는 실행과 같은 계산이어야 한다 — 매칭 · 수동 자리를 그대로 넘긴다"""
    seen = []
    srv["app"].planner = lambda s, m, p, a=None, mp=None: (seen.append((s, m, a, mp)) or ("ok", {"mode": "local"}))
    code, body = call(srv["base"], "POST", "/api/plan",
                      {"ui": ui(sources=["pi13", "a2"], models=["yoloe"], assign={"a2": []},
                                manual=True, manualPlace={"object_detect.device": "pi1"})})
    assert code == 200 and body["assign"] == {"a2": []} and body["manual"] == {"object_detect.device": "pi1"}
    assert seen[0][2] == {"a2": []} and seen[0][3] == {"object_detect.device": "pi1"}


def test_places_endpoint_serves_nodes_and_units(srv):
    srv["app"].places_fn = lambda: {"ok": True, "why": "", "mode": "k3s",
                                    "nodes": [{"name": "pi1", "role": "robot", "ready": True, "stale": False,
                                               "limited": True, "caps": {}, "thermal": {}}],
                                    "units": [{"unit": "grasp.device", "pin": "device", "roles": ["robot"],
                                               "pattern": "split", "criticality": "mission", "must_device": True}]}
    code, body = call(srv["base"], "GET", "/api/places")
    assert code == 200 and body["nodes"][0]["name"] == "pi1" and body["units"][0]["unit"] == "grasp.device"
    srv["app"].places_fn = None
    code, body = call(srv["base"], "GET", "/api/places")
    assert code == 503 and body["errors"]
