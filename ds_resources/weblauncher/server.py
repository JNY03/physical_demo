"""웹 런처 HTTP 서버 — 표준 라이브러리 http.server 만 쓴다 (core, 2026-11-25, Docker 계획 D7).

    python -m ds_resources.weblauncher [--host 127.0.0.1] [--port 10080] [--redis redis://127.0.0.1:6380/0]
                                       [--state-dir state]          (환경변수 DS_STATE_DIR 도 된다)

웹 런처는 **원하는 상태(ds_desired/1)만 쓴다.** 프로세스를 띄우거나 끄지 않고 docker 소켓도 받지 않는다.
감독자(ds_resources.supervisor)가 그것을 읽어 실제를 맞추고 `ds_reported:<역할>` · orchestration_stream 에 알린다.

**인증이 없다.** 기본은 127.0.0.1 에만 연다. `--host 0.0.0.0` 은 그 망의 누구나 실행 · 끄기를 할 수 있다는 뜻이다
(테일넷처럼 믿는 망에서만).

Host 검사: 모든 요청의 `Host` 헤더가 localhost · 127.0.0.1 · ::1 · 이 기기 이름(settings.node_name()) ·
환경변수 `DS_WEB_ALLOWED_HOSTS`(쉼표 구분) 중 하나여야 한다(포트는 무시). 아니면 403.
다른 이름(테일넷 이름 등)으로 열려면 그 환경변수에 넣는다.

API
  GET  /health            {"ok": true}
  GET  /api/catalog       cue_catalog.catalog() + sources · robot_cams · slice_v · ports
  GET  /api/desired       {"desired", "source", "ui"}
  PUT  /api/desired       {"ui": 화면 상태, "running": 참/거짓} → 검사 → desired.write. 틀리면 400 {"errors": [...]}
                          (ui 없이 원하는 상태 문서 그대로 보내도 된다)
  POST /api/plan          {"ui"} → {"text", "json"} — 배치 보기 (ds_resources.cli plan 과 같은 계산, 띄우지 않음)
  GET  /api/status        원하는 상태 · ds_reported:* · orchestration_stream(execution_id 별 마지막) · 소스 /health · 상태 줄
  POST /api/register_target {"name", "size", "colour"} → 직접 입력 대상을 등록 대상으로 저장 (런처 「등록으로 저장」)
"""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ds_resources import desired as ds
from ds_resources.streams import ORCH_STREAM, latest_by_execution
from ds_resources.weblauncher import profile as pf

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
REPORTED_PREFIX = "ds_reported:"
STALE_S = 30.0                    # 감독자 보고가 이보다 오래되면 「응답 없음」
CATALOG_TTL_S = 5.0
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".ico": "image/x-icon"}
#: 감독자 구성 요소 상태 → 화면 글 (모르는 값은 그대로)
STATUS_KO = {"running": "실행 중", "starting": "띄우는 중", "stopping": "끄는 중", "stopped": "꺼짐", "off": "꺼짐",
             "failed": "실패", "error": "오류", "backoff": "다시 띄우기 대기", "waiting": "대기", "pending": "대기",
             "restarting": "다시 띄우는 중", "refused": "못 함", "degraded": "낮춤"}
BAD = {"failed", "error", "backoff", "refused", "crashloop"}
#: 받아 주는 Host 헤더 (포트는 떼고 비교). 그 밖의 이름으로 오면 403
#: — 인증이 없으니 아무 웹페이지가 POST /api/plan · /api/register_target 을 부르는 것(과 DNS rebinding)을 여기서 막는다
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def host_of(header):
    """Host 헤더의 호스트 부분(소문자, 포트 없음). IPv6 은 대괄호를 뗀다. 없으면 빈 글자."""
    h = (header or "").strip().lower()
    if h.startswith("["):                                  # [::1]:10080
        return h[1:].split("]", 1)[0]
    return h.split(":", 1)[0] if h.count(":") == 1 else h   # 콜론이 여럿이면 대괄호 없는 IPv6 — 그대로


def allowed_hosts():
    """받아 주는 호스트 이름들 — localhost · 127.0.0.1 · ::1 · 이 기기 이름 · DS_WEB_ALLOWED_HOSTS(쉼표)."""
    from ds_resources import settings
    out = set(LOCAL_HOSTS)
    out.add(str(settings.node_name()).lower())
    for x in (os.environ.get("DS_WEB_ALLOWED_HOSTS") or "").split(","):
        x = host_of(x)
        if x:
            out.add(x)
    return out


# ------------------------------------------------------------ 카탈로그 · 배치 (기본 구현)
def _import_cue_catalog():
    spec = importlib.util.spec_from_file_location("cue_catalog", REPO / "vision" / "cue_catalog.py")
    mod = sys.modules.get("cue_catalog")
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cue_catalog"] = mod
        spec.loader.exec_module(mod)
    return mod


def default_catalog():
    """vision/cue_catalog.py --json 과 같은 것 (torch 를 import 하지 않는다 — ds-core 에서 된다)"""
    cc = _import_cue_catalog()
    cfg = cc._load_yaml(str(REPO / "vision" / "vision_config.yaml"))
    return cc.catalog(cfg)


def default_planner(redis_url):
    """ds_resources.cli plan --json 과 같은 계산을 프로세스 안에서 (읽기만 · 띄우지 않음)"""
    def plan(sources, models, profile):
        from ds_resources import cli
        a = argparse.Namespace(redis=redis_url, redis_public=redis_url,
                               cluster=os.environ.get("DS_CLUSTER", "auto"))
        k = cli.build(a, writer=False)
        p = k.preview(sources, models, profile)
        return cli.brief(p), p["plan"] | {"mode": p["mode"], "basis": p["basis"], "why": p["why"]}
    return plan


def default_health(port, host="127.0.0.1", timeout=0.5):
    try:
        with urllib.request.urlopen("http://%s:%d/health" % (host, port), timeout=timeout) as resp:
            return {"ok": resp.status == 200}
    except Exception as e:  # noqa: BLE001 — 안 떠 있으면 연결 거부
        return {"ok": False, "error": type(e).__name__}


def _ts_age(ts, now):
    if ts in (None, ""):
        return None
    try:
        return now - float(ts)
    except (TypeError, ValueError):
        pass
    try:
        return now - datetime.fromisoformat(str(ts)).timestamp()
    except ValueError:
        return None


# ------------------------------------------------------------ 앱
class App:
    def __init__(self, r, state_dir, catalog_fn=default_catalog, planner=None, health_fn=default_health,
                 clock=time.time, target_dir=None):
        self.r = r
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / "desired.json"
        self.catalog_fn, self.planner, self.health_fn, self.clock = catalog_fn, planner, health_fn, clock
        self.target_dir = target_dir
        self._cat, self._cat_t = None, 0.0
        self.lock = threading.Lock()                 # 쓰기 · 배치 계산을 한 번에 하나씩
        self.last_profile_check = None

    # -- 카탈로그 (실패하면 마지막으로 읽은 것 · state/catalog_cache.json)
    def catalog(self, fresh=False):
        now = self.clock()
        if self._cat is not None and not fresh and now - self._cat_t < CATALOG_TTL_S:
            return self._cat
        cache = self.state_dir / "catalog_cache.json"
        try:
            cat = self.catalog_fn()
            try:
                ds._atomic_write(cache, cat)
            except OSError:
                pass
        except Exception:  # noqa: BLE001
            cat = self._cat or ds._load_file(cache)
            if cat is None:
                raise
        self._cat, self._cat_t = cat, now
        return cat

    def catalog_payload(self):
        cat = dict(self.catalog(fresh=True))
        cat.update(sources=list(pf.SOURCES), robot_cams=list(pf.ROBOT_CAMS), slice_v=pf.SLICE_V, ports=dict(pf.PORTS),
                   label_items=[list(x) for x in pf.LABEL_ITEMS], flag_items=[list(x) for x in pf.FLAG_ITEMS])
        return cat

    def get_desired(self):
        d, src = ds.read(self.r, self.path)
        return {"desired": d, "source": src, "ui": pf.ui_from_desired(d, self.catalog())}

    def put_desired(self, body):
        """(상태 코드, 응답)"""
        if not isinstance(body, dict):
            return 400, {"errors": ["JSON 사전을 보내세요."]}
        with self.lock:
            prev, _ = ds.read(self.r, self.path)
            if "ui" in body:
                cat = self.catalog()
                ui = pf.normalize_ui(body.get("ui"), cat)
                errs = pf.validate_ui(ui, cat)
                doc = pf.to_desired(ui, body.get("running", prev.get("running")), cat, prev)
            else:
                errs, doc = [], dict(body)
            out, errs2 = (None, errs) if errs else ds.write(self.r, self.path, doc, by="web", clock=self.clock)
            if errs or errs2:
                return 400, {"errors": errs or errs2}
        return 200, {"desired": out, "profile_check": self.check_profile(out)}

    def check_profile(self, d):
        """비전이 받아들일 프로필인지 미리 본다(막지는 않는다 — 런처처럼 비전이 거절하면 「반영 안 됨」)"""
        v = d.get("vision") or {}
        if not v.get("enabled") or not v.get("models"):
            return None
        try:
            cc = _import_cue_catalog()
            cfg = cc._load_yaml(str(REPO / "vision" / "vision_config.yaml"))
            cc.resolve_profile(v.get("profile") or {}, cfg, v["models"])
        except Exception as e:  # noqa: BLE001 — ProfileError 와 그 밖(파일 없음 등)
            return str(e)
        return None

    def plan(self, body):
        cat = self.catalog()
        ui = pf.normalize_ui((body or {}).get("ui"), cat)
        doc = pf.to_desired(ui, True, cat)
        srcs, models = ds.vision_sources(doc), doc["vision"]["models"]
        if not srcs or not models:
            return 400, {"errors": ["소스와 비전 모델을 고르세요."]}
        if self.planner is None:
            return 503, {"errors": ["배치 계산기가 없습니다."]}
        with self.lock:
            try:
                text, js = self.planner(srcs, models, doc["vision"]["profile"])
            except Exception as e:  # noqa: BLE001
                return 500, {"errors": ["배치를 계산하지 못했습니다: %s" % e]}
        return 200, {"text": text, "json": js, "sources": srcs, "models": models}

    def register_target(self, body):
        body = body or {}
        name = str(body.get("name") or "").strip().lower()
        size = str(body.get("size") or "").strip().lower()
        colour = str(body.get("colour") or "").strip()
        cc = _import_cue_catalog()
        try:
            spec = cc.parse_target_arg(name + (":size=" + size if size else "") + (":colour=" + colour if colour else ""))
            kw = {} if self.target_dir is None else {"target_dir": self.target_dir}
            f = cc.save_direct_target(spec["name"], spec.get("size_mm"), spec.get("colours"), **kw)
        except cc.ProfileError as e:
            return 400, {"ok": False, "errors": [str(e)]}
        except OSError as e:
            return 500, {"ok": False, "errors": ["저장 실패: %s" % e]}
        self._cat = None
        return 200, {"ok": True, "file": f}

    # -- 상태
    def reported(self):
        out = {}
        if self.r is None:
            return out
        try:
            for k in self.r.scan_iter(match=REPORTED_PREFIX + "*"):
                k = k.decode() if isinstance(k, bytes) else k
                raw = self.r.get(k)
                try:
                    out[k[len(REPORTED_PREFIX):]] = json.loads(raw) if raw else None
                except ValueError:
                    out[k[len(REPORTED_PREFIX):]] = {"invalid": True}
        except Exception:  # noqa: BLE001 — Redis 가 없으면 빈 보고
            pass
        return out

    def orchestration(self, count=200, keep=12):
        if self.r is None:
            return []
        try:
            rows = self.r.xrevrange(ORCH_STREAM, count=count)
        except Exception:  # noqa: BLE001
            return []
        ents = []
        for _id, f in reversed(rows):
            f = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in f.items()}
            f["id"] = _id.decode() if isinstance(_id, bytes) else _id
            ents.append(f)
        last = sorted(latest_by_execution(ents).values(), key=lambda e: e["id"], reverse=True)
        return [{k: e.get(k, "") for k in ("id", "event", "request_id", "execution_id", "mode", "reason", "ts",
                                            "desired_version", "converged")} for e in last[:keep]]

    def health(self, d):
        if not d.get("running") or self.health_fn is None:
            return {}
        srcs = [s for s in ds.vision_sources(d) if s in pf.PORTS]
        out, ths = {}, []

        def one(s):
            out[s] = dict(self.health_fn(pf.PORTS[s]), port=pf.PORTS[s])
        for s in srcs:
            t = threading.Thread(target=one, args=(s,), daemon=True)
            t.start()
            ths.append(t)
        for t in ths:
            t.join(1.5)
        return out

    def vision_files(self):
        """비전이 프로필 옆에 남기는 반영 결과(status_*.json) 중 가장 새 것 — 런처의 상태 줄 폴링과 같다"""
        try:
            fs = sorted(self.state_dir.glob("status_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return None
        return ds._load_file(fs[0]) if fs else None

    def status(self):
        d, src = ds.read(self.r, self.path)
        rep = self.reported()
        now = self.clock()
        st = {"desired_version": d["version"], "running": d["running"], "source": src, "reported": rep,
              "orchestration": self.orchestration(), "health": self.health(d), "vision_file": self.vision_files()}
        st.update(summarize(d, rep, now, st["vision_file"]))
        return st


def summarize(d, rep, now, vision_file=None):
    """상태 줄 — converged · 구성 요소 · 반영 안 됨 이유"""
    ver = d["version"]
    roles = {k: v for k, v in rep.items() if isinstance(v, dict)}
    stale = sorted(k for k, v in roles.items() if (_ts_age(v.get("ts"), now) or 0) > STALE_S)
    comps, problems = [], []
    for role in sorted(roles):
        for name, c in sorted((roles[role].get("components") or {}).items()):
            c = c or {}
            s = str(c.get("status") or "")
            comps.append({"role": role, "name": name, "status": s, "shown": STATUS_KO.get(s, s),
                          "reason": c.get("reason") or "", "pid": c.get("pid"), "restarts": c.get("restarts")})
            if c.get("reason") and (s in BAD or s not in ("running", "stopped", "off")):
                problems.append("%s — %s" % (name, c["reason"]))
    if vision_file and vision_file.get("ok") is False and d["running"]:
        problems.append("비전 — %s" % vision_file.get("message"))
    behind = [k for k, v in roles.items() if int(v.get("desired_version") or 0) != ver]
    converged = bool(roles) and not stale and not behind and all(v.get("converged") for v in roles.values())
    if not roles:
        line = ("● 실행 요청됨 (v%d) · 감독자 보고 없음" % ver) if d["running"] else "꺼짐"
    elif d["running"]:
        line = ("● 실행 중 · 반영됨 (v%d)" if converged else "● 반영 중 (v%d)") % ver
    else:
        line = ("꺼짐 (v%d)" if converged else "끄는 중 (v%d)") % ver
    if roles and comps:
        line += " · " + " · ".join("%s %s" % (c["name"], c["shown"]) for c in comps)
    if stale:
        line += " · 응답 없음: " + ", ".join(stale)
    if problems:
        line += " · 반영 안 됨: " + " / ".join(problems)
    return {"converged": converged, "components": comps, "problems": problems, "stale": stale, "line": line}


# ------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    app = None                       # make_server 가 단다
    server_version = "ds-weblauncher/1"

    def log_message(self, fmt, *args):  # noqa: D401 — 조용히 (오류만)
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _route(self, method):
        path = self.path.split("?", 1)[0]
        a = self.app
        try:
            host = host_of(self.headers.get("Host"))
            if host not in allowed_hosts():
                return self._send(403, {"errors": [
                    "이 이름(%s)으로는 받지 않습니다 — localhost · 127.0.0.1 · 이 기기 이름으로 열거나 "
                    "DS_WEB_ALLOWED_HOSTS 에 넣으세요." % (host or "Host 없음")]})
            if method == "GET" and path == "/health":
                return self._send(200, {"ok": True})
            if method == "GET" and path == "/api/catalog":
                return self._send(200, a.catalog_payload())
            if method == "GET" and path == "/api/desired":
                return self._send(200, a.get_desired())
            if method == "GET" and path == "/api/status":
                return self._send(200, a.status())
            if method in ("PUT", "POST") and path in ("/api/desired", "/api/plan", "/api/register_target"):
                try:
                    body = self._body()
                except ValueError:
                    return self._send(400, {"errors": ["JSON 을 읽지 못했습니다."]})
                if path == "/api/desired" and method == "PUT":
                    return self._send(*a.put_desired(body))
                if path == "/api/plan" and method == "POST":
                    return self._send(*a.plan(body))
                if path == "/api/register_target" and method == "POST":
                    return self._send(*a.register_target(body))
                return self._send(405, {"errors": ["허용하지 않는 방식"]})
            if method == "GET":
                return self._static(path)
            return self._send(404, {"errors": ["없는 경로"]})
        except Exception as e:  # noqa: BLE001 — 서버는 살아 있게
            return self._send(500, {"errors": ["서버 오류: %s" % e]})

    def _static(self, path):
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (STATIC / name).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self._send(404, {"errors": ["없는 경로"]})
        return self._send(200, f.read_bytes(), CTYPES.get(f.suffix, "application/octet-stream"))

    def do_GET(self):
        self._route("GET")

    def do_PUT(self):
        self._route("PUT")

    def do_POST(self):
        self._route("POST")


def make_server(app, host="127.0.0.1", port=10080):
    h = type("BoundHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), h)
    srv.daemon_threads = True
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description="데이터 스트림 웹 런처 (원하는 상태만 쓴다)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 은 인증 없이 망 전체에 연다 — 믿는 망에서만")
    ap.add_argument("--port", type=int, default=10080)
    ap.add_argument("--redis", default=os.environ.get("DS_REDIS", "redis://127.0.0.1:6380/0"))
    ap.add_argument("--state-dir", default=os.environ.get("DS_STATE_DIR", str(REPO / "state")))
    a = ap.parse_args(argv)
    import redis
    r = redis.Redis.from_url(a.redis, socket_connect_timeout=2, socket_timeout=5, decode_responses=True)
    app = App(r, a.state_dir, planner=default_planner(a.redis))
    srv = make_server(app, a.host, a.port)
    print("웹 런처: http://%s:%d/  (Redis %s · 상태 %s)%s" % (
        "127.0.0.1" if a.host == "0.0.0.0" else a.host, a.port, a.redis, app.path,
        "  ※ 인증 없음 — 0.0.0.0 으로 열었다" if a.host not in ("127.0.0.1", "localhost") else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
