"""감독자 — 원하는 상태(ds_desired/1)와 실제를 맞춘다 (D6, 2026-11-25).

Docker 계획 §5. 런처(WinForms · 웹)는 원하는 상태만 쓰고, 감독자가 프로세스를 띄우고 끈다.
컨테이너마다 하나(역할별), 지금 기기에서는 WSL venv 에서 런처 창 대신 하나를 띄워도 된다.

  python -m ds_resources.supervisor --role core|vision|all [--redis URL] [--poll 2] [--state-dir 폴더]

  core    데이터 서버 (server_stream_multi_source.py --on <비전 소스> --robot-cam N) · 360 슬라이스 (e2p/slice360_live.py)
  vision  비전. 프로필을 <state>/profile.json 에 쓰고(내용이 바뀔 때만 profile_version +1 — 비전은 파일 mtime 을 보고 반영),
          orchestrate 면 오케스트레이터 컨트롤러를 **이 프로세스 안에서** 돌리고(cli run 과 같은 순서: 정리 → 요청 → 감시 → 멈춤),
          아니면 vision/vision_infer.py 하나를 띄운다(런처의 GPU 여유에 따른 창 나누기는 하지 않는다 — 한 프로세스).
  all     둘 다 (지금 기기에서 런처 창 셋을 대신할 때)

다시 띄우기 규칙 (런처 Apply-Changes 와 같다 — desired.component_keys):
  server 열쇠가 바뀌면 모두 다시 · slice 는 slice 만 · vision 은 켜기 · 모델 · 소스 · 방식이 바뀔 때만 · 프로필만 바뀌면 파일만.
  같은 감독자 안에서는 서버가 뜬 뒤 SLICE_DELAY · VISION_DELAY 초를 기다렸다가 슬라이스 · 비전을 띄운다(런처와 같다).
스스로 끝난 프로세스는 지수 대기(BACKOFF_S … BACKOFF_MAX_S)로 다시 띄운다. 오래(STABLE_S) 돌면 대기를 처음으로.
**감독자는 「돌 것이 없음」으로 끝나지 않는다** (cli run 의 코드 2 와 다르다 — restart: unless-stopped 와 만나 고리가 된다).
SIGTERM · SIGINT 이면 자식을 SIGINT → SIGTERM → SIGKILL (gpu_executor.STOP_WAITS 와 같은 대기) 로 끄고 코드 0 으로 끝난다.

옛 창 방식과 함께:
  * 비전(orchestrate): 실행 상태 파일 잠금(StateFile 단일 Writer)을 못 잡으면 blocked — GPU 를 두 번 띄우지 않는다.
  * 데이터 서버: 소스 포트가 이미 열려 있거나 다른 server_stream_multi_source.py 가 돌면 blocked (계속 띄우려 하지 않는다).
  * 슬라이스 · 비전(직접): 다른 slice360_live.py · vision_infer.py 가 돌면 blocked.

보고: Redis `ds_reported:<role>` (JSON, TTL ≈ 3×poll) + <state>/reported_<role>.json
  {"role", "desired_version", "desired_source", "components": {이름: {key, pid, status, since, restarts, reason}},
   "converged", "errors", "ts"}    status: running | starting | stopped | crashed | blocked
맞춤 상태가 바뀔 때만 orchestration_stream 에 한 건 (request_id "desired-v<version>", desired_version · converged 칸):
  새 버전을 보고 아직 안 맞음 → requested · 맞음 → applied (돌 것이 없으면 stopped) · 같은 버전에서 어긋남 → degraded.
"""
import argparse
import ast
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ds_resources import desired as ds  # noqa: E402
from ds_resources.streams import StreamWriter  # noqa: E402

ROLES = {"core": ("server", "slice"), "vision": ("vision",), "all": ("server", "slice", "vision")}
ORDER = ("server", "slice", "vision")
STOP_WAITS = (6.0, 2.0, 1.0)          # SIGINT 뒤 6초 · SIGTERM 뒤 2초 · SIGKILL 뒤 1초 (gpu_executor.STOP_WAITS 와 같다)
SLICE_DELAY = 3.0                     # 런처 $SliceDelay
VISION_DELAY = 5.0                    # 런처 $VisionDelay
RUNNING_AFTER_S = 2.0                 # 이만큼 살아 있으면 running (그 전은 starting)
BACKOFF_S = 2.0
BACKOFF_MAX_S = 60.0
STABLE_S = 60.0                       # 이만큼 돌면 다음 고장의 대기를 처음부터
SLICE_V = -30                         # 런처 $SliceV (원하는 상태에 v 가 없을 때)
VISION_ENV = {"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
SERVER_SCRIPT = "server_stream_multi_source.py"
SLICE_SCRIPT = "slice360_live.py"
VISION_SCRIPT = "vision_infer.py"
REPORT_PREFIX = "ds_reported:"
STATUSES = ("running", "starting", "stopped", "crashed", "blocked")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def state_dir(arg=None):
    return Path(arg or os.environ.get("DS_STATE_DIR") or (REPO / "state")).resolve()


def child_python():
    return os.environ.get("DS_PYTHON") or sys.executable


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.stem + "-", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


# ------------------------------------------------------------ 프로필
def write_profile(path, profile):
    """원하는 상태의 프로필을 파일로. 내용(profile_version 뺀 것)이 같으면 쓰지 않는다 — 비전이 mtime 을 보고 다시 읽으므로.
    (profile_version, 썼나)."""
    path = Path(path)
    want = copy.deepcopy(profile or {})
    want.pop("profile_version", None)
    old = None
    try:
        old = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        pass
    ver = 0
    if isinstance(old, dict):
        try:
            ver = int(old.get("profile_version") or 0)
        except (TypeError, ValueError):
            ver = 0
        cur = dict(old)
        cur.pop("profile_version", None)
        if ds._key(cur) == ds._key(want):
            return ver, False
    _atomic_json(path, dict(want, profile_version=ver + 1))
    return ver + 1, True


# ------------------------------------------------------------ 옛 창 방식 · 다른 프로세스 찾기
def server_ports():
    """server_stream_multi_source.SOURCES 를 import 하지 않고 읽는다 (av · cv2 를 끌어오지 않게)."""
    try:
        tree = ast.parse((REPO / SERVER_SCRIPT).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets):
                return dict(ast.literal_eval(node.value))
    except (OSError, SyntaxError, ValueError):
        pass
    return {}


def port_in_use(port, host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        return s.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        s.close()


def find_procs(script, exclude_pgids=()):
    """명령줄 인자 중 하나의 이름이 script 인 프로세스 번호들 (이 감독자와 그 자식 묶음은 뺀다)."""
    me, out = os.getpid(), []
    excl = set(exclude_pgids)
    for d in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(d.name)
            if pid == me:
                continue
            args = (d / "cmdline").read_bytes().split(b"\0")
            if not any(os.path.basename(a.decode("utf-8", "replace")) == script for a in args if a):
                continue
            if os.getpgid(pid) in excl:
                continue
            out.append(pid)
        except (OSError, ValueError):
            continue
    return out


# ------------------------------------------------------------ 구성 요소
class Component:
    """한 구성 요소의 공통 상태. key = 지금 맞추려는 열쇠(None = 꺼져 있어야 함)."""

    def __init__(self, name, sup):
        self.name, self.sup = name, sup
        self.key = None
        self.status, self.reason = "stopped", ""
        self.since = sup.wall()
        self.restarts = 0            # 이 열쇠에서 다시 띄운 횟수 (보고)
        self.fails = 0               # 연달아 고장 난 횟수 (대기 계산)
        self.next_try = 0.0
        self.started_at = None

    def set(self, status, reason=""):
        if status != self.status:
            self.since = self.sup.wall()
        self.status, self.reason = status, reason

    def pid(self):
        return None

    def report(self):
        return {"key": self.key, "pid": self.pid(), "status": self.status, "since": self.since,
                "restarts": self.restarts, "reason": self.reason}

    # 아래는 종류마다
    def alive(self):
        raise NotImplementedError

    def blocked(self, d):
        return ""

    def start(self, d):
        raise NotImplementedError

    def stop(self, reason):
        raise NotImplementedError

    def healthy(self):
        """살아 있는 동안: running 으로 볼 만한가"""
        return self.started_at is not None and self.sup.clock() - self.started_at >= RUNNING_AFTER_S

    def tick(self):
        pass


class ProcComponent(Component):
    """자식 프로세스 하나 (자기 세션, 로그는 <state>/logs/sup-<이름>.log)."""

    script = ""

    def __init__(self, name, sup):
        super().__init__(name, sup)
        self.proc = None

    def pid(self):
        return self.proc.pid if self.proc is not None else None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def command(self, d):
        raise NotImplementedError

    def env(self):
        return dict(os.environ, PYTHONUNBUFFERED="1")

    def blocked(self, d):
        other = self.sup.find_procs(self.script, self.sup.child_pgids())
        if other:
            return "다른 %s 가 돌고 있음 (pid %s) — 런처 창을 끄거나 그쪽을 쓰세요" % (
                self.script, ", ".join(map(str, sorted(other)[:5])))
        return ""

    def start(self, d):
        cmd = self.command(d)
        lp = self.sup.logs / ("sup-%s.log" % self.name)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "ab") as lf:
            lf.write(("\n=== %s 시작: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd))).encode("utf-8"))
            lf.flush()
            self.proc = self.sup.popen(cmd, cwd=str(REPO), env=self.env(), stdout=lf, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.started_at = self.sup.clock()
        self.sup.log("%s 띄움 (pid %s)" % (self.name, self.proc.pid))

    def stop(self, reason):
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            for sig, wait in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), STOP_WAITS):
                try:
                    self.sup.killpg(p.pid, sig)
                except OSError:
                    break                                   # 이미 없다
                t0 = self.sup.clock()
                while p.poll() is None and self.sup.clock() - t0 < wait:
                    self.sup.sleep(0.1)
                if p.poll() is not None:
                    break
            self.sup.log("%s 끔 (%s)" % (self.name, reason))
        self.proc, self.started_at = None, None

    def exit_reason(self):
        rc = self.proc.poll() if self.proc is not None else None
        return "스스로 끝남 (코드 %s) — 로그 %s" % (rc, self.sup.logs / ("sup-%s.log" % self.name))

    def reap(self):
        self.proc, self.started_at = None, None


class ServerComponent(ProcComponent):
    script = SERVER_SCRIPT

    def command(self, d):
        return [self.sup.python, str(REPO / SERVER_SCRIPT), "--on", *ds.vision_sources(d),
                "--robot-cam", str(d["server"]["robot_cam"]), "--redis", self.sup.redis_url]

    def blocked(self, d):
        why = super().blocked(d)
        if why:
            return why
        busy = ["%s:%d" % (s, self.sup.ports[s]) for s in ds.vision_sources(d)
                if s in self.sup.ports and self.sup.port_in_use(self.sup.ports[s])]
        if busy:
            return "포트가 이미 열려 있음 (%s) — 다른 데이터 서버(런처 창 · 컨테이너)가 돌고 있음" % ", ".join(busy)
        return ""


class SliceComponent(ProcComponent):
    script = SLICE_SCRIPT

    def command(self, d):
        s = d["slice"]
        cmd = [self.sup.python, str(REPO / "e2p" / SLICE_SCRIPT)]
        if s.get("u") not in (None, ""):
            cmd += ["--u", str(s["u"])]
        cmd += ["--v", str(SLICE_V if s.get("v") in (None, "") else s["v"]), "--redis", self.sup.redis_url]
        return cmd


class VisionProc(ProcComponent):
    """orchestrate 가 아닐 때 — vision_infer.py 한 프로세스가 모든 소스를 맡는다."""
    script = VISION_SCRIPT

    def env(self):
        return dict(super().env(), **VISION_ENV)

    def command(self, d):
        v = d["vision"]
        cmd = [self.sup.python, str(REPO / "vision" / VISION_SCRIPT), "--devices", *ds.vision_sources(d),
               "--models", *v["models"], "--profile", str(self.sup.profile_path)]
        if v.get("max_fps"):
            cmd += ["--max-fps", *["%s=%g" % (k, float(f)) for k, f in sorted(v["max_fps"].items())]]
        return cmd


def nothing_running(k, rid):
    """cli run 의 끝내는 조건과 같다 — 감독자는 끝내지 않고 고장으로 보고 대기 뒤 다시 요청한다."""
    units = k.state.units()
    return not k.ex.active_units() and not any(
        u.get("kind") == "cpu" and u.get("status") in ("starting", "running") for u in units.values()) \
        and not any(u.get("status") == "local" for u in units.values() if u.get("request_id") == rid)


def build_controller(sup):
    """cli.build 와 같다 — 경로만 감독자의 state 폴더. 잠금을 못 잡으면 (None, 이유)."""
    import redis

    from ds_resources import controller as ctl
    from ds_resources import gpu_executor as ge
    from ds_resources import memory as mm
    from ds_resources.backends import resolve_cluster
    from ds_resources.recorder import load_source
    from ds_resources.state import StateFile
    path = sup.state_dir / "orchestrator_state.json"
    st = StateFile(path, writer=True)
    if not st.writer:
        st.close()
        return None, "다른 오케스트레이터가 실행 상태 파일을 잡고 있음 (%s.lock) — 런처 창의 비전을 끄세요" % path
    r = redis.Redis.from_url(sup.redis_url, socket_connect_timeout=2, socket_timeout=5)
    mem = mm.ResourceMemory(sup.state_dir / "resource_memory.json", writer=False)
    ex = ge.GpuExecutor(st, r, python=sup.python, log=sup.log)
    src_path, cpu_path = resolve_cluster(sup.cluster)
    cluster = cpu = None
    if src_path:
        try:
            cluster = load_source(src_path, memory=mem, log=sup.log, probe_net=False)
            import yaml
            tasks = (yaml.safe_load(open(REPO / "vision" / "model_registry.yaml", encoding="utf-8")) or {}).get("cpu_tasks") or {}
            cpu = load_source(cpu_path, tasks=tasks, redis_url=sup.redis_public or sup.redis_url)
        except Exception as e:  # noqa: BLE001 — k3s 쪽이 안 되면 local 로
            sup.log("k3s 백엔드를 못 불러옴 (%s) — local 로" % e)
            cluster = cpu = None
    return ctl.Controller(r, st, ex, mem, cluster=cluster, cpu_executor=cpu, log=sup.log), ""


class VisionOrch(Component):
    """orchestrate 일 때 — 오케스트레이터 컨트롤러를 이 프로세스 안에서 (cli run 을 띄우지 않는다)."""

    def __init__(self, name, sup):
        super().__init__(name, sup)
        self.k, self.rid, self._why = None, None, ""

    def pid(self):
        return os.getpid() if self.k is not None else None

    def alive(self):
        if self.k is None:
            return False
        if nothing_running(self.k, self.rid) and not self.k.pending_sources(self.rid):
            req = self.k.state.data["requests"].get(self.rid) or {}
            why = [x["reason"] for x in (req.get("plan") or {}).get("refused", [])]
            self._why = "돌고 있는 비전 작업이 없음" + ((" — " + " / ".join(why)) if why else "")
            return False
        return True

    def healthy(self):
        units = self.k.state.units().values() if self.k is not None else ()
        return any(u.get("request_id") == self.rid and
                   ((u.get("kind") == "gpu" and u.get("status") == "loaded") or
                    (u.get("kind") == "cpu" and u.get("status") == "running") or u.get("status") == "local")
                   for u in units)

    def blocked(self, d):
        k, why = self.sup.controller_factory(self.sup)
        if k is None:
            return why
        try:
            n = k.cleanup_previous()
            other = self.sup.find_procs(VISION_SCRIPT, self.sup.child_pgids())
        except Exception:
            k.state.close()                         # 잠금을 쥔 채로 남기지 않는다
            raise
        if n:
            self.sup.log("지난 실행이 남긴 작업 %d개를 정리함" % n)
        if other:                                   # 오케스트레이터가 아닌 런처 창(직접 비전)이 GPU 를 쓰고 있다
            k.state.close()
            return "다른 %s 가 돌고 있음 (pid %s) — 런처 창의 비전을 끄세요" % (
                VISION_SCRIPT, ", ".join(map(str, sorted(other)[:5])))
        self.k = k
        return ""

    def start(self, d):
        v = d["vision"]
        prof = None
        try:
            prof = json.loads(self.sup.profile_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            prof = None
        self.rid = self.k.submit(ds.vision_sources(d), v["models"], prof, str(self.sup.profile_path),
                                 target_fps=v.get("max_fps") or None)
        self.started_at = self.sup.clock()
        req = self.k.state.data["requests"].get(self.rid) or {}
        self.sup.log("vision 요청 %s — %s" % (self.rid, req.get("mode") or ""))

    def tick(self):
        if self.k is None:
            return
        for e in self.k.poll():
            self.sup.log("vision %-9s %s" % (e["event"], e["reason"]))
        new = self.k.replan_if_needed(self.rid)          # 빠졌던 입력 대기 소스에 영상이 들어옴 → 다시 배치
        if new:
            self.sup.log("vision 다시 배치: %s → %s" % (self.rid, new))
            self.rid = new

    def exit_reason(self):
        return self._why or "돌고 있는 비전 작업이 없음"

    def reap(self):
        self._close("작업 없음")

    def stop(self, reason):
        if self.k is None:
            return
        self._close(reason)
        self.sup.log("vision 끔 (%s)" % reason)

    def _close(self, reason):
        k, self.k = self.k, None
        try:
            if self.rid:
                k.stop(self.rid, reason)
            k.flush_outbox()
        finally:
            k.state.close()
            self.rid, self.started_at = None, None


# ------------------------------------------------------------ 감독자
class Supervisor:
    def __init__(self, role, redis_client=None, state=None, redis_url="redis://127.0.0.1:6380/0", poll=2.0,
                 popen=subprocess.Popen, clock=time.monotonic, wall=time.time, sleep=time.sleep, killpg=os.killpg,
                 port_in_use=port_in_use, find_procs=find_procs, controller_factory=build_controller, python=None,
                 cluster="auto", redis_public=None, log=log):
        if role not in ROLES:
            raise ValueError("role 은 %s 중 하나" % ", ".join(ROLES))
        self.role, self.r, self.redis_url, self.poll = role, redis_client, redis_url, float(poll)
        self.state_dir = state_dir(state)
        self.logs = self.state_dir / "logs"
        self.desired_path = self.state_dir / "desired.json"
        self.profile_path = self.state_dir / "profile.json"
        self.report_path = self.state_dir / ("reported_%s.json" % role)
        self.popen, self.clock, self.wall, self.sleep, self.killpg = popen, clock, wall, sleep, killpg
        self.port_in_use, self.find_procs, self.controller_factory = port_in_use, find_procs, controller_factory
        self.python = python or child_python()
        self.cluster, self.redis_public, self.log = cluster, redis_public, log
        self.ports = server_ports()
        self.names = ROLES[role]
        self.comps = {}
        if "server" in self.names:
            self.comps["server"] = ServerComponent("server", self)
        if "slice" in self.names:
            self.comps["slice"] = SliceComponent("slice", self)
        if "vision" in self.names:
            self.comps["vision"] = VisionProc("vision", self)
        self.writer = StreamWriter(redis_client, maxlen=2000) if redis_client is not None else None
        self.version, self.source, self.errors = 0, "default", []
        self.converged = None
        self.emitted = None                         # (version, converged) — 스트림에 마지막으로 남긴 것
        self.stop_event = threading.Event()

    def child_pgids(self):
        return [c.pid() for c in self.comps.values() if isinstance(c, ProcComponent) and c.pid()]

    # -------------------------------------------- 열쇠
    @staticmethod
    def wanted(d):
        """구성 요소별 열쇠. 서버가 바뀌면 모두 다시(런처) — 슬라이스 · 비전 열쇠에 서버 열쇠를 붙인다."""
        k = ds.component_keys(d)
        out = {"server": k["server"]}
        for n in ("slice", "vision"):
            out[n] = None if k[n] is None else ds._key([k["server"], k[n]])
        return out

    def _vision_comp(self, d):
        """orchestrate 에 맞는 종류로 (바뀌면 옛것을 끄고 바꾼다 — 열쇠에도 orchestrate 가 있어 어차피 다시 띄운다)"""
        cls = VisionOrch if d["vision"].get("orchestrate") else VisionProc
        c = self.comps["vision"]
        if type(c) is cls:
            return c
        if c.alive():
            c.stop("방식 바뀜")
        elif c.started_at is not None:
            c.reap()
        nc = cls("vision", self)
        nc.key = c.key
        self.comps["vision"] = nc
        return nc

    # -------------------------------------------- 한 바퀴
    def tick(self):
        d, self.source = ds.read(self.r, self.desired_path)
        self.version = int(d.get("version") or 0)
        self.errors = ds.validate(d)
        if not self.errors:
            self.reconcile(d)
        self.report()
        return self.converged

    def reconcile(self, d):
        want = self.wanted(d)
        now = self.clock()
        if "vision" in self.comps and want["vision"] is not None:
            ver, wrote = write_profile(self.profile_path, d["vision"].get("profile"))
            if wrote:
                self.log("프로필 %s (profile_version %d)" % (self.profile_path, ver))
            self._vision_comp(d)
        # 열쇠가 바뀐 것은 먼저 끈다 (비전 → 슬라이스 → 서버, 런처 Stop-Windows 순서)
        for name in reversed(ORDER):
            c = self.comps.get(name)
            if c is None or c.key == want[name]:
                continue
            if c.alive():
                c.stop("원하는 상태가 바뀜 (v%d)" % self.version)
            elif c.started_at is not None:
                c.reap()
            c.key, c.restarts, c.fails, c.next_try = want[name], 0, 0, 0.0
            c.set("stopped")
        for name in ORDER:
            c = self.comps.get(name)
            if c is None:
                continue
            c.tick()
            if c.key is None:
                c.set("stopped")
                continue
            if c.started_at is not None:                # 띄운 것
                if c.alive():
                    if c.healthy():
                        c.set("running")
                        if c.fails and now - c.started_at >= STABLE_S:
                            c.fails = 0
                    else:
                        c.set("starting")
                    continue
                why = c.exit_reason()
                c.reap()
                c.fails += 1
                c.restarts += 1
                wait = min(BACKOFF_MAX_S, BACKOFF_S * (2 ** (c.fails - 1)))
                c.next_try = now + wait
                c.set("crashed", "%s · %d초 뒤 다시 (%d번째)" % (why, wait, c.restarts))
                self.log("%s %s" % (name, c.reason))
                continue
            if now < c.next_try:
                continue                                # 고장 대기 중 (crashed 그대로)
            dep = self._waiting_for_server(name, now)
            if dep:
                c.set("starting", dep)
                continue
            why = c.blocked(d)
            if why:
                if c.status != "blocked" or c.reason != why:
                    self.log("%s 막힘: %s" % (name, why))
                c.set("blocked", why)
                continue
            try:
                c.start(d)
                c.set("starting")
            except Exception as e:  # noqa: BLE001 — 띄우기 실패도 고장으로 보고 대기 뒤 다시
                if isinstance(c, VisionOrch) and c.k is not None:
                    c._close("띄우기 실패")
                c.started_at = None
                c.fails += 1
                c.restarts += 1
                wait = min(BACKOFF_MAX_S, BACKOFF_S * (2 ** (c.fails - 1)))
                c.next_try = now + wait
                c.set("crashed", "띄우기 실패: %s: %s · %d초 뒤 다시" % (type(e).__name__, e, wait))
                self.log("%s %s" % (name, c.reason))

    def _waiting_for_server(self, name, now):
        srv = self.comps.get("server")
        if name == "server" or srv is None or srv.key is None:
            return ""
        delay = SLICE_DELAY if name == "slice" else VISION_DELAY
        if srv.status != "running" or srv.started_at is None:
            return "데이터 서버를 기다림"
        if now - srv.started_at < delay:
            return "데이터 서버가 뜬 뒤 %g초 기다림" % delay
        return ""

    # -------------------------------------------- 보고
    def is_converged(self):
        if self.errors:
            return False
        for c in self.comps.values():
            if c.key is None and c.status != "stopped":
                return False
            if c.key is not None and c.status != "running":
                return False
        return True

    def report(self):
        conv = self.is_converged()
        rep = {"role": self.role, "desired_version": self.version, "desired_source": self.source,
               "components": {n: c.report() for n, c in self.comps.items()}, "converged": conv,
               "errors": list(self.errors), "ts": self.wall()}
        try:
            _atomic_json(self.report_path, rep)
        except OSError as e:
            self.log("보고 파일을 못 씀: %s" % e)
        if self.r is not None:
            try:
                self.r.set(REPORT_PREFIX + self.role, json.dumps(rep, ensure_ascii=False),
                           ex=max(5, int(round(3 * self.poll))))
            except Exception:  # noqa: BLE001 — Redis 가 없어도 맞추기는 계속
                pass
        self.converged = conv
        self._emit(conv)
        return rep

    def _emit(self, conv, final=None):
        mark = (self.version, conv, final)
        if mark == self.emitted:
            return None
        prev = self.emitted
        self.emitted = mark
        anything = any(c.key is not None for c in self.comps.values())
        if final:
            event, reason = "stopped", final
        elif conv:
            event = "applied" if anything else "stopped"
            reason = "원하는 상태 v%d 에 맞춤 (%s)" % (self.version, self.role)
        elif prev is None or prev[0] != self.version:
            event, reason = "requested", "원하는 상태 v%d 를 맞추는 중 (%s)" % (self.version, self.role)
        else:
            bad = ["%s %s%s" % (n, c.status, (": " + c.reason) if c.reason else "")
                   for n, c in self.comps.items() if (c.key is None) != (c.status == "stopped") or
                   (c.key is not None and c.status != "running")]
            event, reason = "degraded", "원하는 상태 v%d 와 어긋남 — %s" % (self.version, " / ".join(bad) or ", ".join(self.errors))
        detail = {"role": self.role, "components": {n: {"status": c.status, "reason": c.reason}
                                                    for n, c in self.comps.items()}}
        if self.errors:
            detail["errors"] = list(self.errors)
        if self.writer is not None:
            self.writer.orch(event, "desired-v%d" % self.version, reason=reason, detail=detail,
                             desired_version=self.version, converged=conv)
        return event

    # -------------------------------------------- 돌기 · 끝내기
    def shutdown(self, reason="감독자 끝남"):
        for name in reversed(ORDER):
            c = self.comps.get(name)
            if c is None:
                continue
            if c.alive() or isinstance(c, VisionOrch) and c.k is not None:
                c.stop(reason)
            c.started_at = None
            c.set("stopped", reason)
        rep = self.report()
        self._emit(rep["converged"], final=reason)
        return rep

    def run(self):
        self.log("감독자 시작 — 역할 %s · state %s · Python %s" % (self.role, self.state_dir, self.python))
        try:
            while not self.stop_event.is_set():
                try:
                    self.tick()
                except Exception as e:  # noqa: BLE001 — 한 바퀴가 실패해도 감독자는 끝나지 않는다
                    self.log("한 바퀴 실패: %s: %s" % (type(e).__name__, e))
                self.stop_event.wait(self.poll)
        finally:
            self.shutdown()
            self.log("끝")
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="원하는 상태(ds_desired)를 맞추는 감독자")
    ap.add_argument("--role", choices=tuple(ROLES), required=True)
    ap.add_argument("--redis", default=os.environ.get("DS_REDIS", "redis://127.0.0.1:6380/0"))
    ap.add_argument("--redis-public", help="파이 파드가 닿는 Redis 주소 (orchestrate · k3s 때)")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--state-dir", help="상태 폴더 (기본 $DS_STATE_DIR, 없으면 저장소 state/)")
    ap.add_argument("--cluster", default=os.environ.get("DS_CLUSTER", "auto"), help="auto | none | 모듈:클래스")
    a = ap.parse_args(argv)
    import redis
    r = redis.Redis.from_url(a.redis, socket_connect_timeout=2, socket_timeout=5)
    if "vision" in ROLES[a.role]:
        for k, v in VISION_ENV.items():
            os.environ.setdefault(k, v)
    sup = Supervisor(a.role, r, a.state_dir, redis_url=a.redis, poll=a.poll, cluster=a.cluster,
                     redis_public=a.redis_public)

    def on_signal(*_):
        sup.stop_event.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    return sup.run()


if __name__ == "__main__":
    sys.exit(main())
