"""k3s 노드 원천 — 자원 기반 비전 오케스트레이션 전용 (k3s 의존은 이 폴더에만, 2026-11-24).

core(ds_resources)는 이 파일을 import 하지 않는다. 자원 기록기가 경로 문자열
`orchestrator.k3s_source:K3sNodeSource` 로 불러와 자기 스레드 안에서 돌린다 — 그래서 기억 파일의 Writer 는
기록기 하나다(단일 Writer). 컨트롤러는 같은 클래스를 cluster 백엔드(check · nodes)로 쓴다.

읽는 것 (kubectl JSON, 호출마다 시간 제한)
  Node            ready · role(ds.role) · taint · 용량 · 할당 가능량 · 신원(bootID · 커널 · OS) · 테일넷 주소
  특성 · 온도      arch 는 .status.nodeInfo.architecture, 나머지는 파이가 올린 라벨 · 애너테이션 (2026-11-25)
                   ds.model · ds.sensors(쉼표 구분) · ds.accel · ds.cores_reserved ·
                   ds.thermal/temp_c · ds.thermal/throttled
                   라벨 값에는 공백 · 쉼표를 쓸 수 없어(k8s 규칙) 기기 이름 · 센서 목록은 애너테이션으로 올린다 —
                   손으로 라벨에 붙인 것도 읽는다(라벨 > 같은 이름의 애너테이션). 없는 칸은 None 이다(추측 안 함).
                   올리는 쪽은 cluster_up.sh caps (파이에서 pi_node.sh caps 로 읽어 kubectl 로 붙인다).
  metrics API     CPU(n코어) · 메모리 — 측정 시각(timestamp)이 STALE_S 보다 오래되면 stale
GPU 는 여기서 읽지 않는다(metrics-server 는 GPU 를 모른다) — data_stream 기기의 nvidia-smi 가 따로 잰다.
data_stream 기기 자신의 노드는 건너뛴다 — 그 기록은 로컬 기록기가 쓴다. 로컬 노드 = 다음 중 하나라도 맞는 노드
(M11 R5): InternalIP == 테일넷 주소 · 노드 이름 == probe.node_name() · 호스트 이름 · DS_NODE_NAME · DS_K3S_LOCAL_NODE
(쉼표로 여럿). install_server.sh 로 깐 클러스터는 IP 로, 이름만 같은 클러스터는 이름으로, 둘 다 다르면 환경변수로.

네트워크 (기록기가 부른다 — 처음 · 신원 변화 · 기억의 net_due, 별도 스레드 — 기록은 기록기 tick 이 쓴다)
  tailscale ping   지연(ms) · 직접 연결인지 중계(DERP)인지
  대역폭           그 노드에 짧은 파드를 띄워 서버의 HTTP 로 PROBE_BYTES 를 받아 가게 하고, **서버 쪽에서** 보내기
                   시작부터 끝까지 잰다(서버 → 노드 방향 — 프레임이 가는 방향). busybox date 는 %N 을 몰라
                   파드 쪽 시계로는 못 잰다(2026-11-24).
  결과의 net_status  ok · failed(파드 거절 · 시간 초과 등 — 기억이 물러남 뒤 다시 묻는다) ·
                     unsupported(테일넷 주소가 없어 측정 서버를 열 수 없음 — 다시 묻지 않는다)

클러스터 객체 (M11 R2): orchestrator/k3s/ 의 객체(PriorityClass ds-vision-low …)는 install_server.sh 가 폴더째
apply 한다. 실행 중에는 만들지 않는다 — check() 가 없음을 caps {"cpu_tasks": False, "why"} 로 알리고 컨트롤러는
k3s 모드 · 노드 정보를 그대로 둔 채 파이 CPU 배치만 막는다.
"""
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

KUBECTL_TIMEOUT = 5.0
STALE_S = 180.0
PROBE_BYTES = 8 * 1024 * 1024
PROBE_IMAGE = "busybox:1.36"
NAMESPACE = "ds-system"
K3S_DIR = Path(__file__).resolve().parent / "k3s"     # 클러스터 범위 객체 — install_server.sh 가 폴더째 apply
KIND_SHOWN = {"priorityclasses": "PriorityClass"}   # kubectl 오류의 자원 이름 → 사람이 읽는 이름 (없으면 그대로)
CAPS_TTL_S = 60.0                                     # 클러스터 객체 확인 결과를 이만큼 다시 쓴다 (poll 마다 묻지 않게)


def _env():
    return dict(os.environ, KUBECONFIG=os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config"))


def kubectl_json(*args, timeout=KUBECTL_TIMEOUT):
    exe = shutil.which("kubectl")
    if not exe:
        raise RuntimeError("kubectl 없음")
    p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout, env=_env())
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip()[:200] or "kubectl 실패")
    return json.loads(p.stdout)


def cpu_m(q):
    q = str(q)
    if q.endswith("n"):
        return int(q[:-1]) / 1e6
    if q.endswith("u"):
        return int(q[:-1]) / 1e3
    if q.endswith("m"):
        return float(q[:-1])
    return float(q) * 1000.0


def mem_mb(q):
    q = str(q)
    for suf, mul in (("Ki", 1 / 1024), ("Mi", 1), ("Gi", 1024), ("k", 1 / 1048.576), ("M", 1 / 1.048576)):
        if q.endswith(suf):
            return float(q[:-len(suf)]) * mul
    return float(q) / 1048576.0


def iso_epoch(s):
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def local_ts_ip():
    try:
        return subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.split()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def tailscale_ping(host, timeout=5.0):
    """(지연 ms, 중계인가) — 못 재면 (None, None)"""
    try:
        out = subprocess.run(["tailscale", "ping", "-c", "1", "--timeout", "%ds" % timeout, host],
                             capture_output=True, text=True, timeout=timeout + 2).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    m = re.search(r"via (\S+) in ([\d.]+)ms", out)
    if not m:
        return None, None
    return float(m.group(2)), m.group(1).upper().startswith("DERP")


class _Blob(http.server.BaseHTTPRequestHandler):
    payload = os.urandom(PROBE_BYTES)
    timings = None                                       # 측정마다 새 사전을 꽂는다 (클래스를 만들 때)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        t0 = time.perf_counter()
        view, step = memoryview(self.payload), 256 * 1024
        for i in range(0, len(view), step):
            self.wfile.write(view[i:i + step])
        self.wfile.flush()
        if self.timings is not None:
            self.timings["s"] = time.perf_counter() - t0

    def log_message(self, *a):
        pass


def bandwidth_probe(node, server_ip, timeout=60.0):
    """그 노드에 파드를 띄워 서버에서 PROBE_BYTES 를 받아 오는 시간으로 Mbit/s. 못 재면 None."""
    return bandwidth_probe_ex(node, server_ip, timeout)[0]


def bandwidth_probe_ex(node, server_ip, timeout=60.0):
    """(Mbit/s 또는 None, 못 잰 이유 또는 None)"""
    timings = {}
    handler = type("_ProbeBlob", (_Blob,), {"timings": timings})
    srv = http.server.ThreadingHTTPServer((server_ip, 0), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    name = "ds-netprobe-%s-%d" % (re.sub(r"[^a-z0-9-]", "-", node.lower()), int(time.time()))
    spec = {"apiVersion": "v1", "kind": "Pod",
            "metadata": {"name": name, "namespace": NAMESPACE, "labels": {"app": "ds-netprobe"}},
            "spec": {"nodeName": node, "restartPolicy": "Never", "priorityClassName": "ds-vision-low",
                     "tolerations": [{"key": "ds.role", "operator": "Exists", "effect": "NoSchedule"}],
                     "containers": [{"name": "c", "image": PROBE_IMAGE,
                                     "command": ["wget", "-q", "-O", "/dev/null", "http://%s:%d/" % (server_ip, port)],
                                     "resources": {"requests": {"cpu": "10m", "memory": "8Mi"},
                                                   "limits": {"memory": "32Mi"}}}]}}
    exe = shutil.which("kubectl")
    try:
        subprocess.run([exe, "create", "namespace", NAMESPACE], capture_output=True, timeout=10, env=_env())
        p = subprocess.run([exe, "apply", "-f", "-"], input=json.dumps(spec), capture_output=True, text=True,
                           timeout=15, env=_env())
        if p.returncode != 0:
            return None, "측정 파드 거절: %s" % ((p.stderr or p.stdout).strip()[:160] or "kubectl apply 실패")
        t0, ph = time.time(), ""
        while time.time() - t0 < timeout:
            ph = subprocess.run([exe, "-n", NAMESPACE, "get", "pod", name, "-o", "jsonpath={.status.phase}"],
                                capture_output=True, text=True, timeout=10, env=_env()).stdout
            if ph in ("Succeeded", "Failed"):
                break
            time.sleep(2)
        secs = timings.get("s")
        if secs:
            return round(PROBE_BYTES * 8 / 1e6 / secs, 1), None
        return None, "측정 파드가 받아 가지 못함 (phase %s, %d초)" % (ph or "?", timeout)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return None, "측정 실패: %s" % str(e)[:160]
    finally:
        subprocess.run([exe, "-n", NAMESPACE, "delete", "pod", name, "--wait=false"], capture_output=True,
                       timeout=10, env=_env())
        srv.shutdown()
        srv.server_close()


def _tagged(labels, ann, key):
    """라벨 > 같은 이름의 애너테이션 (없으면 None). 라벨 값에는 공백 · 쉼표를 쓸 수 없어 기기 이름 ·
    센서 목록은 애너테이션으로 올라온다 — 손으로 라벨에 붙인 것도 읽는다."""
    v = (labels or {}).get(key)
    if v is None or str(v).strip() == "":
        v = (ann or {}).get(key)
    v = None if v is None else str(v).strip()
    return v or None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _flag(v):
    """throttled 애너테이션 → 참 · 거짓 · None. 파이의 get_throttled 값(0x0 · 0x50005)을 그대로 올리니
    숫자면 0 이 아닌지로 본다. true/false 도 받는다."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    try:
        return int(s, 0) != 0
    except ValueError:
        return None


def node_caps(info, labels, ann):
    """노드 특성 {arch, model, sensors, accel, cores_reserved} — arch 는 k3s 가 아는 값, 나머지는 파이가 올린 것.

    카메라는 /dev 로 확인할 수 없다 — 파이 5 의 CSI 카메라는 libcamera 쪽이라 /dev/video* 에 캡처 장치로
    안 보인다(파이가 올리는 목록에 camera 가 빠진다). 그래서 **ds.source 라벨**(그 노드가 영상 소스를 맡고
    있다는 사실, cluster_up.sh 가 nodes.json 을 보고 붙인다)을 카메라가 달렸다는 근거로 쓴다.
    이미 올라온 목록에 **더하기만** 한다. 목록이 아예 없으면(모름) 그대로 모름으로 둔다 —
    모름을 「camera 하나만 있음」으로 바꾸면 다른 센서를 찾는 작업이 잘못 떨어진다."""
    sensors = _tagged(labels, ann, "ds.sensors")
    cores = _num(_tagged(labels, ann, "ds.cores_reserved"))
    got = sorted({x.strip() for x in sensors.split(",") if x.strip()}) if sensors else None
    if got is not None and (labels or {}).get("ds.source"):
        got = sorted(set(got) | {"camera"})
    return {"arch": (info or {}).get("architecture") or None,
            "model": _tagged(labels, ann, "ds.model"),
            "sensors": got,
            "accel": _tagged(labels, ann, "ds.accel"),
            "cores_reserved": None if cores is None else max(0, int(cores))}


def node_thermal(ann):
    """노드 온도 {temp_c, throttled} 또는 None — 애너테이션이 둘 다 없으면 None(파이가 아직 안 올렸음)."""
    temp, thr = (ann or {}).get("ds.thermal/temp_c"), (ann or {}).get("ds.thermal/throttled")
    if temp is None and thr is None:
        return None
    return {"temp_c": _num(temp), "throttled": _flag(thr)}


def local_node_names():
    """이 기기를 가리킬 수 있는 노드 이름들 — probe.node_name() · 호스트 이름 · DS_NODE_NAME · DS_K3S_LOCAL_NODE"""
    names = {socket.gethostname(), socket.gethostname().lower()}
    try:
        from ds_resources import probe                   # core → 플러그인 방향은 괜찮다 (반대만 막는다)
        names.add(probe.node_name())
    except Exception:  # noqa: BLE001
        pass
    for var in ("DS_NODE_NAME", "DS_K3S_LOCAL_NODE"):
        names |= {x.strip() for x in (os.environ.get(var) or "").split(",") if x.strip()}
    return {n for n in names if n}


class K3sNodeSource:
    """기록기 플러그인 (samples · net_updates) + 컨트롤러 cluster 백엔드 (check · nodes)."""

    def __init__(self, memory=None, log=print, probe_net=True, clock=time.time, local_names=None):
        self.mem, self.log, self.probe_net, self.clock = memory, log, probe_net, clock
        self.local_ip = local_ts_ip()
        self.local_names = set(local_names) if local_names is not None else local_node_names()
        self._caps, self._caps_at = None, None
        self.last = {}                                   # 노드 → 마지막 표본 (nodes() 가 쓴다)
        self.net_pending = {}                            # 노드 → 네트워크 측정 결과 (기록기가 가져간다)
        self.net_running = set()
        self.lock = threading.Lock()

    # ------------------------------------------------------------ 읽기
    def _read(self):
        nodes = kubectl_json("get", "nodes", "-o", "json")["items"]
        try:
            metrics = {m["metadata"]["name"]: m for m in
                       kubectl_json("get", "--raw", "/apis/metrics.k8s.io/v1beta1/nodes")["items"]}
        except Exception:  # noqa: BLE001 — metrics 가 없어도 노드 목록은 낸다(사용량은 stale)
            metrics = {}
        now = self.clock()
        out = []
        for n in nodes:
            name = n["metadata"]["name"]
            ip = next((a["address"] for a in n["status"].get("addresses", []) if a["type"] == "InternalIP"), None)
            if self.is_local(name, ip):
                continue                                 # data_stream 기기 — 로컬 기록기 몫
            info = n["status"].get("nodeInfo", {})
            cap, alloc = n["status"].get("capacity", {}), n["status"].get("allocatable", {})
            ready = any(c["type"] == "Ready" and c["status"] == "True" for c in n["status"].get("conditions", []))
            m = metrics.get(name)
            mts = iso_epoch(m.get("timestamp")) if m else None
            stale = (mts is None) or (now - mts > STALE_S) or not ready
            c_now = cpu_m(m["usage"]["cpu"]) if m else None
            m_now = mem_mb(m["usage"]["memory"]) if m else None
            alloc_c, alloc_m = cpu_m(alloc.get("cpu", "0")), mem_mb(alloc.get("memory", "0"))
            labels = n["metadata"].get("labels", {})
            ann = n["metadata"].get("annotations", {})
            role, source = labels.get("ds.role", ""), labels.get("ds.source")
            out.append({"name": name, "ip": ip, "role": role, "source_name": source, "ready": ready, "stale": stale,
                        "identity": {"boot_id": info.get("bootID"), "kernel": info.get("kernelVersion"),
                                     "os": info.get("osImage")},
                        "baseline": {"arch": info.get("architecture"), "cpu_count": int(cpu_m(cap.get("cpu", "0")) / 1000),
                                     "mem_total_mb": round(mem_mb(cap.get("memory", "0"))),
                                     "disk_total_gb": round(mem_mb(cap.get("ephemeral-storage", "0")) / 1024),
                                     "gpus": []},
                        "caps": node_caps(info, labels, ann), "thermal": node_thermal(ann),
                        "taints": [t["key"] for t in n.get("spec", {}).get("taints", [])],
                        "cpu_alloc_m": alloc_c, "mem_alloc_mb": alloc_m, "cpu_now_m": c_now, "mem_now_mb": m_now,
                        "cpu_pct": (100.0 * c_now / alloc_c) if c_now is not None and alloc_c else None,
                        "mem_pct": (100.0 * m_now / alloc_m) if m_now is not None and alloc_m else None,
                        "measured_at": mts})
        return out

    def is_local(self, name, ip):
        """IP · 이름 중 하나라도 맞으면 data_stream 기기 자신 (M11 R5)"""
        return bool(ip and self.local_ip and ip == self.local_ip) or \
            str(name).lower() in {n.lower() for n in self.local_names}

    # ------------------------------------------------------------ 기록기 플러그인
    def samples(self):
        """[(노드, 표본)] — 표본은 NodeChangeFilter · ResourceMemory 에 그대로 들어간다"""
        rows = self._read()
        with self.lock:
            self.last = {r["name"]: r for r in rows}
        out = []
        for r in rows:
            out.append((r["name"], {"role": r["role"], "source": "k3s", "ready": r["ready"], "stale": r["stale"],
                                    "cpu_pct": r["cpu_pct"], "mem_pct": r["mem_pct"], "mem_mb": r["mem_now_mb"],
                                    "cpu_m": r["cpu_now_m"], "temp_c": None, "gpu": [],
                                    "identity": r["identity"], "baseline": r["baseline"], "ip": r["ip"],
                                    "caps": r["caps"], "thermal": r["thermal"],
                                    "source_name": r["source_name"]}))
        return out

    def want_net(self, node, ip):
        """네트워크 측정을 (다른 스레드에서) 시작한다. 결과는 net_updates() 로 가져간다."""
        if not self.probe_net or node in self.net_running or not ip:
            return
        self.net_running.add(node)

        def run():
            try:
                res = self.measure_net(node, ip)
            except Exception as e:  # noqa: BLE001 — 측정 스레드가 죽어도 실패로 남긴다(물러남 뒤 다시)
                res = {"latency_ms": None, "relay": None, "bw_mbps": None, "net_status": "failed",
                       "error": "%s: %s" % (type(e).__name__, str(e)[:160])}
            res["measured_at"] = time.time()
            with self.lock:
                self.net_pending[node] = res
            self.net_running.discard(node)
        threading.Thread(target=run, name="netprobe-" + node, daemon=True).start()

    def measure_net(self, node, ip):
        """한 노드의 네트워크 — {latency_ms, relay, bw_mbps, net_status, error?}"""
        lat, relay = tailscale_ping(ip)
        out = {"latency_ms": lat, "relay": relay, "bw_mbps": None}
        if not self.local_ip:                            # 측정 서버를 묶을 테일넷 주소가 없다 — 실패가 아니라 해당 없음
            return dict(out, net_status="unsupported", error="이 기기에 테일넷 주소가 없어 대역폭 측정 서버를 열 수 없음")
        bw, err = bandwidth_probe_ex(node, self.local_ip)
        if bw:
            return dict(out, bw_mbps=bw, net_status="ok")
        return dict(out, net_status="failed", error=err or "대역폭을 못 잼")

    def net_updates(self):
        with self.lock:
            d, self.net_pending = self.net_pending, {}
        return d

    # ------------------------------------------------------------ 컨트롤러 cluster 백엔드
    def check(self):
        """(ok, why, caps). ok = API 가 노드를 읽힌다. caps = {"cpu_tasks": 파이에 CPU 작업을 올릴 수 있나, "why"} —
        orchestrator/k3s/ 의 클러스터 객체가 없으면 cpu_tasks False (만들지 않는다 — install_server.sh 가 apply)."""
        try:
            kubectl_json("get", "nodes", "-o", "json", timeout=3)
        except Exception as e:  # noqa: BLE001
            return False, "k3s API 응답 없음 (%s)" % str(e)[:120], {}
        return True, "", self.caps()

    def caps(self):
        now = self.clock()
        if self._caps is not None and now - self._caps_at < CAPS_TTL_S:
            return dict(self._caps)
        try:
            kubectl_json("get", "-f", str(K3S_DIR), "-o", "json", timeout=3)
            caps = {"cpu_tasks": True}
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            missing = re.findall(r'([a-z][\w.-]*)\s+"([^"]+)"\s+not found', msg)
            if missing:
                what = ", ".join("%s %s" % (KIND_SHOWN.get(res.split(".")[0], res.split(".")[0]), n) for res, n in missing)
                why = "%s 없음 — kubectl apply -f orchestrator/k3s/ (install_server.sh)" % what
            else:
                why = "클러스터 객체(orchestrator/k3s/)를 확인 못 함: %s" % msg[:120]
            caps = {"cpu_tasks": False, "why": why}
        self._caps, self._caps_at = caps, now
        return dict(caps)

    def nodes(self):
        """placement 스냅숏용 노드 목록 — 서비스 몫 · 네트워크는 기억(읽기 전용)에서 붙인다"""
        rows = self._read()
        out = []
        for r in rows:
            n = {k: r[k] for k in ("name", "role", "ready", "stale", "cpu_alloc_m", "mem_alloc_mb", "cpu_now_m",
                                   "mem_now_mb")}
            # 그 노드가 맡은 기기 이름 (ds.source 라벨 — cluster_up.sh 가 nodes.json 을 보고 붙인다).
            # 사람이 「pi7」보다 「robot1」로 알아보므로 배치 화면이 이것으로 이름을 짓는다 (2026-11-26)
            n["source"] = r["source_name"]
            mem = (self.mem.node(r["name"]) if self.mem is not None else None) or {}
            share = self.mem.service_share(r["name"]) if self.mem is not None else None
            if r["cpu_alloc_m"] and share:
                n["service"] = {"cpu_p95_m": (share["cpu_pct"] or 0) * r["cpu_alloc_m"] / 100.0,
                                "mem_p95_mb": share["mem_mb"] or 0, "active_measured": share["active_measured"],
                                "mode": share.get("mode")}
            elif r["role"] == "robot":
                n["service"] = {"cpu_p95_m": 0, "mem_p95_mb": 0, "active_measured": False}
            n["net"] = mem.get("net") or {}
            # 특성 · 온도 — 지금 읽은 라벨 · 애너테이션이 우선, 없는 칸은 기억(지난번 값 · 최근 최고 온도)에서
            n["caps"] = dict(mem.get("caps") or {}, **{k: v for k, v in r["caps"].items() if v is not None})
            n["thermal"] = dict(mem.get("thermal") or {},
                                **{k: v for k, v in (r["thermal"] or {}).items() if v is not None})
            out.append(n)
        return out
