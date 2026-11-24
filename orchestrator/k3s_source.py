"""k3s 노드 원천 — 자원 기반 비전 오케스트레이션 전용 (k3s 의존은 이 폴더에만, 2026-11-24).

core(ds_resources)는 이 파일을 import 하지 않는다. 자원 기록기가 경로 문자열
`orchestrator.k3s_source:K3sNodeSource` 로 불러와 자기 스레드 안에서 돌린다 — 그래서 기억 파일의 Writer 는
기록기 하나다(단일 Writer). 컨트롤러는 같은 클래스를 cluster 백엔드(check · nodes)로 쓴다.

읽는 것 (kubectl JSON, 호출마다 시간 제한)
  Node            ready · role(ds.role) · taint · 용량 · 할당 가능량 · 신원(bootID · 커널 · OS) · 테일넷 주소
  metrics API     CPU(n코어) · 메모리 — 측정 시각(timestamp)이 STALE_S 보다 오래되면 stale
GPU 는 여기서 읽지 않는다(metrics-server 는 GPU 를 모른다) — data_stream 기기의 nvidia-smi 가 따로 잰다.
data_stream 기기 자신의 노드(테일넷 주소가 같은 노드)는 건너뛴다 — 그 기록은 로컬 기록기가 쓴다.

네트워크 (노드를 처음 볼 때 · 신원이 바뀔 때, 별도 스레드 — 기록은 기록기 tick 이 쓴다)
  tailscale ping   지연(ms) · 직접 연결인지 중계(DERP)인지
  대역폭           그 노드에 짧은 파드를 띄워 서버의 HTTP 로 PROBE_BYTES 를 받아 가게 하고, **서버 쪽에서** 보내기
                   시작부터 끝까지 잰다(서버 → 노드 방향 — 프레임이 가는 방향). busybox date 는 %N 을 몰라
                   파드 쪽 시계로는 못 잰다(2026-11-24).
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

KUBECTL_TIMEOUT = 5.0
STALE_S = 180.0
PROBE_BYTES = 8 * 1024 * 1024
PROBE_IMAGE = "busybox:1.36"
NAMESPACE = "ds-system"


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
            return None
        t0 = time.time()
        while time.time() - t0 < timeout:
            ph = subprocess.run([exe, "-n", NAMESPACE, "get", "pod", name, "-o", "jsonpath={.status.phase}"],
                                capture_output=True, text=True, timeout=10, env=_env()).stdout
            if ph in ("Succeeded", "Failed"):
                break
            time.sleep(2)
        secs = timings.get("s")
        return round(PROBE_BYTES * 8 / 1e6 / secs, 1) if secs else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    finally:
        subprocess.run([exe, "-n", NAMESPACE, "delete", "pod", name, "--wait=false"], capture_output=True,
                       timeout=10, env=_env())
        srv.shutdown()
        srv.server_close()


class K3sNodeSource:
    """기록기 플러그인 (samples · net_updates) + 컨트롤러 cluster 백엔드 (check · nodes)."""

    def __init__(self, memory=None, log=print, probe_net=True, clock=time.time):
        self.mem, self.log, self.probe_net, self.clock = memory, log, probe_net, clock
        self.local_ip = local_ts_ip()
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
            if ip and ip == self.local_ip:
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
            role = n["metadata"].get("labels", {}).get("ds.role", "")
            out.append({"name": name, "ip": ip, "role": role, "ready": ready, "stale": stale,
                        "identity": {"boot_id": info.get("bootID"), "kernel": info.get("kernelVersion"),
                                     "os": info.get("osImage")},
                        "baseline": {"arch": info.get("architecture"), "cpu_count": int(cpu_m(cap.get("cpu", "0")) / 1000),
                                     "mem_total_mb": round(mem_mb(cap.get("memory", "0"))),
                                     "disk_total_gb": round(mem_mb(cap.get("ephemeral-storage", "0")) / 1024),
                                     "gpus": []},
                        "taints": [t["key"] for t in n.get("spec", {}).get("taints", [])],
                        "cpu_alloc_m": alloc_c, "mem_alloc_mb": alloc_m, "cpu_now_m": c_now, "mem_now_mb": m_now,
                        "cpu_pct": (100.0 * c_now / alloc_c) if c_now is not None and alloc_c else None,
                        "mem_pct": (100.0 * m_now / alloc_m) if m_now is not None and alloc_m else None,
                        "measured_at": mts})
        return out

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
                                    "identity": r["identity"], "baseline": r["baseline"], "ip": r["ip"]}))
        return out

    def want_net(self, node, ip):
        """네트워크 측정을 (다른 스레드에서) 시작한다. 결과는 net_updates() 로 가져간다."""
        if not self.probe_net or node in self.net_running or not ip:
            return
        self.net_running.add(node)

        def run():
            lat, relay = tailscale_ping(ip)
            bw = bandwidth_probe(node, self.local_ip) if self.local_ip else None
            with self.lock:
                self.net_pending[node] = {"latency_ms": lat, "relay": relay, "bw_mbps": bw, "measured_at": time.time()}
            self.net_running.discard(node)
        threading.Thread(target=run, name="netprobe-" + node, daemon=True).start()

    def net_updates(self):
        with self.lock:
            d, self.net_pending = self.net_pending, {}
        return d

    # ------------------------------------------------------------ 컨트롤러 cluster 백엔드
    def check(self):
        try:
            kubectl_json("get", "nodes", "-o", "json", timeout=3)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, "k3s API 응답 없음 (%s)" % str(e)[:120]

    def nodes(self):
        """placement 스냅숏용 노드 목록 — 서비스 몫 · 네트워크는 기억(읽기 전용)에서 붙인다"""
        rows = self._read()
        out = []
        for r in rows:
            n = {k: r[k] for k in ("name", "role", "ready", "stale", "cpu_alloc_m", "mem_alloc_mb", "cpu_now_m",
                                   "mem_now_mb")}
            mem = (self.mem.node(r["name"]) if self.mem is not None else None) or {}
            u = mem.get("usage") or {}
            if r["cpu_alloc_m"]:
                n["service"] = {"cpu_p95_m": (u.get("cpu_p95") or 0) * r["cpu_alloc_m"] / 100.0,
                                "mem_p95_mb": u.get("mem_p95_mb") or 0}
            n["net"] = mem.get("net") or {}
            out.append(n)
        return out
