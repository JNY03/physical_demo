"""k3s CPU 실행기 — 배치표의 CPU 작업을 파이에 Deployment 로 띄운다 (k3s 의존은 이 폴더에만, 2026-11-24).

GPU 작업은 여기서 띄우지 않는다(GPU Executor 몫, 파이로 옮기지 않는다).
Deployment 하나 = CPU 작업 하나. 노드 고정(kubernetes.io/hostname), ds.role taint 견딤, requests/limits,
PriorityClass ds-vision-low. 작업은 Redis cpu_telemetry:<unit_id> 에 하트비트를 남긴다 — 그것이 running 의 근거다.
컨트롤러(core)는 이 클래스를 주입받아 쓴다(launch · status · stop) — core 가 이 파일을 import 하지 않는다.

기능 계약의 기기 단계 (2026-11-26, docker/device-runner)
  옛 cpu_tasks 는 이름으로 self.tasks 에서 찾는다. 기능 단계(`object_detect.device@pi13` 처럼)는 그 이름이 cpu_tasks 에
  없으니 컨트롤러가 단위에 `spec`(image · command · env)과 `args`(function · stage · source · fps · slots · cpu_m ·
  mem_mb)를 실어 보낸다. 그때만 command 의 자리표시를 채우고 환경변수를 더 준다 — 옛 작업의 파드 모양은 그대로다.
    이미지    {registry}/ds-device-runner:<git 해시> — {registry} 는 DS_REGISTRY 또는 이 기기 테일넷 IP:5000 (registry_host)
    자리표시  {source} {function} {stage} {fps} {unit_id} {node} {cpu_m} {mem_mb} {slot.<역할>}
              (multi 슬롯은 쉼표로 잇는다. 값이 없으면 — 고른 모델이 없는 선택 슬롯 — 띄우지 않고 오류를 낸다)
    환경변수  DS_SOURCE · DS_FUNCTION · DS_STAGE · DS_FPS · DS_CPU_M · DS_MEM_MB · DS_SLOT_<역할> ·
              DS_SOURCE_NODE (그 소스 카메라가 달린 노드 — 알 때만) ·
              HOST_IP (status.hostIP — 파드가 그 노드의 카메라 :8090 에 닿는 주소)
"""
import json
import os
import re
import shutil
import subprocess

from orchestrator.k3s_source import NAMESPACE, _env

KUBECTL_TIMEOUT = 15.0


def _k(*args, input=None, timeout=KUBECTL_TIMEOUT):
    exe = shutil.which("kubectl")
    if not exe:
        raise RuntimeError("kubectl 없음")
    p = subprocess.run([exe, *args], input=input, capture_output=True, text=True, timeout=timeout, env=_env())
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip()[:200])
    return p.stdout


def dns_name(s):
    return re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")[:50]


#: 이미지 이름의 {registry} 자리 — 파이가 당기는 사설 레지스트리 (cluster_up.sh 의 REGISTRY 와 같은 규칙:
#: DS_REGISTRY > 이 기기 테일넷 IPv4 : nodes.json registry_port(기본 5000)). 기기 주소를 저장소에 박지 않으려는 것이다
NODES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster", "nodes.json")
_registry_cache = []


def registry_host(env=None, tailscale_ip=None):
    """파이가 당길 레지스트리 host:port — 못 정하면 None"""
    env = os.environ if env is None else env
    if env.get("DS_REGISTRY"):
        return env["DS_REGISTRY"]
    probed = tailscale_ip is None
    if probed:
        if _registry_cache:
            return _registry_cache[0]
        exe = shutil.which("tailscale")
        try:
            out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5).stdout if exe else ""
        except (OSError, subprocess.SubprocessError):
            out = ""
        tailscale_ip = (out.split() or [None])[0]
    if not tailscale_ip:
        return None
    try:
        with open(NODES_JSON, encoding="utf-8") as f:
            port = int(json.load(f).get("registry_port", 5000))
    except (OSError, ValueError):
        port = 5000
    host = "%s:%d" % (tailscale_ip, port)
    if probed and not _registry_cache:                  # 테일넷 IP 는 잘 안 바뀐다 — 한 번만 묻는다
        _registry_cache.append(host)
    return host


def fill_image(image, env=None, tailscale_ip=None):
    """{registry} 를 채운다. 자리가 없으면 그대로"""
    if "{registry}" not in image:
        return image
    host = registry_host(env, tailscale_ip)
    if not host:
        raise RuntimeError("이미지 %s 의 {registry} 를 못 채움 — DS_REGISTRY=<host:port> 를 주세요" % image)
    return image.replace("{registry}", host)


_HOLE = re.compile(r"\{(slot\.[A-Za-z0-9_]+|source|function|stage|fps|unit_id|node|cpu_m|mem_mb)\}")


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def fill_command(cmd, unit):
    """command 목록의 자리표시를 단위의 args 로 채운다. args 가 없는 단위(옛 cpu_tasks)는 그대로 돌려준다.
    값이 없는 자리(고른 모델이 없는 슬롯 등)가 있으면 ValueError — 빈 인자로 파드를 띄워 헛돌지 않게"""
    args = unit.get("args")
    if not args:
        return list(cmd)
    ctx = {k: args.get(k) for k in ("source", "function", "stage", "fps", "cpu_m", "mem_mb")}
    ctx.update(unit_id=unit.get("unit_id"), node=unit.get("node"))
    for role, v in (args.get("slots") or {}).items():
        ctx["slot." + role] = v

    def one(m):
        v = ctx.get(m.group(1))
        if v is None or v == [] or v == "":
            what = ("역할 슬롯 %s 에 고른 모델이 없다" % m.group(1)[5:]) if m.group(1).startswith("slot.") \
                else ("%s 값이 없다" % m.group(1))
            raise ValueError("%s 의 명령을 못 채움 — %s" % (unit.get("unit_id"), what))
        return _fmt(v)
    return [_HOLE.sub(one, str(x)) for x in cmd]


def stage_env(unit):
    """기능 단계 단위에만 더하는 환경변수 (args 가 없으면 빈 목록)"""
    args = unit.get("args")
    if not args:
        return []
    out = [{"name": "HOST_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}}]
    for k in ("source", "function", "stage", "fps", "cpu_m", "mem_mb", "source_node"):
        if args.get(k) is not None:
            out.append({"name": "DS_" + k.upper(), "value": _fmt(args[k])})
    for role, v in sorted((args.get("slots") or {}).items()):
        if v not in (None, [], ""):
            out.append({"name": "DS_SLOT_" + re.sub(r"[^A-Z0-9_]", "_", role.upper()), "value": _fmt(v)})
    return out


def task_of(unit, tasks):
    """단위 → 작업 명세 (image · command · env). 옛 cpu_tasks 이름이 먼저, 없으면 단위에 실려 온 spec"""
    t = (tasks or {}).get(unit["task"]) or unit.get("spec")
    if not t:
        raise RuntimeError("작업 %s 의 명세가 없다 (cpu_tasks 에도, 단위의 spec 에도)" % unit["task"])
    if not t.get("image") or not t.get("command"):
        raise RuntimeError("작업 %s 에 image · command 가 없어 파이에 띄울 수 없다" % unit["task"])
    return t


def deployment(unit, task, redis_url):
    """unit: {unit_id, node, requests, limits, priority_class, request_id, execution_id}
    task: 레지스트리 cpu_tasks.<이름> — image · command(목록) · env(사전)"""
    name = dns_name(unit["unit_id"])
    labels = {"app": "ds-cpu", "ds-unit": name}
    env = [{"name": "REDIS_URL", "value": redis_url}, {"name": "UNIT_ID", "value": unit["unit_id"]},
           {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}]
    env += stage_env(unit)
    env += [{"name": k, "value": str(v)} for k, v in sorted((task.get("env") or {}).items())]
    req, lim = unit["requests"], unit["limits"] or {}
    # 한도 면제(placement/3)로 온 일은 limits.cpu_m 이 None 이다 — **CPU 상한 없이** 올린다.
    # 쓸 수 있는데도 조여 마감을 놓치지 않게 하려는 것이고, 몫은 requests + PriorityClass 가 지킨다.
    # 메모리 상한은 면제하지 않는다(넘기면 노드가 죽어 그 기기의 제어까지 같이 죽는다).
    limits = {}
    if lim.get("cpu_m") is not None:
        limits["cpu"] = "%dm" % lim["cpu_m"]
    if lim.get("mem_mb") is not None:
        limits["memory"] = "%dMi" % lim["mem_mb"]
    return {"apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": {"name": name, "namespace": NAMESPACE, "labels": labels,
                         "annotations": {"ds/request-id": str(unit.get("request_id")),
                                         "ds/execution-id": str(unit.get("execution_id"))}},
            "spec": {"replicas": 1, "selector": {"matchLabels": labels},
                     "template": {"metadata": {"labels": labels}, "spec": {
                         "nodeSelector": {"kubernetes.io/hostname": unit["node"]},
                         "tolerations": [{"key": "ds.role", "operator": "Exists", "effect": "NoSchedule"}],
                         "priorityClassName": unit.get("priority_class", "ds-vision-low"),
                         "terminationGracePeriodSeconds": 5,
                         "containers": [{"name": "task", "image": fill_image(task["image"]), "command": fill_command(task["command"], unit),
                                         "env": env,
                                         "resources": dict({"requests": {"cpu": "%dm" % req["cpu_m"],
                                                                         "memory": "%dMi" % req["mem_mb"]}},
                                                           **({"limits": limits} if limits else {}))}]}}}}


class K3sCpuExecutor:
    kind = "k3s"

    def __init__(self, tasks, redis_url):
        self.tasks, self.redis_url = tasks, redis_url

    def launch(self, unit):
        dep = deployment(unit, task_of(unit, self.tasks), self.redis_url)   # 명령을 못 채우면 클러스터를 건드리기 전에 멈춘다
        try:
            _k("get", "namespace", NAMESPACE)
        except RuntimeError:
            _k("create", "namespace", NAMESPACE)
        _k("apply", "-f", "-", input=json.dumps(dep))
        return dns_name(unit["unit_id"])

    def status(self, unit):
        """{phase, ready, node, restarts} — 파드가 없으면 phase None"""
        out = json.loads(_k("-n", NAMESPACE, "get", "pods", "-l", "ds-unit=" + dns_name(unit["unit_id"]), "-o", "json"))
        pods = out.get("items") or []
        if not pods:
            return {"phase": None, "ready": False, "node": None, "restarts": 0}
        p = pods[0]
        cs = (p["status"].get("containerStatuses") or [{}])[0]
        return {"phase": p["status"].get("phase"), "ready": bool(cs.get("ready")), "node": p["spec"].get("nodeName"),
                "restarts": int(cs.get("restartCount") or 0),
                "waiting": ((cs.get("state") or {}).get("waiting") or {}).get("reason")}

    def stop(self, unit):
        try:
            _k("-n", NAMESPACE, "delete", "deployment", dns_name(unit["unit_id"]), "--wait=false")
        except RuntimeError:
            pass
