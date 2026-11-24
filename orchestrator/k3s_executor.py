"""k3s CPU 실행기 — 배치표의 CPU 작업을 파이에 Deployment 로 띄운다 (k3s 의존은 이 폴더에만, 2026-11-24).

GPU 작업은 여기서 띄우지 않는다(GPU Executor 몫, 파이로 옮기지 않는다).
Deployment 하나 = CPU 작업 하나. 노드 고정(kubernetes.io/hostname), ds.role taint 견딤, requests/limits,
PriorityClass ds-vision-low. 작업은 Redis cpu_telemetry:<unit_id> 에 하트비트를 남긴다 — 그것이 running 의 근거다.
컨트롤러(core)는 이 클래스를 주입받아 쓴다(launch · status · stop) — core 가 이 파일을 import 하지 않는다.
"""
import json
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


def deployment(unit, task, redis_url):
    """unit: {unit_id, node, requests, limits, priority_class, request_id, execution_id}
    task: 레지스트리 cpu_tasks.<이름> — image · command(목록) · env(사전)"""
    name = dns_name(unit["unit_id"])
    labels = {"app": "ds-cpu", "ds-unit": name}
    env = [{"name": "REDIS_URL", "value": redis_url}, {"name": "UNIT_ID", "value": unit["unit_id"]},
           {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}]
    env += [{"name": k, "value": str(v)} for k, v in sorted((task.get("env") or {}).items())]
    req, lim = unit["requests"], unit["limits"]
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
                         "containers": [{"name": "task", "image": task["image"], "command": list(task["command"]),
                                         "env": env,
                                         "resources": {"requests": {"cpu": "%dm" % req["cpu_m"], "memory": "%dMi" % req["mem_mb"]},
                                                       "limits": {"cpu": "%dm" % lim["cpu_m"], "memory": "%dMi" % lim["mem_mb"]}}}]}}}}


class K3sCpuExecutor:
    kind = "k3s"

    def __init__(self, tasks, redis_url):
        self.tasks, self.redis_url = tasks, redis_url

    def launch(self, unit):
        try:
            _k("get", "namespace", NAMESPACE)
        except RuntimeError:
            _k("create", "namespace", NAMESPACE)
        _k("apply", "-f", "-", input=json.dumps(deployment(unit, self.tasks[unit["task"]], self.redis_url)))
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
