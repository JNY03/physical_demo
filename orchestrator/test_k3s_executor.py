from orchestrator import k3s_executor as kx

UNIT = {"unit_id": "cpu-probe_x-ab12", "node": "pi6", "requests": {"cpu_m": 200, "mem_mb": 64},
        "limits": {"cpu_m": 400, "mem_mb": 128}, "priority_class": "ds-vision-low", "request_id": "r", "execution_id": "e"}
TASK = {"image": "busybox:1.36", "command": ["sh", "-c", "true"], "env": {"REDIS_HOST": "h"}}


def test_deployment_pins_node_tolerates_and_limits():
    d = kx.deployment(UNIT, TASK, "redis://h:6380/0")
    pod = d["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert d["metadata"]["name"] == "cpu-probe-x-ab12"                  # DNS 이름으로
    assert pod["nodeSelector"] == {"kubernetes.io/hostname": "pi6"}
    assert pod["tolerations"][0]["key"] == "ds.role" and pod["priorityClassName"] == "ds-vision-low"
    assert c["resources"] == {"requests": {"cpu": "200m", "memory": "64Mi"}, "limits": {"cpu": "400m", "memory": "128Mi"}}
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert env["UNIT_ID"] == "cpu-probe_x-ab12" and env["REDIS_HOST"] == "h" and env["REDIS_URL"] == "redis://h:6380/0"


def test_waived_unit_gets_no_cpu_limit_but_keeps_the_memory_limit():
    """한도 면제(placement/3)로 온 일은 CPU 상한 없이 올린다 — 쓸 수 있는데도 조여 마감을 놓치지 않게.
    메모리 상한은 그대로다(넘기면 그 노드가 죽어 기기의 제어까지 같이 죽는다)."""
    unit = dict(UNIT, limits={"cpu_m": None, "mem_mb": 128}, priority_class="ds-safety-high")
    c = kx.deployment(unit, TASK, "redis://h:6380/0")["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"] == {"requests": {"cpu": "200m", "memory": "64Mi"}, "limits": {"memory": "128Mi"}}
    # 상한이 아예 없으면 limits 칸을 넣지 않는다 (k8s 가 BestEffort 로 본다)
    none = dict(UNIT, limits=None)
    c2 = kx.deployment(none, TASK, "redis://h:6380/0")["spec"]["template"]["spec"]["containers"][0]
    assert c2["resources"] == {"requests": {"cpu": "200m", "memory": "64Mi"}}
