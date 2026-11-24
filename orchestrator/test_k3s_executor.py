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
