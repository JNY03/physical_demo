import pytest

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


# ---------------------------------------------------------------- 기기 단계 (ds-device-runner, 2026-11-26)
CMD = ["python", "/ds/vision/device_stage.py", "--function", "{function}", "--source", "{source}",
       "--model", "{slot.detector}", "--fps", "{fps}", "--imgsz", "320"]
STAGE_TASK = {"image": "{registry}/ds-device-runner:abc1234", "command": CMD, "env": {}}
ARGS = {"function": "object_detect", "stage": "device", "source": "pi13", "fps": 2.0, "cpu_m": 1600.0,
        "mem_mb": 512.0, "slots": {"detector": "yolo26n_ncnn"}, "source_node": "pi13"}


def stage_unit(**kw):
    u = dict(UNIT, unit_id="cpu-object_detect.device@pi13-ab12cd", task="object_detect.device@pi13", node="pi13",
             requests={"cpu_m": 1600, "mem_mb": 512}, limits={"cpu_m": 3200, "mem_mb": 1024},
             spec=STAGE_TASK, args=ARGS)
    u.update(kw)
    return u


def test_fill_command_puts_the_slot_pick_and_source_in():
    assert kx.fill_command(CMD, stage_unit()) == [
        "python", "/ds/vision/device_stage.py", "--function", "object_detect", "--source", "pi13",
        "--model", "yolo26n_ncnn", "--fps", "2", "--imgsz", "320"]


def test_fill_command_leaves_old_tasks_alone():
    cmd = ["sh", "-c", "echo {source} ${HOME}"]
    assert kx.fill_command(cmd, UNIT) == cmd                           # args 가 없는 옛 cpu_tasks — 그대로


def test_empty_slot_refuses_to_build_a_pod():
    u = stage_unit(args=dict(ARGS, slots={"detector": None}))
    with pytest.raises(ValueError, match="detector"):
        kx.fill_command(CMD, u)


def test_multi_slot_is_joined_with_commas():
    u = stage_unit(args=dict(ARGS, slots={"detector": ["a", "b"]}))
    assert kx.fill_command(["--model", "{slot.detector}"], u) == ["--model", "a,b"]


def test_task_of_prefers_cpu_tasks_then_the_units_spec():
    assert kx.task_of(dict(UNIT, task="probe_x"), {"probe_x": TASK}) is TASK
    assert kx.task_of(stage_unit(), {}) is STAGE_TASK
    with pytest.raises(RuntimeError):
        kx.task_of(dict(UNIT, task="nope"), {})
    with pytest.raises(RuntimeError):
        kx.task_of(stage_unit(spec={"image": None, "command": CMD}), {})


def test_registry_host_rules():
    assert kx.registry_host({"DS_REGISTRY": "10.0.0.1:5001"}) == "10.0.0.1:5001"
    assert kx.registry_host({}, tailscale_ip="100.1.2.3") == "100.1.2.3:5000"     # nodes.json registry_port
    assert kx.fill_image("busybox:1.36", {}) == "busybox:1.36"
    assert kx.fill_image("{registry}/x:1", {"DS_REGISTRY": "h:5000"}) == "h:5000/x:1"


def test_device_stage_pod_spec(monkeypatch):
    """기기 단계 파드 — 노드 고정 · 한도 · 채운 명령 · 슬롯 · HOST_IP(카메라 :8090 에 닿는 주소)"""
    monkeypatch.setenv("DS_REGISTRY", "100.125.104.23:5000")
    d = kx.deployment(stage_unit(), STAGE_TASK, "redis://100.125.104.23:6380/0")
    pod = d["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert d["metadata"]["name"] == "cpu-object-detect-device-pi13-ab12cd"
    assert pod["nodeSelector"] == {"kubernetes.io/hostname": "pi13"}
    assert c["image"] == "100.125.104.23:5000/ds-device-runner:abc1234"
    assert c["command"][c["command"].index("--model") + 1] == "yolo26n_ncnn"
    assert c["resources"] == {"requests": {"cpu": "1600m", "memory": "512Mi"},
                              "limits": {"cpu": "3200m", "memory": "1024Mi"}}
    env = {e["name"]: e.get("value", e.get("valueFrom")) for e in c["env"]}
    assert env["HOST_IP"] == {"fieldRef": {"fieldPath": "status.hostIP"}}
    assert env["DS_SLOT_DETECTOR"] == "yolo26n_ncnn" and env["DS_SOURCE"] == "pi13" and env["DS_FPS"] == "2"
    assert env["DS_CPU_M"] == "1600" and env["DS_MEM_MB"] == "512" and env["DS_SOURCE_NODE"] == "pi13"
    assert env["UNIT_ID"] == "cpu-object_detect.device@pi13-ab12cd"


def test_registry_device_stages_have_a_runnable_command():
    """레지스트리의 기기 단계 command 는 그 단계 슬롯만 가리킨다 — 없는 슬롯 자리표시가 있으면 파드가 못 뜬다"""
    from ds_resources import functions as fnc
    funcs = fnc.load(fnc._registry(None))
    for name in ("object_detect", "zeroshot_detect", "segmentation", "depth_estimate"):
        st = funcs[name]["stages"]["device"]
        assert "{registry}/ds-device-runner:" in st["image"]
        slots = {s["role"]: "m" for s in st["slots"]}
        cmd = kx.fill_command(st["command"], stage_unit(args=dict(ARGS, function=name, slots=slots)))
        assert cmd[cmd.index("--model") + 1] == "m" and cmd[cmd.index("--function") + 1] == name
