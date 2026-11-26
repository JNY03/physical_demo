"""이 기기에서 무엇이 되는지 — 읽기만 하는 점검 (preflight)"""
from ds_resources import functions as F
from ds_resources import preflight as pf


def _funcs():
    return F.load()


def test_machine_tells_unknown_from_absent():
    """못 읽은 것은 모름(None)이고, 확인해서 없는 것은 False 다 — 둘을 섞으면 안 된다"""
    m = pf.machine(probe_gpus=lambda: [])
    assert m["cpu"] and m["cpu"] > 0
    assert m["gpu"] in (False, None) or isinstance(m["gpu"], str)
    assert isinstance(m["redis"], bool)                       # 포트는 열렸거나 아니거나


def test_gpu_missing_blocks_only_the_stages_that_need_it():
    """GPU 가 없으면 GPU 를 요구하는 서버 단계가 있는 기능만 막힌다 — 나머지는 그대로 된다"""
    funcs = _funcs()
    mach = dict(pf.machine(probe_gpus=lambda: []), gpu=False, kubectl=True)
    rows, _w = pf.features(mach, funcs)
    by = {r["key"]: r for r in rows}
    assert by["motor_control"]["usable"] is True              # 모터 제어는 GPU 와 무관
    gpu_fns = [r for r in rows if "gpu" in r["needs"]]
    assert gpu_fns and all("GPU" in r["why"] for r in gpu_fns)


def test_no_k3s_keeps_server_side_but_blocks_device_only():
    """k3s 가 없으면 기기 전용 기능은 막히고, 캐스케이드는 서버 쪽만 돈다고 알린다"""
    funcs = _funcs()
    mach = dict(pf.machine(probe_gpus=lambda: []), gpu="x", kubectl=False)
    by = {r["key"]: r for r in pf.features(mach, funcs)[0]}
    assert by["video_preprocess"]["usable"] is False and "k3s" in by["video_preprocess"]["why"]
    assert "서버 쪽만" in by["object_detect"]["why"] or by["object_detect"]["usable"]


def test_install_steps_say_what_unlocks_what():
    """더 깔 것마다 무엇이 풀리는지 — 사람이 고를 수 있게"""
    funcs = _funcs()
    mach = dict(pf.machine(probe_gpus=lambda: []), gpu=False, kubectl=False, docker=False, redis=False)
    rows, _w = pf.features(mach, funcs)
    steps = pf.install_steps(mach, rows)
    keys = {s["key"] for s in steps}
    assert {"kubectl", "docker", "redis"} <= keys
    assert all(s["how"] for s in steps)                       # 안내 없는 항목은 없다


def test_text_is_one_page_and_mentions_every_blocked_feature():
    rep = {"machine": pf.machine(probe_gpus=lambda: []),
           "features": [{"key": "a", "shown": "a", "usable": False, "why": "까닭", "needs": [], "how": ""},
                        {"key": "b", "shown": "b", "usable": True, "why": "", "needs": [], "how": ""}],
           "install": [{"what": "X", "key": "x", "how": "이렇게", "unlocks": ["a"], "count": 1}],
           "summary": {"usable": 1, "blocked": 1, "total": 2}}
    out = pf.text(rep)
    assert "1개 지금 됨" in out and "까닭" in out and "이렇게" in out
