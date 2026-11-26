from ds_resources import backends, cli


def test_resolve_cluster(monkeypatch):
    monkeypatch.setattr(backends, "k3s_available", lambda: False)
    assert backends.resolve_cluster("auto") == (None, None)            # k3s 없는 사용자 — 아무것도 안 고른다
    assert backends.resolve_cluster("none") == (None, None)
    monkeypatch.setattr(backends, "k3s_available", lambda: True)
    assert backends.resolve_cluster("auto") == (backends.K3S_SOURCE, backends.K3S_CPU)
    assert backends.resolve_cluster("pkg.mod:Cls")[0] == "pkg.mod:Cls"


PLAN = {"gpu_units": [{"models": ["yoloe"], "sources": ["pi13"], "node": "wsl", "need_mb": 1560, "need_basis": "estimate",
                       "fps_cap": {"pi13": 6.29}, "gpu_util": 0.849}],
        "cpu_units": [{"task": "cam360_slice", "node": "wsl", "reason": "비교 측정 전"}],
        "degraded": [{"item": "gpu:yoloe:pi13", "reason": "fps 10 → 6.29"}],
        "refused": [{"item": "input:robot1", "reason": "입력 영상 없음"}]}


def test_brief_preview_and_run():
    run = cli.brief({"plan": PLAN, "mode": "local", "basis": "saved", "ok": False, "why": "k3s 없음"})
    assert "방식: local (기억한 자원) — k3s 없음" in run and "지금 여유" not in run      # run 에는 들이기 정보가 없다
    assert "fps 상한 pi13 6.3" in run and "못 함: input:robot1" in run and "CPU cam360_slice: wsl (비교 측정 전)" in run
    p = dict(PLAN, gpu_units=[dict(PLAN["gpu_units"][0], admit={"ok": False, "why": "GPU 메모리 부족", "room_mb": 900})])
    pre = cli.brief({"plan": p, "mode": "k3s", "basis": "live", "ok": True, "why": ""})
    assert "지금 여유 900 MB: 안 됨: GPU 메모리 부족" in pre


def test_plan_json_adds_nodes_and_keeps_old_keys():
    import json
    gpu = dict(PLAN["gpu_units"][0], admit={"ok": True, "why": "", "room_mb": 4000}, limit_mode="none")
    p = {"plan": dict(PLAN, gpu_units=[gpu]), "mode": "k3s", "basis": "live", "ok": True, "why": "", "caps": {},
         "vision_off": ["a2"],
         "snapshot": {"nodes": [{"name": "wsl", "role": "gpu", "ready": True, "stale": False,
                                 "gpus": [{"mem_total_mb": 8192, "mem_used_mb": 1000}], "gpu_util_committed": 0.2},
                                {"name": "pi13", "role": "robot", "ready": True, "stale": True}], "costs": {}}}
    j = json.loads(json.dumps(cli.plan_json(p), ensure_ascii=False))    # 그대로 JSON 으로 나가야 한다
    assert j["mode"] == "k3s" and j["basis"] == "live" and j["why"] == "" and j["caps"] == {}     # 옛 칸 그대로
    assert j["gpu_units"][0]["admit"]["room_mb"] == 4000 and j["gpu_units"][0]["limit_mode"] == "none"
    assert j["refused"][0]["reason"] == "입력 영상 없음" and j["degraded"] and j["cpu_units"][0]["node"] == "wsl"
    assert j["waiting"] == [] and j["functions"] == {} and j["ok"] is True and j["vision_off"] == ["a2"]
    assert [n["name"] for n in j["nodes"]] == ["wsl", "pi13"]
    assert j["nodes"][0]["gpu"]["total_mb"] == 8192 and "gpu" not in j["nodes"][1] and j["nodes"][1]["stale"] is True
    assert "snapshot" not in j                                           # 비용 표 같은 속내는 싣지 않는다
