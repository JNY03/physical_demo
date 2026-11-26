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
