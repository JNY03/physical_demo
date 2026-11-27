"""기기마다 다른 대상 (2026-11-27) — 한 번 찾고 그 기기의 대상만 남긴다 (vision_infer._detect_one)."""
import numpy as np

import vision_infer as vi


class FakeTargeted:
    """allow 를 받지만 거르지 않는다 — _detect_one 의 마지막 거름(안전장치)을 본다."""
    def __init__(self):
        self.allow = []

    def __call__(self, bgr, roi_crop=None, allow=None):
        self.allow.append(allow)
        return {}, [{"name": "person", "conf": 0.9, "xyxy": [0, 0, 10, 10]},
                    {"name": "chair", "conf": 0.8, "xyxy": [20, 20, 40, 40]}]


def _run(cfg, device):
    cfg.setdefault("stabilize", {"enabled": False, "targeted": {"enabled": False}})
    out = vi._detect_one("targeted", np.zeros((64, 64, 3), np.uint8), {"run": "t"}, {}, device,
                         {"targeted": FakeTargeted()}, cfg, {}, None, True)
    return out


def _names(out):
    for x in out if isinstance(out, (list, tuple)) else [out]:
        if isinstance(x, list) and x and isinstance(x[0], dict) and "name" in x[0]:
            return sorted(d["name"] for d in x)
        if isinstance(x, (list, tuple)):
            n = _names(x)
            if n is not None:
                return n
    return None


def test_device_targets_filter_only_that_device():
    cfg = {"models": {}, "_targets_dev": {"pi12": ["chair"]}}
    assert _names(_run(dict(cfg), "pi12")) == ["chair"]
    assert _names(_run(dict(cfg), "robot1")) == ["chair", "person"]      # 적지 않은 기기는 전부


def test_scan_id_only_on_scan_device():
    """스캔 판 번호(profile.scan)는 판 기기의 targeted 결과에만 (2026-11-27, 스캔판대상_장비별_요청_VZ_261126)."""
    cfg = {"models": {}, "_profile_version": 7, "_scan": "S1", "_scan_devices": ["robot1"],
           "_scan_mark": {"robot1": 2}}
    on = _run(dict(cfg), "robot1")[0][1]["profile"]
    off = _run(dict(cfg), "pi12")[0][1]["profile"]
    assert on["scan"] == "S1" and on["mark"] == 2
    assert "scan" not in off and "mark" not in off and off["version"] == 7


def test_detect_one_passes_device_targets_to_runner():
    cfg = {"models": {}, "_targets_dev": {"pi12": ["chair"]},
           "stabilize": {"enabled": False, "targeted": {"enabled": False}}}
    fake = FakeTargeted()
    for dv in ("pi12", "robot1"):
        vi._detect_one("targeted", np.zeros((64, 64, 3), np.uint8), {"run": "t"}, {}, dv,
                       {"targeted": fake}, cfg, {}, None, True)
    assert fake.allow == [["chair"], None]


# ---------------------------------------------------------------- 실행기 안에서 먼저 거르기 (2026-11-27)
def _runner():
    """가짜 arm 을 단 진짜 TargetedRunner.__call__ — 텍스트(person · chair) · 학습(demo_box) · OVD(chair)."""
    calls = {"text": 0, "trained": 0, "ovd": 0, "depth": [], "evidence": []}

    def text(sub):
        calls["text"] += 1
        return [{"cls_name": "person", "prompt": "person", "conf": 0.9, "xyxy": [0, 0, 10, 10]},
                {"cls_name": "chair", "prompt": "chair", "conf": 0.8, "xyxy": [20, 20, 40, 40]}]

    def trained(bgr):
        calls["trained"] += 1
        return ([{"cls_name": "demo_box", "conf": 0.7, "xyxy": [44, 44, 60, 60]}],
                [{"xyxy": [1, 1, 2, 2]}], [], 3.0)

    class Ovd:
        name = arm = "grounding_dino"

        def __call__(self, sub):
            calls["ovd"] += 1
            return [{"cls_name": "chair", "xyxy": [20, 20, 40, 40], "conf": 0.5, "prompt": "chair"}]

    class Depth:
        def __call__(self, bgr, dets):
            calls["depth"].append(sorted(d["name"] for d in dets))
            return 0.0

    def evidence(bgr, dets, ovd):
        calls["evidence"].append(sorted(d["name"] for d in dets))
        for d in dets:
            d["assess"] = {"score": 50.0, "ovd": len(ovd.get(d["name"]) or [])}
        return 0.0

    tr = vi.TargetedRunner.__new__(vi.TargetedRunner)
    tr.arms, tr.trained, tr.ovd = [("yoloe", text)], [("demo_box", trained)], [Ovd()]
    tr.ovd_names, tr.ovd_rules = {"chair"}, {}
    tr.targets = [{"name": n, "cues": {}} for n in ("person", "chair", "demo_box")]
    tr.names = [t["name"] for t in tr.targets]
    tr.alias, tr.same_as = {}, {}
    tr.conf, tr.max_det, tr.area_cap, tr.merge_iou = 0.15, 100, 0.9, 0.5
    tr.distance, tr.depth, tr.evidence = True, Depth(), evidence
    tr.resolve_same_as = tr.suppress_duplicates = lambda d: d
    tr.arm_names, tr.missing = ["yoloe", "demo_box", "grounding_dino"], []
    return tr, calls


def _call(allow):
    tr, calls = _runner()
    info, dets = tr(np.zeros((64, 64, 3), np.uint8), allow=allow)
    return info, {d["name"]: d for d in dets}, calls


def test_runner_all_targets_when_no_allow():
    info, dets, calls = _call(None)
    assert sorted(dets) == ["chair", "demo_box", "person"]
    assert (calls["text"], calls["trained"], calls["ovd"]) == (1, 1, 1)
    assert calls["depth"] == [["chair", "demo_box", "person"]] and info["rejected_candidates"]


def test_runner_skips_trained_arm_and_filters_before_depth():
    info, dets, calls = _call(["chair"])
    assert sorted(dets) == ["chair"]
    assert calls["trained"] == 0 and calls["ovd"] == 1             # demo_box 검출기는 돌지 않는다
    assert calls["depth"] == [["chair"]] and calls["evidence"] == [["chair"]]
    assert info["rejected_candidates"] == [] and "demo_box" not in info["arm_ms"]


def test_runner_skips_ovd_when_no_owner_is_wanted():
    _info, dets, calls = _call(["person"])
    assert sorted(dets) == ["person"] and calls["ovd"] == 0 and calls["trained"] == 0


def test_runner_empty_allow_runs_nothing_but_keeps_record_shape():
    info, dets, calls = _call([])
    assert dets == {} and (calls["text"], calls["trained"], calls["ovd"]) == (0, 0, 0)
    assert set(info) == set(_call(None)[0])                        # 결과 칸 모양은 같다


def test_runner_wanted_candidates_equal_full_run():
    """받는 대상의 후보 · 근거는 전부를 찾을 때와 같다 (탐지 성능 그대로)."""
    full = _call(None)[1]
    for allow in (["chair"], ["person"], ["demo_box"], ["chair", "demo_box"]):
        part = _call(allow)[1]
        assert part == {n: full[n] for n in allow}


def test_runner_keeps_same_as_partner_for_resolution():
    tr, calls = _runner()
    tr.same_as = {"demo_box": "chair"}
    _info, dets = tr(np.zeros((64, 64, 3), np.uint8), allow=["chair"])
    assert calls["trained"] == 1 and sorted(d["name"] for d in dets) == ["chair", "demo_box"]   # 기기 거름은 _detect_one
