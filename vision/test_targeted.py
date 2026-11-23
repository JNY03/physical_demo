"""지정 클래스 탐지 — 카탈로그 · 프로필 검증 · 판정 역할 (2026-11-23).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python -m pytest test_targeted.py -q

모델을 적재하지 않는다(CLIP 이 필요한 근거는 빼고 잰다) — 빠르게 도는 시험이다.
"""
import copy

import numpy as np
import pytest

import cue_catalog as cc
import target_assess as ta

CFG = {"models": {"targeted": {}}}


def _resolve(profile, models=("targeted",)):
    return cc.resolve_profile(profile, CFG, list(models))


# ---------------------------------------------------------------- 카탈로그 · 프로필
def test_default_targets_when_profile_empty():
    r = _resolve({})
    assert [t["name"] for t in r["targeted"]["targets"]] == cc.DEFAULT_TARGETS
    assert r["targeted"]["distance"] is True
    assert r["overlay"] == {}


def test_empty_profile_does_not_touch_other_models():
    r = _resolve({}, ("yoloe", "river"))
    assert r["targeted"] is None and r["overlay"] == {}


def test_targets_rejected_for_other_models():
    with pytest.raises(cc.ProfileError, match="지정 클래스"):
        _resolve({"models": {"river": {"targets": [{"name": "door"}]}}}, ("river",))


def test_default_cues_follow_target_file():
    db = cc.load_targets()
    r = _resolve({"models": {"targeted": {"targets": [{"name": "demo_box"}]}}})
    assert r["targeted"]["targets"][0]["cues"] == db["demo_box"]["cues"]


def test_default_with_override():
    r = _resolve({"models": {"targeted": {"targets": [
        {"name": "door", "cues": {"default": True, "aspect": "off", "semantic": "off"}}]}}})
    cues = r["targeted"]["targets"][0]["cues"]
    assert "aspect" not in cues and "semantic" not in cues and cues["class"] == "score"


def test_unknown_cue_rejected():
    with pytest.raises(cc.ProfileError, match="모르는 근거"):
        _resolve({"models": {"targeted": {"targets": [{"name": "door", "cues": ["class", "telepathy"]}]}}})


def test_bad_role_rejected():
    with pytest.raises(cc.ProfileError, match="역할"):
        _resolve({"models": {"targeted": {"targets": [{"name": "door", "cues": {"class": "veto"}}]}}})


def test_explicit_unavailable_cue_rejected_but_default_drops_silently():
    # 사람에는 색 선언이 없다 — 직접 고르면 거절, 기본값이면 아예 없다
    with pytest.raises(cc.ProfileError, match="쓸 수 없음"):
        _resolve({"models": {"targeted": {"targets": [{"name": "person", "cues": ["class", "colour"]}]}}})
    r = _resolve({"models": {"targeted": {"targets": [{"name": "person"}]}}})
    assert "colour" not in r["targeted"]["targets"][0]["cues"]


def test_distance_off_drops_size_veto():
    r = _resolve({"models": {"targeted": {"targets": [{"name": "chair"}], "distance": False}}})
    t = r["targeted"]["targets"][0]
    assert "size_veto" not in t["cues"] and t["distance"] is False


def test_direct_target_zero_shot():
    r = _resolve({"models": {"targeted": {"targets": [
        {"name": "traffic cone", "size_mm": [300, 300, 700], "colours": ["orange"]}]}}})
    t = r["targeted"]["targets"][0]
    assert t["direct"] and t["provisional"] and t["min_score"] == cc.DIRECT_MIN_SCORE
    for k in ("class", "semantic", "shape", "colour"):
        assert k in t["cues"]
    r2 = _resolve({"models": {"targeted": {"targets": [{"name": "traffic cone"}]}}})
    assert "shape" not in r2["targeted"]["targets"][0]["cues"]
    assert "colour" not in r2["targeted"]["targets"][0]["cues"]


def test_parse_target_arg():
    assert cc.parse_target_arg("door") == {"name": "door"}
    t = cc.parse_target_arg("traffic cone:size=300x300x700:colour=orange+white")
    assert t == {"name": "traffic cone", "size_mm": [300.0, 300.0, 700.0], "colours": ["orange", "white"]}
    with pytest.raises(cc.ProfileError):
        cc.parse_target_arg("cone:size=30x30")
    with pytest.raises(cc.ProfileError):
        cc.parse_target_arg("cone:weight=3")
    assert cc.parse_cues_arg("default") == "default"
    assert cc.parse_cues_arg("default,aspect=off,colour") == {"default": True, "aspect": "off", "colour": True}


def test_unknown_colour_rejected():
    with pytest.raises(cc.ProfileError, match="색"):
        _resolve({"models": {"targeted": {"targets": [{"name": "cone", "colours": ["mauve"]}]}}})


def test_duplicate_target_rejected():
    with pytest.raises(cc.ProfileError, match="두 번"):
        _resolve({"models": {"targeted": {"targets": [{"name": "door"}, {"name": "door"}]}}})


def test_overlay_options_by_model():
    o = cc.overlay_options(["yoloe", "river", "targeted", "unidepth"], targeted_distance=False)
    assert o["yoloe"]["cards"] is not True and o["river"]["cards"] is True
    assert o["targeted"]["dist"] is not True                # 거리 측정 꺼짐
    assert o["yoloe"]["dist"] is True                       # 깊이 모델을 함께 골랐다
    assert o["unidepth"]["pose"] is True and o["yoloe"]["pose"] is not True
    assert o["targeted"]["show_rejected"] is not True       # demo_box 를 안 골랐다


def test_overlay_validation():
    r = _resolve({"models": {"targeted": {"overlay": {"cards": "hover", "count": True, "label": [["name", "dist"]]}},
                             "yoloe": {"overlay": {"count": True}}}}, ("yoloe", "targeted"))
    assert r["overlay"]["targeted"]["cards"] == "hover" and r["overlay"]["yoloe"]["count"] is True
    with pytest.raises(cc.ProfileError, match="cards"):
        _resolve({"models": {"yoloe": {"overlay": {"cards": "hover"}}}}, ("yoloe",))
    with pytest.raises(cc.ProfileError, match="거리 측정"):
        _resolve({"models": {"targeted": {"distance": False, "overlay": {"label": [["name", "dist"]]}}}})
    with pytest.raises(cc.ProfileError, match="cards"):
        _resolve({"models": {"targeted": {"overlay": {"cards": "sometimes"}}}})


def test_catalog_lists_only_available_cues():
    cat = cc.catalog(CFG)
    names = [t["name"] for t in cat["targets"]]
    assert names[:6] == cc.DEFAULT_TARGETS
    for t in cat["targets"]:
        keys = [c["key"] for c in t["cues"]]
        assert "class" in keys
        if t["name"] == "person":
            assert "colour" not in keys and "hue" not in keys
    assert {m["key"] for m in cat["models"]} == set(cc.MODELS)


# ---------------------------------------------------------------- 판정 역할 (CLIP 없이)
def _target(cues, **kw):
    t = {"name": "thing", "shown": "thing", "detector": "yoloe_text", "cues": cues,
         "weights": {"class": 0.5, "colour": 0.5}, "min_score": 50, "colours": ["red"]}
    t.update(kw)
    return t


def _red(h=60, w=40):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (0, 0, 220)                              # BGR 빨강
    return img


def _run(t, conf=0.9, img=None, box=None, ovd=None, **det):
    img = _red() if img is None else img
    box = box or [0, 0, img.shape[1], img.shape[0]]
    ev = ta.TargetEvidence([t], log=None)
    d = dict({"name": t["name"], "conf": conf, "xyxy": box}, **det)
    ev(img, [d], ovd or {})
    return d["assess"]


def test_score_role_weighted_mean():
    a = _run(_target({"class": "score", "colour": "score"}))
    assert a["suspect"] == "thing"
    assert abs(a["score"] - (0.5 * 90 + 0.5 * 100)) < 1e-6


def test_gate_failure_blocks_suspect_but_keeps_score():
    t = _target({"class": "score", "aspect": "gate"}, aspect={"h_over_w": [3.0, None]})
    a = _run(t)                                       # h/w 1.5 < 3
    assert a["suspect"] is None and a["gates_failed"] == ["aspect"] and a["score"] is not None


def test_unmeasured_gate_does_not_pass():
    t = _target({"class": "score", "saturation": "gate"}, saturation={"min": 10})
    tiny = _red(8, 8)                                 # crop 이 너무 작아 채도를 못 잰다
    a = _run(t, img=tiny)
    assert a["suspect"] is None and "saturation" in a["gates_failed"]


def test_veto_zeroes_score():
    t = _target({"class": "score", "size_veto": "veto"}, size_mm=[100, 100, 100])
    a = _run(t, real_mm={"long": 2000.0, "short": 1500.0})
    assert a["score"] == 0.0 and a["suspect"] is None and a["vetoed"]


def test_detector_cue_merges_text_free_similar_and_ovd():
    """탐지기 근거 = 텍스트 · prompt-free · 비슷한 이름(0.7 배) · OVD 의 noisy-OR (2026-11-23)."""
    t = _target({"class": "score"}, weights={"class": 1.0})
    by_arm = {"yoloe": {"chair": (0.20, False)}, "yoloe_free": {"electric chair": (0.56, True)}}
    ovd = {"thing": [{"xyxy": [0, 0, 40, 60], "conf": 0.45, "arm": "yolo_world"}]}
    a = _run(t, conf=0.20, ovd=ovd, _by_arm=by_arm)
    want = 100 * (1 - (1 - 0.20) * (1 - 0.56 * ta.SIMILAR_WEIGHT) * (1 - 0.45))
    assert abs(a["score"] - want) < 0.1
    row = a["cards"]["thing"]["rows"][0]
    assert row[0] == "class" and "~electric chair" in row[2] and "yolo_world" in row[2]
    # 정확한 이름이 있으면 같은 arm 의 비슷한 이름은 세지 않는다
    b = _run(t, conf=0.20, _by_arm={"yoloe_free": {"chair": (0.30, False), "electric chair": (0.9, True)}})
    assert abs(b["score"] - 30.0) < 0.1


def test_pair_iou_gate():
    t = _target({"class": "score", "ovd_pair_iou": "gate"}, weights={"class": 1.0}, pair_iou_min=0.42)
    box = [0, 0, 40, 60]
    near = {"thing": [{"xyxy": [1, 1, 40, 60], "conf": 0.8, "arm": "yolo_world"}]}
    assert _run(t, box=box, ovd=near)["suspect"] == "thing"
    inside_small = {"thing": [{"xyxy": [5, 5, 15, 15], "conf": 0.9, "arm": "grounding_dino"}]}
    b = _run(t, box=box, ovd=inside_small)            # 포함으로는 동의하지만 짝 IoU 는 못 넘는다(main 2026-09-15)
    assert b["suspect"] is None and "ovd_pair_iou" in b["gates_failed"]


def test_min_evidence():
    t = _target({"class": "score", "semantic": "score"}, weights={"class": 0.2, "semantic": 0.8})
    a = _run(t, conf=0.99)                            # CLIP 이 없어 글 설명을 못 잼 → 잰 무게 0.2 < 0.45
    assert a["suspect"] is None and a["enough"] is False


def test_demo_box_gate_rows_pass_when_deferred():
    t = _target({"class": "score", "ground": "gate", "aspect": "gate"}, detector="trained",
                weights={"class": 1.0}, min_score=25)
    gates = {"ground_gate": {"applicable": False, "passed": True, "words": "판정 보류"},
             "shape_gate": {"passed": True, "words": "박스 모양"}}
    a = _run(t, conf=0.6, box_gates=gates)
    assert a["suspect"] == "thing"


def test_same_as_keeps_higher():
    a = {"name": "door", "xyxy": [0, 0, 10, 20], "assess": {"suspect": None, "score": 40}}
    b = {"name": "demo_door", "xyxy": [0, 0, 10, 20], "assess": {"suspect": "demo_door", "score": 60},
         "_same_as": "door"}
    kept = ta.resolve_same_as([a, b])
    assert kept == [b]


def test_colour_fraction():
    frac, per = ta.colour_fraction(_red(), ["red"])
    assert frac > 0.99 and per["red"] > 0.99
    frac, _ = ta.colour_fraction(_red(), ["green"])
    assert frac < 0.01


# ---------------------------------------------------------------- 축 단위 판정 (2026-11-23)
def _axdet(local, glob_=True, geom=None, declared=True, tid=1, box=(0, 0, 40, 80)):
    return {"name": "person", "track_id": tid, "xyxy": list(box),
            "assess": {"suspect": "person" if glob_ else None, "scores": {"person": 60.0},
                       "cards": {"person": {"rows": []}},
                       "axes": {"global": glob_, "local": local, "local_declared": declared, "geometry": geom}}}


def test_axes_local_majority_rejects_lookalike():
    import vision_infer as vi
    ad = vi.AxisDecider()
    for _ in range(10):                               # 외투 건 의자: 전역은 사람, 부분은 거의 없다
        d = ad.update([_axdet(0.0)])[0]
    assert d["assess"]["suspect"] is None and d["assess"]["axes"]["local_ratio"] == 0.0
    ad2 = vi.AxisDecider()
    for i in range(10):                               # 사람: 부분이 대부분의 장에서 잡힌다
        d = ad2.update([_axdet(0.14 if i % 4 else 0.0)])[0]
    assert d["assess"]["suspect"] == "person"
    assert any(r[0] == "axes" for r in d["assess"]["cards"]["person"]["rows"])


def test_axes_unmeasured_local_falls_back_to_global():
    import vision_infer as vi
    ad = vi.AxisDecider()
    d = ad.update([_axdet(None)])[0]                  # 부분이 너무 작아 못 잼 — 반대가 아니다
    assert d["assess"]["suspect"] == "person"
    d = vi.AxisDecider().update([_axdet(None, geom=False)])[0]
    assert d["assess"]["suspect"] is None             # 기하가 반대하면 판정하지 않는다


def test_axes_new_track_inherits_history():
    import vision_infer as vi
    ad = vi.AxisDecider()
    for _ in range(6):
        ad.update([_axdet(0.2, tid=5)])
    d = ad.update([_axdet(0.0, tid=9)])[0]            # 트래커가 번호를 다시 붙였다 — 같은 자리의 기록을 잇는다
    assert d["assess"]["suspect"] == "person" and d["assess"]["axes"]["local_frames"] == 7


def test_part_measurability_by_scale():
    t = {"name": "door", "parts": {"names": ["door handle"], "scale": 0.05}}
    probe = ta.PartsProbe.__new__(ta.PartsProbe)
    probe.of = {"door": (["door handle"], "any")}
    probe.scale = {"door": 0.05}
    probe.vocab = ["door handle"]
    d = {"name": "door", "xyxy": [0, 0, 20, 55]}      # 55 px 문의 손잡이 ≈ 2.75 px — 볼 수 없다
    probe(np.zeros((100, 100, 3), np.uint8), [d])
    assert d["_part"] is None
