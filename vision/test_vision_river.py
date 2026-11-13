"""river 옵션 — 동의 판정 · 없는 arm · 검출 모델 경계.

가짜 arm 을 끼워 넣어 가중치 없이 돈다. 실행:
  cd vision && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv/bin/python -m pytest test_vision_river.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "vision_infer_under_test", Path(__file__).with_name("vision_infer.py"))
vi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vi)

IMG = np.zeros((100, 100, 3), np.uint8)
CLASSES = ["styrofoam_box", "water_jug", "butane_canister", "beverage_can", "wooden_box_mine"]
PROMPTS = ["styrofoam box", "plastic water jug", "butane gas canister",
           "beverage can", "wooden block"]


class FakeArm:
    """설정에 적어 둔 검출을 그대로 돌려준다."""

    def __init__(self, spec, prompts, classes, device, shared):
        if spec.get("absent"):
            raise vi._ArmMissing("없다 (시험)")
        if spec.get("explodes"):
            self.boom = True
        else:
            self.boom = False
        self.dets = [dict(d) for d in spec.get("dets") or []]
        self.name = spec.get("name") or spec["kind"]
        self.unconfirmed = spec.get("unconfirmed_class")   # 프리 프롬프트 arm 처럼 가설 칸을 가진다
        self.free_vocab = bool(spec.get("free_vocab"))     # 어휘가 가중치 안에 있는 arm
        self.shared = dict(shared)

    def __call__(self, bgr):
        if self.boom:
            raise RuntimeError("arm 이 터졌다 (시험)")
        return self.dets


@pytest.fixture(autouse=True)
def fake_arms(monkeypatch):
    monkeypatch.setitem(vi.ARM_KINDS, "fake", FakeArm)
    monkeypatch.setattr(vi, "log", lambda *a, **k: None)


def cfg(arms, **over):
    base = {"classes": CLASSES, "prompts": PROMPTS, "conf": 0.2, "min_arms": 2,
            "arm_conf": 0.05, "merge_iou": 0.5, "max_area_ratio": 0.08,
            "max_det": 100, "arms": arms}
    base.update(over)
    return base


def det(cls, conf, xyxy):
    return {"cls_name": cls, "conf": conf, "xyxy": list(xyxy)}


BOX = (10.0, 10.0, 20.0, 20.0)          # 화면의 1% — 면적 상한에 걸리지 않는다


def arm(name, dets, **extra):
    return {"kind": "fake", "name": name, "dets": dets, **extra}


# ---------------------------------------------------------------- 동의 판정
def test_two_arms_agreeing_on_the_same_box_pass():
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.6, BOX)]),
                            arm("b", [det("water_jug", 0.5, BOX)])]), "cpu")
    info, dets = r(IMG)
    assert len(dets) == 1
    assert dets[0]["name"] == "water_jug"
    assert dets[0]["agree"] == 2 and dets[0]["arms"] == ["a", "b"]
    assert dets[0]["conf"] == 0.6, "합의 신뢰도는 동의한 arm 중 최대값"
    assert info["min_arms_applied"] == 2


def test_one_arm_alone_is_dropped_when_two_are_required():
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.9, BOX)]), arm("b", [])]), "cpu")
    _info, dets = r(IMG)
    assert dets == [], "한 arm 만 말한 것은 동의가 아니다"


def test_min_arms_one_is_the_plain_union():
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.9, BOX)]), arm("b", [])],
                           min_arms=1), "cpu")
    _info, dets = r(IMG)
    assert [d["agree"] for d in dets] == [1]


def test_different_classes_on_the_same_spot_are_not_merged():
    """캔과 부탄은 서로 다른 조치를 부른다 — 한 칸에 합치면 그 구분이 사라진다."""
    r = vi.RiverRunner(cfg([arm("a", [det("beverage_can", 0.6, BOX)]),
                            arm("b", [det("butane_canister", 0.6, BOX)])],
                           min_arms=1), "cpu")
    _info, dets = r(IMG)
    assert sorted(d["name"] for d in dets) == ["beverage_can", "butane_canister"]
    assert all(d["agree"] == 1 for d in dets), "다른 클래스는 서로의 동의가 아니다"


def test_boxes_below_merge_iou_stay_separate():
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.6, (10, 10, 20, 20))]),
                            arm("b", [det("water_jug", 0.6, (60, 60, 70, 70))])],
                           min_arms=1), "cpu")
    _info, dets = r(IMG)
    assert len(dets) == 2


def test_conf_floor_applies_after_merging():
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.10, BOX)]),
                            arm("b", [det("water_jug", 0.12, BOX)])], conf=0.2), "cpu")
    _info, dets = r(IMG)
    assert dets == [], "동의해도 conf 하한을 넘지 못하면 받지 않는다"


def test_scene_sized_boxes_are_dropped():
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.9, (0, 0, 100, 100))]),
                            arm("b", [det("water_jug", 0.9, (0, 0, 100, 100))])]), "cpu")
    _info, dets = r(IMG)
    assert dets == [], "화면 전체를 가리킨 검출은 물체가 아니다"


def test_results_are_sorted_by_agreement_then_conf():
    r = vi.RiverRunner(cfg([
        arm("a", [det("water_jug", 0.9, (10, 10, 20, 20)), det("styrofoam_box", 0.4, (60, 60, 70, 70))]),
        arm("b", [det("styrofoam_box", 0.4, (60, 60, 70, 70))]),
    ], min_arms=1), "cpu")
    _info, dets = r(IMG)
    assert [d["agree"] for d in dets] == [2, 1]


def test_max_det_caps_the_output():
    many = [det("water_jug", 0.5, (x, 10, x + 5, 15)) for x in range(0, 90, 10)]
    r = vi.RiverRunner(cfg([arm("a", many), arm("b", many)], max_det=3), "cpu")
    _info, dets = r(IMG)
    assert len(dets) == 3


# ---------------------------------------------------------------- 없는 arm · 터진 arm
def test_a_missing_arm_is_dropped_and_min_arms_is_clamped_and_recorded():
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.6, BOX)]),
                            arm("gone", [], absent=True)], min_arms=2), "cpu")
    info, dets = r(IMG)
    assert r.arm_names == ["a"]
    assert info["missing_arms"] == ["gone"]
    assert info["min_arms_declared"] == 2
    assert info["min_arms_applied"] == 1, "남은 arm 수로 줄인다"
    assert info["arms_declared"] == 2 and info["unanimous"] is True
    assert len(dets) == 1, "줄인 작동점에서는 한 arm 으로도 통과한다"


def test_dropping_an_arm_turns_three_of_four_into_unanimity_and_says_so(monkeypatch):
    """3/4 와 3/3 은 다른 작동점이다 — 3/3 은 만장일치라 선언한 것보다 엄해진다."""
    said = []
    monkeypatch.setattr(vi, "log", lambda *a: said.append(a[0] % a[1:] if len(a) > 1 else a[0]))
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.6, BOX)]),
                            arm("b", [det("water_jug", 0.6, BOX)]),
                            arm("c", [det("water_jug", 0.6, BOX)]),
                            arm("gone", [], absent=True)], min_arms=3), "cpu")
    info, dets = r(IMG)
    assert (info["arms_declared"], info["min_arms_declared"]) == (4, 3)
    assert (len(info["arms"]), info["min_arms_applied"]) == (3, 3)
    assert info["unanimous"] is True
    assert any("작동점이 바뀌" in m and "만장일치" in m for m in said), \
        "작동점이 바뀐 사실을 말해야 한다 — 조용히 지나가면 숫자가 어느 작동점의 것인지 알 수 없다"
    assert len(dets) == 1


def test_a_full_set_of_arms_is_not_reported_as_a_changed_operating_point(monkeypatch):
    said = []
    monkeypatch.setattr(vi, "log", lambda *a: said.append(a[0] % a[1:] if len(a) > 1 else a[0]))
    r = vi.RiverRunner(cfg([arm("a", []), arm("b", []), arm("c", [])], min_arms=2), "cpu")
    info, _dets = r(IMG)
    assert info["unanimous"] is False
    assert not any("작동점이 바뀌" in m for m in said)


def test_refuse_mode_will_not_stand_up_with_a_missing_arm():
    with pytest.raises(SystemExit):
        vi.RiverRunner(cfg([arm("a", []), arm("gone", [], absent=True)],
                           on_missing_arm="refuse"), "cpu")


def test_all_arms_missing_is_refused_even_in_degrade_mode():
    with pytest.raises(SystemExit):
        vi.RiverRunner(cfg([arm("x", [], absent=True), arm("y", [], absent=True)]), "cpu")


def test_one_arm_failing_at_runtime_does_not_stop_the_others():
    """선택 구성요소의 장애가 관련 없는 기능을 막지 않는다 (AI-C-05)."""
    r = vi.RiverRunner(cfg([arm("boom", [], explodes=True),
                            arm("a", [det("water_jug", 0.6, BOX)]),
                            arm("b", [det("water_jug", 0.6, BOX)])]), "cpu")
    info, dets = r(IMG)
    assert len(dets) == 1
    assert info["arm_ms"]["boom"] is None, "터진 arm 은 시간이 아니라 None 으로 남는다"


# ---------------------------------------------------------------- 설정 거절
def test_classes_and_prompts_must_line_up():
    with pytest.raises(SystemExit):
        vi.RiverRunner(cfg([arm("a", [])], prompts=PROMPTS[:2]), "cpu")


def test_arm_conf_above_conf_is_refused():
    """arm 이 먼저 버린 검출은 동의에 쓸 수 없다 — 조용히 동의 수가 깎인다."""
    with pytest.raises(SystemExit):
        vi.RiverRunner(cfg([arm("a", [])], arm_conf=0.3, conf=0.2), "cpu")


def test_empty_arms_is_refused():
    with pytest.raises(SystemExit):
        vi.RiverRunner(cfg([]), "cpu")


def test_unknown_arm_kind_is_refused():
    with pytest.raises(SystemExit):
        vi.RiverRunner(cfg([{"kind": "no-such-model"}]), "cpu")


def test_min_arms_below_one_is_refused():
    with pytest.raises(SystemExit):
        vi.RiverRunner(cfg([arm("a", [])], min_arms=0), "cpu")


# ---------------------------------------------------------------- 검출 모델 경계
def test_river_is_a_detection_model_and_yoloe_wins_the_fusion_slot():
    assert vi.DETECT_MODELS == ("yoloe", "river")
    assert "river" in vi.MODELS and "river" in vi.RUNNERS
    assert vi.active_detector(["river", "unidepth"]) == "river"
    assert vi.active_detector({"river": 1, "yoloe": 1}) == "yoloe"
    assert vi.active_detector(["unidepth"]) is None


def test_river_overlay_falls_back_to_the_yoloe_label_spec():
    assert vi.label_spec({"yoloe": {"label": [["name"]]}}, "river") == [["name"]]
    assert vi.label_spec({"river": {"label": [["name", "conf"]]}},
                         "river") == [["name", "conf"]]
    assert vi.label_spec({}, "river") == [["name", "conf"]]


def test_the_shipped_config_tunes_nothing_on_the_labelled_data():
    """**라벨 219장은 전부 test 다.** 거기서 고른 값이 설정에 있으면 그 설정은 그
    다섯 기기의 것이 되고 촬영 기기에 상관없이 쓸 수 없다. 그래서 기본 설정에는
    클래스별·문구별 임계값이 **없어야** 한다 — 전역 conf 하나다.

    (이 시험은 글자를 고정하는 것이 아니라 *선언이 서로 어긋나지 않는지* 를 본다:
    모든 임계값이 arm_conf 이상이고, 이름이 classes 안에 있고, arm 이 하나면
    min_arms·min_prompts 가 1 이어야 한다.)
    """
    import yaml
    cfg_path = Path(__file__).with_name("vision_config.yaml")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    river = data["models"]["river"]
    # arm 둘 — 이름을 대는 텍스트 arm 과 자리를 내는 프리 프롬프트 arm 이다
    assert [a["kind"] for a in river["arms"]] == ["yoloe", "yoloe_free"]
    free = river["arms"][1]
    hypothesis = free["unconfirmed_class"]
    classes = [c for c in river["classes"] if c != hypothesis]
    # **표적은 둘, 합쳐지는 캔까지 셋이다**(2026-11-13). 스티로폼·물통은 클래스에서
    # 내려갔지만 **어휘에서 사라지지 않았다** — decoys 로 남아 표를 붙잡는다.
    assert set(classes) == {"wooden_box_mine", "butane_canister", "beverage_can"}
    assert {"styrofoam box", "plastic water jug"} <= set(river["decoys"])
    assert set(river["prompts"]) == set(river["classes"])
    # **가설 칸에는 텍스트 문구가 없다**(null). 있으면 그 낱말이 어휘 전체 argmax 에서
    # 표를 가져가 진짜 물체를 지운다
    assert river["prompts"][hypothesis] is None
    assert all(river["prompts"][c] for c in classes)
    # 재질을 말하는 이름만 우리 클래스로 올린다 — 모양 이름은 가설로 남는다
    mapped = {n for names in free["vocab_map"].values() for n in names}
    assert not (mapped & {"box", "crate", "breadbox", "book", "storage box"})
    assert set(free["vocab_map"]) <= set(classes)
    assert set(river["class_conf"]) <= set(river["classes"])
    floor = river["arm_conf"]
    for name, value in list(river["class_conf"].items()) + list((river["prompt_conf"] or {}).items()):
        assert value >= floor, name
    if len(river["arms"]) == 1:
        assert river["min_arms"] == 1 and river["min_prompts"] == 1
    vi.check_overlay_cfg(data)          # overlay 설정이 river 를 받아들인다
    # 라벨 219장에서 고른 값은 설정에 없다. 지금 하한을 가진 둘은 **그 자료로 고른
    # 것이 아니다** — 가설 칸의 0.25 는 검수 대기 줄의 길이를 정하는 운영값이고,
    # 부탄의 0.10 은 웹 수집본(dev1)과 실내 판정(dev2)에서 골랐다
    # (prereg_butane_strict.md). 나머지 **넷은 비어 있어야 한다.**
    assert set(river["class_conf"]) <= {hypothesis, "butane_canister"}
    assert not ({"beverage_can", "wooden_box_mine"} & set(river["class_conf"]))
    assert not river["prompt_conf"]
    # 설정이 실제로 서는지 — 가짜 arm 으로 바꿔 끼워 세운다 (가중치 없이)
    spun = vi.RiverRunner(dict(river, arms=[
        {"kind": "fake", "name": "yoloe", "dets": []},
        {"kind": "fake", "name": "yoloe_free", "dets": [], "unconfirmed_class": hypothesis},
    ]), "cpu")
    assert spun.min_prompts == 1
    assert spun.hypothesis == {hypothesis}
    # 부탄만 자기 하한(dev 에서 고름)을 가지고 **나머지 넷은 전역 conf 그대로**다
    plain = [c for c in classes if c != "butane_canister"]
    assert {spun.threshold(c) for c in plain} == {river["conf"]}
    assert spun.threshold("butane_canister") >= river["conf"]
    # 트래킹은 켜져 있되 **거르지 않아야** 한다 — 거르면 재현율이 0.640 → 0.024 다
    river_track = data["stabilize"]["river"]
    assert river_track["enabled"] is True
    assert river_track["on_untracked"] == "keep"


# ---------------------------------------------------------------- 어휘: 여러 문구 → 한 이름
VOCAB = {"styrofoam_box": ["styrofoam box", "white foam box"],
         "water_jug": ["plastic water jug", "water container"],
         "butane_canister": ["butane gas canister"],
         "beverage_can": ["beverage can"],
         "wooden_box_mine": ["wooden block"]}


def pdet(cls, conf, xyxy, prompt):
    return {"cls_name": cls, "conf": conf, "xyxy": list(xyxy), "prompt": prompt}


def test_many_prompts_expand_to_one_vocabulary_but_keep_class_order():
    prompts, owners = vi._expand_vocab(CLASSES, VOCAB)
    assert prompts == ["styrofoam box", "white foam box", "plastic water jug",
                       "water container", "butane gas canister", "beverage can",
                       "wooden block"]
    assert owners == ["styrofoam_box", "styrofoam_box", "water_jug", "water_jug",
                      "butane_canister", "beverage_can", "wooden_box_mine"]
    # 평면 목록(옛 모양)도 그대로 받는다
    assert vi._expand_vocab(CLASSES, PROMPTS) == (PROMPTS, CLASSES)


def test_a_synonym_hit_reports_the_class_name_not_the_prompt():
    """`white foam box` 로 잡혀도 나가는 이름은 `styrofoam_box` 다 — 상위 계약의
    클래스는 어휘를 늘려도 늘지 않는다."""
    arms = [{"kind": "fake", "name": "a",
             "dets": [pdet("styrofoam_box", 0.8, (10, 10, 20, 20), "white foam box")]}]
    _meta, dets = vi.RiverRunner(cfg(arms, prompts=VOCAB, min_arms=1), "cpu")(IMG)
    assert [d["name"] for d in dets] == ["styrofoam_box"]
    assert dets[0]["cls"] == CLASSES.index("styrofoam_box")
    assert dets[0]["prompts"] == ["white foam box"]        # 어느 문구가 말했는지는 남는다


def test_two_synonyms_on_one_object_fold_into_one_detection():
    """같은 물체를 두 문구가 가리키면 **검출 하나**다. 접지 않으면 한 물체가 둘로
    세어져 오검출이 부풀고 화면에 두 박스가 겹친다."""
    arms = [{"kind": "fake", "name": "a",
             "dets": [pdet("styrofoam_box", 0.8, (10, 10, 20, 20), "styrofoam box"),
                      pdet("styrofoam_box", 0.6, (11, 11, 21, 21), "white foam box")]}]
    _meta, dets = vi.RiverRunner(cfg(arms, prompts=VOCAB, min_arms=1), "cpu")(IMG)
    assert len(dets) == 1
    assert dets[0]["conf"] == 0.8                          # 센 쪽이 남는다
    assert dets[0]["prompts"] == ["styrofoam box", "white foam box"]


def test_a_prompt_shared_by_two_classes_is_refused():
    bad = dict(VOCAB, water_jug=["plastic water jug", "styrofoam box"])
    with pytest.raises(SystemExit) as e:
        vi._expand_vocab(CLASSES, bad)
    assert "두 클래스가" in str(e.value)


def test_prompts_for_an_unknown_class_or_an_empty_class_are_refused():
    with pytest.raises(SystemExit) as e:
        vi._expand_vocab(CLASSES, dict(VOCAB, driftwood=["driftwood"]))
    assert "classes 에 없는" in str(e.value)
    with pytest.raises(SystemExit) as e:
        vi._expand_vocab(CLASSES, dict(VOCAB, beverage_can=[]))
    assert "문구가 없는" in str(e.value)


def test_a_flat_prompt_list_of_the_wrong_length_says_how_to_use_a_mapping():
    with pytest.raises(SystemExit) as e:
        vi._expand_vocab(CLASSES, PROMPTS[:3])
    assert "prompts 를 클래스" in str(e.value)


# ---------------------------------------------------------------- 임계값
def test_class_conf_raises_the_bar_for_one_class_only():
    arms = [{"kind": "fake", "name": "a",
             "dets": [det("butane_canister", 0.12, (10, 10, 20, 20)),
                      det("water_jug", 0.12, (40, 40, 50, 50))]}]
    runner = vi.RiverRunner(cfg(arms, conf=0.05, min_arms=1,
                                class_conf={"butane_canister": 0.2}), "cpu")
    assert runner.threshold("butane_canister") == 0.2
    assert runner.threshold("water_jug") == 0.05
    _meta, dets = runner(IMG)
    assert [d["name"] for d in dets] == ["water_jug"]       # 부탄은 0.2 를 못 넘었다


def test_class_conf_can_lower_the_bar_for_a_hazard_class():
    """위험물은 재현율을 위해 **내린다**. 전역을 내리면 다른 클래스의 오검출까지 늘어난다."""
    arms = [{"kind": "fake", "name": "a",
             "dets": [det("butane_canister", 0.12, (10, 10, 20, 20)),
                      det("water_jug", 0.12, (40, 40, 50, 50))]}]
    _meta, dets = vi.RiverRunner(cfg(arms, conf=0.2, min_arms=1, arm_conf=0.05,
                                     class_conf={"butane_canister": 0.1}), "cpu")(IMG)
    assert [d["name"] for d in dets] == ["butane_canister"]


def test_prompt_conf_is_applied_before_merging():
    """문구별 하한은 병합 **전**이다. 병합 뒤에는 어느 문구의 conf 였는지가 최대값에
    가려져, 센 쪽이 약한 문구의 것이면 약한 문구가 조용히 통과한다."""
    arms = [{"kind": "fake", "name": "a",
             "dets": [pdet("styrofoam_box", 0.9, (10, 10, 20, 20), "white foam box"),
                      pdet("styrofoam_box", 0.3, (11, 11, 21, 21), "styrofoam box")]}]
    runner = vi.RiverRunner(cfg(arms, prompts=VOCAB, conf=0.05, min_arms=1,
                                prompt_conf={"white foam box": 0.5}), "cpu")
    meta, dets = runner(IMG)
    assert len(dets) == 1
    # 0.9 ≥ 0.5 — 센 쪽이 남고 하한 없는 문구도 함께 접힌다
    assert dets[0]["prompts"] == ["styrofoam box", "white foam box"]
    assert dets[0]["conf"] == 0.9
    assert meta["prompt_dropped"] == 0

    runner = vi.RiverRunner(cfg(arms, prompts=VOCAB, conf=0.05, min_arms=1,
                                prompt_conf={"white foam box": 0.95}), "cpu")
    meta, dets = runner(IMG)
    assert len(dets) == 1
    assert dets[0]["conf"] == 0.3                          # 센 쪽이 문구 하한에 걸려 빠졌다
    assert dets[0]["prompts"] == ["styrofoam box"]
    assert meta["prompt_dropped"] == 1


def test_a_threshold_below_arm_conf_is_refused_because_it_does_nothing():
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(cfg([{"kind": "fake", "name": "a"}], arm_conf=0.1,
                           class_conf={"water_jug": 0.02}), "cpu")
    assert "아무 일도 하지" in str(e.value)


def test_a_misspelled_threshold_name_is_refused():
    for field, bad in (("class_conf", {"water_jar": 0.3}),
                       ("prompt_conf", {"plastic water jar": 0.3})):
        with pytest.raises(SystemExit) as e:
            vi.RiverRunner(cfg([{"kind": "fake", "name": "a"}], **{field: bad}), "cpu")
        assert field in str(e.value) and "없는 이름" in str(e.value)


def test_threshold_names_are_matched_ignoring_case_and_space():
    runner = vi.RiverRunner(cfg([{"kind": "fake", "name": "a"}], prompts=VOCAB, min_arms=1,
                                class_conf={" Butane_Canister ": 0.3},
                                prompt_conf={"WHITE FOAM BOX": 0.4}), "cpu")
    assert runner.threshold("butane_canister") == 0.3
    assert runner.prompt_conf == {"white foam box": 0.4}


# ---------------------------------------------------------------- min_prompts
def test_min_prompts_above_one_is_refused_for_a_single_arm():
    """한 모델은 한 박스에 **이름 하나**를 붙인다(어휘 전체의 argmax). 그래서 문구 둘이
    같은 자리를 가리키는 일이 사실상 없다 — tune 117장에서 TP 0 · FP 6 이었다.
    세우면 아무것도 검출하지 않으므로 거절한다."""
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(cfg([{"kind": "fake", "name": "a"}], prompts=VOCAB,
                           min_arms=1, min_prompts=2), "cpu")
    assert "arm 이 하나일 때" in str(e.value) and "검출 0" in str(e.value)


def test_min_prompts_counts_distinct_prompts_across_arms():
    same = (10, 10, 20, 20)
    arms = [{"kind": "fake", "name": "a",
             "dets": [pdet("styrofoam_box", 0.8, same, "styrofoam box")]},
            {"kind": "fake", "name": "b",
             "dets": [pdet("styrofoam_box", 0.7, same, "white foam box")]}]
    _meta, dets = vi.RiverRunner(cfg(arms, prompts=VOCAB, min_arms=1, min_prompts=2), "cpu")(IMG)
    assert len(dets) == 1 and dets[0]["prompts"] == ["styrofoam box", "white foam box"]

    # 두 arm 이 **같은 문구**를 말하면 문구 동의는 1 이다 — arm 동의와 다른 축이다
    arms[1]["dets"] = [pdet("styrofoam_box", 0.7, same, "styrofoam box")]
    _meta, dets = vi.RiverRunner(cfg(arms, prompts=VOCAB, min_arms=1, min_prompts=2), "cpu")(IMG)
    assert dets == []


def test_min_prompts_below_one_is_refused():
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(cfg([{"kind": "fake", "name": "a"}], min_arms=1, min_prompts=0), "cpu")
    assert "min_prompts" in str(e.value)


def test_an_arm_that_reports_no_prompt_still_counts_as_one():
    """문구를 돌려주지 않는 arm 의 검출을 빈 집합으로 두면 min_prompts 1 이 그것을
    말없이 버린다. 클래스 이름 자체로 센다."""
    arms = [{"kind": "fake", "name": "a", "dets": [det("water_jug", 0.8, (10, 10, 20, 20))]}]
    _meta, dets = vi.RiverRunner(cfg(arms, min_arms=1), "cpu")(IMG)
    assert len(dets) == 1 and dets[0]["prompts"] == ["water_jug"]


def test_the_record_carries_the_vocabulary_and_threshold_shape():
    arms = [{"kind": "fake", "name": "a", "dets": []}]
    meta, _dets = vi.RiverRunner(cfg(arms, prompts=VOCAB, min_arms=1, conf=0.05,
                                     class_conf={"wooden_box_mine": 0.15}), "cpu")(IMG)
    assert meta["prompts_declared"] == 7 and meta["min_prompts"] == 1
    assert meta["class_conf"] == {"wooden_box_mine": 0.15} and meta["prompt_conf"] == {}
    assert meta["conf"] == 0.05 and meta["prompt"] == "text"


# ---------------------------------------------------------------- 클래스 합침
def test_merged_class_reports_the_target_name_and_keeps_the_source():
    """캔으로 잡혀도 나가는 이름은 부탄이다. **어느 이름으로 잡혔는지는 남긴다** —
    남기지 않으면 부탄이 부탄으로 잡힌 것인지 캔이 넘어온 것인지 모른다."""
    arms = [{"kind": "fake", "name": "a",
             "dets": [det("beverage_can", 0.8, (10, 10, 20, 20))]}]
    meta, dets = vi.RiverRunner(cfg(arms, conf=0.05, min_arms=1,
                                    merge_classes={"beverage_can": "butane_canister"}), "cpu")(IMG)
    assert [d["name"] for d in dets] == ["butane_canister"]
    assert dets[0]["cls"] == CLASSES.index("butane_canister")
    assert dets[0]["merged_from"] == ["beverage_can"]
    assert meta["merge_classes"] == {"beverage_can": "butane_canister"}
    assert meta["merged_dets"] == 1


def test_merging_happens_before_folding_so_one_object_becomes_one_box():
    """같은 물체에 캔과 부탄이 따로 붙은 자리(실제 `sample` 프레임이 그랬다)가
    **하나로 접혀야** 한다. 합침이 병합 뒤면 클래스가 달라 둘 다 남는다."""
    arms = [{"kind": "fake", "name": "a",
             "dets": [det("butane_canister", 0.6, (10, 10, 20, 20)),
                      det("beverage_can", 0.8, (11, 11, 21, 21))]}]
    merged = {"beverage_can": "butane_canister"}
    _m, dets = vi.RiverRunner(cfg(arms, conf=0.05, min_arms=1, merge_classes=merged), "cpu")(IMG)
    assert len(dets) == 1 and dets[0]["conf"] == 0.8
    assert dets[0]["merged_from"] == ["beverage_can"]
    # 합치지 않으면 둘 다 남는다 — 이것이 합치기 전 모양이다
    _m, two = vi.RiverRunner(cfg(arms, conf=0.05, min_arms=1), "cpu")(IMG)
    assert sorted(d["name"] for d in two) == ["beverage_can", "butane_canister"]


def test_class_conf_of_a_merged_class_uses_the_target_name():
    arms = [{"kind": "fake", "name": "a", "dets": [det("beverage_can", 0.08, (10, 10, 20, 20))]}]
    merged = {"beverage_can": "butane_canister"}
    _m, dets = vi.RiverRunner(cfg(arms, conf=0.05, min_arms=1, merge_classes=merged,
                                  class_conf={"butane_canister": 0.2}), "cpu")(IMG)
    assert dets == []                                  # 합친 뒤 이름의 하한이 걸린다


def test_a_merge_chain_or_self_merge_or_unknown_name_is_refused():
    for bad, msg in (({"beverage_can": "butane_canister", "butane_canister": "wooden_box_mine"}, "사슬"),
                     ({"beverage_can": "beverage_can"}, "자기 자신"),
                     ({"soda": "butane_canister"}, "classes 에 없는")):
        with pytest.raises(SystemExit) as e:
            vi.RiverRunner(cfg([{"kind": "fake", "name": "a"}], merge_classes=bad), "cpu")
        assert msg in str(e.value)


# ---------------------------------------------------------------- 트래킹: 고정 카메라
def tdet(name, conf, xyxy):
    return {"cls": 0, "name": name, "conf": conf, "xyxy": list(xyxy)}


def test_center_distance_links_boxes_that_do_not_overlap():
    """고정 카메라·느린 프레임에서는 물체가 자기 폭보다 멀리 움직여 **겹침이 0** 이다.
    IoU 만 보면 track 이 끊기고 매 장 새 번호가 붙는다."""
    far = [(0, 0, 10, 10), (14, 0, 24, 10)]            # IoU 0, 중심거리 14px (대각선 14.1)
    plain = vi.Stabilizer({"iou_match": 0.3, "min_hits": 1, "hold_missed": 0})
    ids = [plain.update([tdet("water_jug", 0.9, b)], i)[0]["track_id"] for i, b in enumerate(far)]
    assert ids[0] != ids[1]                            # 끊긴다

    near = vi.Stabilizer({"iou_match": 0.3, "center_dist": 1.0, "min_hits": 1, "hold_missed": 0})
    ids = [near.update([tdet("water_jug", 0.9, b)], i)[0]["track_id"] for i, b in enumerate(far)]
    assert ids[0] == ids[1]                            # 이어진다


def test_center_distance_does_not_link_across_the_gate():
    far = [(0, 0, 10, 10), (60, 0, 70, 10)]            # 중심거리 60px > 1.0 × 14.1
    stab = vi.Stabilizer({"iou_match": 0.3, "center_dist": 1.0, "min_hits": 1, "hold_missed": 0})
    ids = [stab.update([tdet("water_jug", 0.9, b)], i)[0]["track_id"] for i, b in enumerate(far)]
    assert ids[0] != ids[1]


def test_iou_pairs_win_over_merely_near_pairs():
    """겹친 쌍이 먼저 짝지어져야 한다 — 가까운 쌍이 먼저 가져가면 track 이 뒤바뀐다."""
    stab = vi.Stabilizer({"iou_match": 0.3, "center_dist": 2.0, "min_hits": 1, "hold_missed": 0})
    first = stab.update([tdet("water_jug", 0.9, (0, 0, 10, 10)),
                         tdet("water_jug", 0.9, (20, 0, 30, 10))], 0)
    left = next(d["track_id"] for d in first if d["xyxy"][0] < 5)
    second = stab.update([tdet("water_jug", 0.9, (1, 0, 11, 10)),
                          tdet("water_jug", 0.9, (21, 0, 31, 10))], 1)
    assert next(d["track_id"] for d in second if d["xyxy"][0] < 5) == left


def test_center_distance_is_off_by_default():
    assert vi.Stabilizer({}).center == 0.0


def test_held_detection_keeps_its_evidence_but_drops_the_mask():
    """못 잡은 장을 마지막 박스로 유지할 때 근거를 비우면 '근거 없이 나타난 박스' 가
    되어 읽는 쪽이 설명할 수 없다. 마스크는 이 장의 픽셀이 아니므로 버린다."""
    stab = vi.Stabilizer({"iou_match": 0.3, "min_hits": 1, "hold_missed": 2})
    d = dict(tdet("water_jug", 0.9, (0, 0, 10, 10)), arms=["yoloe"], agree=1,
             prompts=["plastic water jug"], polygon=[[0, 0]])
    stab.update([d], 0)
    out = stab.update([], 1)
    assert len(out) == 1 and out[0]["held"] is True
    assert out[0]["arms"] == ["yoloe"] and out[0]["prompts"] == ["plastic water jug"]
    assert "polygon" not in out[0]


def test_per_model_stabilize_overrides_the_base():
    cfg_all = {"stabilize": {"enabled": False, "iou_match": 0.3,
                             "river": {"enabled": True, "center_dist": 1.0}}}
    assert vi.stabilize_cfg(cfg_all, "river") == {"enabled": True, "iou_match": 0.3,
                                                 "center_dist": 1.0}
    assert vi.stabilize_cfg(cfg_all, "yoloe") == {"enabled": False, "iou_match": 0.3}
    assert vi.stabilize_cfg({}, "river") == {}


# ---------------------------------------------------------------- 기성 트래커
def _vendor(**over):
    sc = {"backend": "bytetrack", "on_untracked": "keep", "keep_boxes": "detection",
          "reset_gap_s": 2.0}
    sc.update(over)
    try:
        return vi.VendorTracker(sc)
    except vi._ArmMissing as exc:
        pytest.skip("ultralytics 트래커를 쓸 수 없다: %s" % exc)


def vdet(name, conf, xyxy, cls=0):
    return {"cls": cls, "name": name, "conf": conf, "xyxy": list(xyxy)}


def test_a_well_detected_object_keeps_the_same_track_id():
    """요구는 이것이다 — **잘 잡힌 물체는 이어져야 한다.** 기성 트래커는
    new_track_thresh(0.25) 위의 검출로만 track 을 시작한다."""
    tr = _vendor()
    ids = [tr.update([vdet("styrofoam_box", 0.9, (10 + i, 10, 30 + i, 30))], i * 0.066, None)
           [0]["track_id"] for i in range(4)]
    assert len(set(ids)) == 1 and ids[0] is not None


def test_a_weak_detection_is_kept_without_a_number():
    """`on_untracked: keep` — 문턱을 못 넘은 검출을 **버리지 않는다.** 버리면
    재현율이 0.640 → 0.024 가 되는 것을 쟀다."""
    tr = _vendor()
    out = tr.update([vdet("styrofoam_box", 0.9, (10, 10, 30, 30)),
                     vdet("water_jug", 0.06, (60, 60, 70, 70), cls=1)], 0.0, None)
    assert len(out) == 2
    by = {d["name"]: d for d in out}
    assert by["styrofoam_box"]["track_id"] is not None
    assert by["water_jug"].get("track_id") is None


def test_drop_mode_keeps_only_tracked_detections():
    tr = _vendor(on_untracked="drop")
    out = tr.update([vdet("styrofoam_box", 0.9, (10, 10, 30, 30)),
                     vdet("water_jug", 0.06, (60, 60, 70, 70), cls=1)], 0.0, None)
    assert [d["name"] for d in out] == ["styrofoam_box"]


def test_the_detector_box_is_kept_by_default():
    """칼만 박스를 쓰면 수십 px 짜리가 반 박자 늦어 IoU 0.5 를 못 넘는 것을 쟀다
    (macro 0.290 → 0.082). 기본은 탐지기 박스 그대로다."""
    tr = _vendor()
    box = (10.0, 10.0, 30.0, 30.0)
    tr.update([vdet("styrofoam_box", 0.9, box)], 0.0, None)
    out = tr.update([vdet("styrofoam_box", 0.9, box)], 0.066, None)
    assert out[0]["xyxy"] == list(box)


def test_other_fields_survive_the_tracker():
    """근거(arms·prompts)를 잃으면 번호는 붙었는데 왜 그 이름인지 설명할 수 없다."""
    tr = _vendor()
    d = dict(vdet("styrofoam_box", 0.9, (10, 10, 30, 30)), arms=["yoloe"],
             prompts=["styrofoam box"], agree=1)
    out = tr.update([d], 0.0, None)
    assert out[0]["arms"] == ["yoloe"] and out[0]["prompts"] == ["styrofoam box"]


def test_an_unknown_backend_is_refused_and_a_known_one_is_listed():
    with pytest.raises(SystemExit) as e:
        vi.VendorTracker({"backend": "sort"})
    assert "bytetrack" in str(e.value)


def test_keys_of_the_other_backend_are_tolerated_but_typos_are_not():
    """한 설정 블록에 builtin 용 값과 기성 트래커 용 값이 같이 적혀 있다 —
    backend 를 바꿔 가며 쓸 수 있어야 한다. 오타는 그래도 거절한다."""
    _vendor(iou_match=0.2, center_dist=1.0, min_hits=2)      # builtin 용 — 통과
    with pytest.raises(SystemExit) as e:
        vi.VendorTracker({"backend": "bytetrack", "track_high_thresholdd": 0.3})
    assert "모르는 값" in str(e.value)


def test_a_time_gap_restarts_the_tracker():
    """끊긴 구간 뒤에는 처음부터다 — 번호를 이어 가면 사라진 물체와 새 물체가 같은
    번호를 받는다. 번호 세는 것이 1 로 돌아가는 것으로 확인한다."""
    far = (90, 90, 110, 110)
    tr = _vendor()
    first = tr.update([vdet("styrofoam_box", 0.9, (10, 10, 30, 30))], 0.0, None)[0]["track_id"]
    # 끊김 없이 다른 자리에 새 물체 → 다음 번호. **첫 장에는 번호가 없다** —
    # 첫 장 이후에 생긴 track 은 한 장 더 확인되어야 나온다(그게 이 트래커의 문턱이다)
    cont = tr.update([vdet("styrofoam_box", 0.9, (10, 10, 30, 30)),
                      vdet("styrofoam_box", 0.9, far)], 0.066, None)
    assert [d.get("track_id") for d in cont] == [first, None]
    cont = tr.update([vdet("styrofoam_box", 0.9, (10, 10, 30, 30)),
                      vdet("styrofoam_box", 0.9, far)], 0.132, None)
    assert max(d["track_id"] for d in cont) > first

    tr2 = _vendor()
    tr2.update([vdet("styrofoam_box", 0.9, (10, 10, 30, 30))], 0.0, None)
    after = tr2.update([vdet("styrofoam_box", 0.9, far)], 99.0, None)
    assert after[0]["track_id"] == first, "끊긴 뒤에는 번호가 처음부터다"


def test_the_shipped_config_tracks_without_filtering():
    """설정이 **번호를 붙이는 층**이어야 한다 — 거르는 층이면 재현율이 무너진다."""
    import yaml
    data = yaml.safe_load(Path(__file__).with_name("vision_config.yaml").read_text(encoding="utf-8"))
    sc = vi.stabilize_cfg(data, "river")
    assert sc["enabled"] is True
    assert sc["on_untracked"] == "keep" and sc["keep_boxes"] == "detection"
    assert sc["backend"] in ("builtin",) + vi.VendorTracker.BACKENDS


# ---------------------------------------------------------------- 미끼 문구
def test_decoy_prompts_enter_the_vocabulary_but_never_leave_as_detections():
    """열린어휘 탐지기는 어휘 전체의 argmax 로 이름 하나를 붙인다. 배경에 실제로 있는
    것을 어휘에 넣어 두면 둔치의 판자가 `wooden_box_mine` 가 아니라 `wooden plank` 로 불려
    **사라진다**. 넣은 것이 어휘에 들어갔는지, 결과에서 빠지는지 둘 다 본다."""
    prompts, owners = vi._expand_vocab(CLASSES, PROMPTS, ["driftwood", "rock"])
    assert prompts[-2:] == ["driftwood", "rock"]
    assert owners[-2:] == [vi.DECOY, vi.DECOY]

    arms = [{"kind": "fake", "name": "a",
             "dets": [det("styrofoam_box", 0.8, (10, 10, 20, 20)),
                      {"cls_name": vi.DECOY, "conf": 0.9, "xyxy": [40, 40, 50, 50],
                       "prompt": "driftwood"}]}]
    meta, dets = vi.RiverRunner(cfg(arms, conf=0.05, min_arms=1,
                                    decoys=["driftwood", "rock"]), "cpu")(IMG)
    assert [d["name"] for d in dets] == ["styrofoam_box"]
    assert meta["decoy_dropped"] == 1 and meta["decoys"] == ["driftwood", "rock"]


def test_a_decoy_that_a_class_already_uses_is_refused():
    with pytest.raises(SystemExit) as e:
        vi._expand_vocab(CLASSES, PROMPTS, ["styrofoam box"])
    assert "이미 쓰는 문구" in str(e.value)


def test_no_decoys_means_nothing_changes():
    arms = [{"kind": "fake", "name": "a", "dets": [det("styrofoam_box", 0.8, (10, 10, 20, 20))]}]
    meta, dets = vi.RiverRunner(cfg(arms, conf=0.05, min_arms=1), "cpu")(IMG)
    assert meta["decoys"] == [] and meta["decoy_dropped"] == 0 and len(dets) == 1


# ---------------------------------------------------------------- 이름 깜빡임
def lsdet(name, conf, xyxy, cls=0):
    return {"cls": cls, "name": name, "conf": conf, "xyxy": list(xyxy)}


def names_over(policy, seq, **over):
    """한 자리에 머무는 물체에 seq 의 이름을 차례로 주고, 나가는 이름을 모은다."""
    ls = vi.LabelStabilizer({"label_policy": policy, "iou_match": 0.3, **over})
    out = []
    for i, nm in enumerate(seq):
        d = lsdet(nm, 0.5, (10, 10, 30, 30), cls=CLASSES.index(nm))
        out.append(ls.update([d], i * 0.066)[0]["name"])
    return out


def test_raw_policy_changes_nothing():
    seq = ["styrofoam_box", "wooden_box_mine", "styrofoam_box"]
    assert names_over("raw", seq) == seq


def test_hysteresis_needs_k_consecutive_disagreements_to_switch():
    """한 장짜리 반대는 이름을 바꾸지 못한다 — 그것이 깜빡임의 정체다."""
    seq = ["styrofoam_box", "wooden_box_mine", "styrofoam_box", "styrofoam_box"]
    assert names_over("hyst-2", seq) == ["styrofoam_box"] * 4
    # 두 장 연속이면 바뀐다
    seq = ["styrofoam_box", "wooden_box_mine", "wooden_box_mine", "wooden_box_mine"]
    assert names_over("hyst-2", seq) == ["styrofoam_box", "styrofoam_box",
                                         "wooden_box_mine", "wooden_box_mine"]


def test_the_frame_s_own_name_is_never_thrown_away():
    """이름을 붙들면 바뀌어야 할 때도 늦는다. 그래서 이 장의 원래 이름을 함께 낸다 —
    프레임별 진실이 필요한 소비자는 label_raw 를 본다."""
    ls = vi.LabelStabilizer({"label_policy": "hyst-2", "iou_match": 0.3})
    ls.update([lsdet("styrofoam_box", 0.5, (10, 10, 30, 30))], 0.0)
    out = ls.update([lsdet("wooden_box_mine", 0.5, (10, 10, 30, 30), cls=4)], 0.066)[0]
    assert out["name"] == "styrofoam_box" and out["label_raw"] == "wooden_box_mine"
    assert out["cls"] == CLASSES.index("styrofoam_box")


def test_an_unsettled_name_is_marked_and_carries_its_candidates():
    ls = vi.LabelStabilizer({"label_policy": "hyst-2", "iou_match": 0.3, "label_margin": 0.9})
    ls.update([lsdet("styrofoam_box", 0.5, (10, 10, 30, 30))], 0.0)
    out = ls.update([lsdet("wooden_box_mine", 0.5, (10, 10, 30, 30), cls=4)], 0.066)[0]
    assert out["label_stable"] is False
    assert set(out["label_candidates"]) == {"styrofoam_box", "wooden_box_mine"}


def test_a_forgotten_track_does_not_lend_its_name_to_a_later_object():
    """사라진 물체를 잊지 않으면 그 이름이 뒤에 들어온 다른 물체에 얹힌다."""
    ls = vi.LabelStabilizer({"label_policy": "hyst-3", "iou_match": 0.3, "label_max_missed": 2})
    ls.update([lsdet("styrofoam_box", 0.9, (10, 10, 30, 30))], 0.0)
    for i in range(1, 5):
        ls.update([], i * 0.066)                        # 한참 안 보인다
    out = ls.update([lsdet("wooden_box_mine", 0.9, (10, 10, 30, 30), cls=4)], 0.4)[0]
    assert out["name"] == "wooden_box_mine"


def test_an_unknown_label_policy_is_refused():
    for bad in ("hyst", "hyst-0", "smooth"):
        with pytest.raises(SystemExit):
            vi.LabelStabilizer({"label_policy": bad})


def test_label_policy_keys_are_not_rejected_by_a_vendor_tracker():
    """이름 안정화 값은 트래커의 것이 아니다. 같은 블록에 적히므로 거절하면
    설정이 아예 안 선다 — 실제로 전체 추론이 그렇게 멈췄다."""
    tr = _vendor(label_policy="hyst-2", label_margin=0.6, label_max_missed=5)
    assert tr.backend == "bytetrack"


# --- 프리 프롬프트 arm: 자리는 믿고 이름은 가설로 둔다 (2026-11-12) ----------

def test_a_free_name_maps_to_our_class_and_the_rest_stay_unconfirmed():
    """지도에 있는 이름만 우리 클래스가 된다. 나머지는 **버리지 않고** 가설로 남는다
    — 1 위를 고르지 않는 것이 요점이다(AI-S-04)."""
    m = vi._ArmYoloeFree._read_map(
        {"wooden_box_mine": ["plywood", "WOOD"], "water_jug": ["jug"]},
        {"wooden_box_mine", "water_jug", "unconfirmed"})
    assert m == {"plywood": "wooden_box_mine", "wood": "wooden_box_mine", "jug": "water_jug"}


def test_one_free_name_cannot_point_at_two_classes():
    """한 이름이 두 클래스를 가리키면 어느 쪽도 믿을 수 없다."""
    with pytest.raises(SystemExit) as e:
        vi._ArmYoloeFree._read_map({"wooden_box_mine": ["crate"], "styrofoam_box": ["crate"]},
                                   {"wooden_box_mine", "styrofoam_box"})
    assert "crate" in str(e.value)


def test_a_vocab_map_class_must_be_declared():
    with pytest.raises(SystemExit) as e:
        vi._ArmYoloeFree._read_map({"metal_drum": ["barrel"]}, {"wooden_box_mine"})
    assert "metal_drum" in str(e.value)


def test_an_empty_vocab_map_is_refused():
    """지도 없이 세우면 **모든 검출이 unconfirmed** 가 되어 아무것도 못 가린다."""
    with pytest.raises(SystemExit):
        vi._ArmYoloeFree._read_map(None, {"wooden_box_mine"})


def test_no_prompts_means_the_class_name_is_the_prompt():
    """프리 프롬프트만 쓰는 구성은 텍스트 문구가 없다. 적지 않으면 클래스 이름이
    문구다 — 빈 dict 는 그대로 거절한다(비워 둔 것과 안 적은 것은 다른 뜻이다)."""
    prompts, owners = vi._expand_vocab(["wooden_box_mine", "unconfirmed"], None)
    assert prompts == ["wooden_box_mine", "unconfirmed"]
    assert owners == ["wooden_box_mine", "unconfirmed"]
    with pytest.raises(SystemExit):
        vi._expand_vocab(["wooden_box_mine"], {})


# --- 가설 칸: 자리는 내고 이름은 고르지 않는다 (2026-11-12) ------------------

HYP_CLASSES = CLASSES + ["unconfirmed"]
HYP_PROMPTS = {c: [p] for c, p in zip(CLASSES, PROMPTS)}
HYP_PROMPTS["unconfirmed"] = None       # 텍스트 문구를 두지 않는다


def test_a_class_can_say_it_has_no_text_prompt():
    """가설 칸은 어휘에 **들어가지 않는다.** 들어가면 그 낱말이 어휘 전체 argmax 에서
    표를 가져가 진짜 물체를 지운다 — 미끼 문구가 현장에서 진 자리가 그것이다.
    빈 목록은 그대로 거절한다(비워 둔 실수와 안 두겠다는 선언은 다른 뜻이다)."""
    prompts, owners = vi._expand_vocab(HYP_CLASSES, HYP_PROMPTS)
    assert "unconfirmed" not in prompts and "unconfirmed" not in owners
    assert len(prompts) == len(CLASSES)
    with pytest.raises(SystemExit) as e:
        vi._expand_vocab(HYP_CLASSES, dict(HYP_PROMPTS, unconfirmed=[]))
    assert "unconfirmed" in str(e.value)


def test_a_hypothesis_class_cannot_also_carry_a_text_prompt():
    """문구가 있으면 텍스트 arm 도 그 이름을 낸다 — 그러면 '이름을 고르지 않았다' 가
    거짓이 되고, 어느 쪽이 말한 것인지 알 수 없는 칸이 생긴다."""
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(cfg([arm("text", []),
                            arm("free", [], unconfirmed_class="water_jug")],
                           classes=HYP_CLASSES, prompts=HYP_PROMPTS, min_arms=1), "cpu")
    assert "water_jug" in str(e.value)


def test_a_hypothesis_is_dropped_where_a_named_box_already_sits():
    """클래스가 다른 겹침은 병합되지 않으므로(캔과 부탄을 한 칸에 합치지 않기 위한
    규칙이다) 이 걸음이 없으면 이름이 붙은 물체마다 가설이 하나씩 더 붙어 **한 물체가
    둘로 세어진다.**"""
    meta, dets = vi.RiverRunner(
        cfg([arm("text", [det("water_jug", 0.6, BOX)]),
             arm("free", [det("unconfirmed", 0.9, BOX)], unconfirmed_class="unconfirmed")],
            classes=HYP_CLASSES, prompts=HYP_PROMPTS, min_arms=1), "cpu")(IMG)
    assert [d["name"] for d in dets] == ["water_jug"]
    assert meta["hypothesis"] == ["unconfirmed"] and meta["hypothesis_covered"] == 1


def test_a_hypothesis_survives_where_nothing_is_named():
    """이름을 아무도 대지 못한 자리가 **남아야** 검수·학습 후보가 생긴다 (AI-S-04, AI-L-01)."""
    meta, dets = vi.RiverRunner(
        cfg([arm("text", [det("water_jug", 0.6, BOX)]),
             arm("free", [det("unconfirmed", 0.9, (60.0, 60.0, 70.0, 70.0))],
                 unconfirmed_class="unconfirmed")],
            classes=HYP_CLASSES, prompts=HYP_PROMPTS, min_arms=1), "cpu")(IMG)
    assert sorted(d["name"] for d in dets) == ["unconfirmed", "water_jug"]
    assert meta["hypothesis_covered"] == 0


def test_an_arm_is_told_the_declared_classes_not_just_the_prompted_ones():
    """arm 이 이름을 검사할 때 보는 것은 **classes** 다. `owners`(문구를 가진 클래스)
    만 주면 문구 없는 가설 칸이 거기 없어서 arm 이 자기 설정을 거절한다 — 가짜 arm 은
    그 검사를 하지 않으므로 시험이 통과했고, **실제 가중치로 한 번 돌려서** 드러났다."""
    r = vi.RiverRunner(cfg([arm("free", [], unconfirmed_class="unconfirmed")],
                           classes=HYP_CLASSES, prompts=HYP_PROMPTS, min_arms=1), "cpu")
    told = r.arms[0][1].shared["classes"]
    assert "unconfirmed" in told and set(told) == set(HYP_CLASSES)


def test_the_free_only_profile_also_stands():
    """`river_open` 은 텍스트 arm 이 없다 — 문구 자리가 placeholder 라 가설 칸이
    거기 들어 있어도 해가 없다. 가설 검사를 구성과 무관하게 걸면 이 설정이 안 선다."""
    import yaml
    data = yaml.safe_load(Path(__file__).with_name("vision_config.yaml").read_text(encoding="utf-8"))
    conf = data["models"]["river_open"]
    assert "prompts" not in conf                     # 적지 않는다 — 프리 어휘를 쓴다
    free = conf["arms"][0]
    spun = vi.RiverRunner(dict(conf, arms=[
        {"kind": "fake", "name": "free", "dets": [], "free_vocab": True,
         "unconfirmed_class": free["unconfirmed_class"]}]), "cpu")
    assert spun.hypothesis == {free["unconfirmed_class"]}


# ------------------------------------------------- 크기별 하한 (area_bands)
# 상수 하나(max_area_ratio 0.08)는 **그 상수를 고른 현장에서만** 맞았다. 하천 GT 는
# 가장 큰 것이 화면의 0.006 이라 한 번도 켜지지 않았고, 실내 근접에서는 나무 상자
# 정답의 88% 가 그 위라 잡고도 버렸다(2026-11-13 실측). 버리는 기준을 **얼마나
# 확신해야 받는가**로 바꾼다.
BANDS = [{"upto": 0.02, "conf": 0.05}, {"upto": 0.40, "conf": 0.30}]
SMALL = (10.0, 10.0, 20.0, 20.0)        # 100 px² = 화면의 0.01
BIG = (10.0, 10.0, 60.0, 60.0)          # 2500 px² = 화면의 0.25
HUGE = (0.0, 0.0, 90.0, 90.0)           # 8100 px² = 화면의 0.81


def banded(dets, **over):
    over.setdefault("area_bands", BANDS)
    return cfg([arm("a", dets)], min_arms=1, conf=0.05, max_area_ratio=None, **over)


def test_a_big_box_needs_more_confidence_than_a_small_one():
    """같은 0.10 인데 작은 것은 받고 큰 것은 받지 않는다 — 멀리 있는 흐릿한 물체와
    코앞의 흐릿한 물체는 같은 사실이 아니다."""
    r = vi.RiverRunner(banded([det("water_jug", 0.10, SMALL),
                               det("water_jug", 0.10, BIG)]), "cpu")
    info, dets = r(IMG)
    assert [d["xyxy"] for d in dets] == [list(SMALL)]
    assert info["dropped_by_area"] == 1


def test_a_big_box_with_enough_confidence_is_kept():
    """예전에는 0.08 을 넘는다는 이유만으로 **확신과 무관하게** 버렸다."""
    r = vi.RiverRunner(banded([det("wooden_box_mine", 0.50, BIG)]), "cpu")
    _info, dets = r(IMG)
    assert len(dets) == 1 and dets[0]["name"] == "wooden_box_mine"


def test_a_box_above_the_last_band_is_a_scene_detection():
    """마지막 upto 가 상한이다 — 아무리 확신해도 화면을 통째로 가리킨 것은 받지 않는다."""
    r = vi.RiverRunner(banded([det("water_jug", 0.99, HUGE)]), "cpu")
    _info, dets = r(IMG)
    assert dets == []


def test_a_declared_class_threshold_beats_the_size_band():
    """위험물에 낮은 하한을 적어 두면 코앞의 큰 위험물도 그 하한으로 받는다.
    크기가 덮어쓰면 재현율을 위해 적은 값이 조용히 무효가 된다."""
    c = banded([det("wooden_box_mine", 0.10, BIG), det("water_jug", 0.10, BIG)],
               class_conf={"wooden_box_mine": 0.05})
    _info, dets = vi.RiverRunner(c, "cpu")(IMG)
    assert [d["name"] for d in dets] == ["wooden_box_mine"]


def test_area_bands_and_max_area_ratio_cannot_both_be_declared():
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(cfg([arm("a", [])], min_arms=1, area_bands=BANDS), "cpu")
    assert "상한의 주인이 둘" in str(e.value)


def test_area_bands_must_be_declared_in_increasing_order():
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(banded([], area_bands=[{"upto": 0.4, "conf": 0.3},
                                              {"upto": 0.02, "conf": 0.05}]), "cpu")
    assert "커지는 차례" in str(e.value)


def test_a_band_below_arm_conf_is_refused():
    """arm 이 먼저 버리는 값을 적으면 아무 일도 하지 않는다 — 조용히 통과시키지 않는다."""
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(banded([], area_bands=[{"upto": 0.4, "conf": 0.01}]), "cpu")
    assert "arm_conf" in str(e.value)


def test_without_bands_the_old_area_cap_still_cuts():
    """선언하지 않은 배치는 **지금까지와 똑같이** 돈다."""
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.9, BIG)])], min_arms=1), "cpu")
    info, dets = r(IMG)
    assert dets == [] and info["area_bands"] == [] and info["area_cap"] == 0.08


def test_the_shipped_config_does_not_retune_the_far_regime():
    """출고 설정은 크기별 하한을 쓴다. **첫 칸의 하한은 전역 conf 와 같아야 한다** —
    멀리 있는 구간(하천 GT 가 전부 여기)의 작동점을 건드리지 않았다는 뜻이고, 그래야
    이 변경이 '근접 구간을 더한 것' 으로만 설명된다. 글자가 아니라 그 관계를 고정한다."""
    import yaml
    river = yaml.safe_load(Path(__file__).with_name("vision_config.yaml")
                           .read_text(encoding="utf-8"))["models"]["river"]
    bands = river["area_bands"]
    assert "max_area_ratio" not in river, "상한의 주인이 둘이 되면 안 된다"
    assert len(bands) >= 2 and bands[0]["conf"] == river["conf"]
    assert [b["upto"] for b in bands] == sorted(b["upto"] for b in bands)
    # 하천 GT 의 가장 큰 상자(0.006)와 실내 GT 의 가장 작은 상자(0.010) 사이에 칸이 갈려야
    # 두 현장이 서로 다른 하한을 받는다. 그 사이가 아니면 한쪽이 다른 쪽 자를 쓴다.
    assert 0.006 < bands[0]["upto"] < 0.010 or bands[0]["upto"] == 0.01
    # 실내 GT 의 가장 큰 것(0.386)이 상한 안에 들어야 한다
    assert bands[-1]["upto"] > 0.386


# ------------------------------------------------- 이긴 낱말 (lead)
def test_a_detection_says_which_word_won():
    """가설 칸은 이름이 전부 `unconfirmed` 다. 어느 낱말이 그 자리를 이겼는지를
    남기지 않으면 **사람이 볼 것이 아무것도 없다** — 화면에 'unconfirmed' 만 23 개 뜬다."""
    r = vi.RiverRunner(cfg([arm("a", [det("water_jug", 0.9, BOX)], prompts=["jug"]),
                            arm("b", [det("water_jug", 0.4, BOX)], prompts=["bottle"])],
                           min_arms=1), "cpu")
    _info, dets = r(IMG)
    assert len(dets) == 1
    # 문구를 돌려주지 않는 가짜 arm 은 클래스 이름으로 센다(_said). 확신이 높은 쪽이 이긴다
    assert dets[0]["lead"] == "water_jug"
    assert dets[0]["lead"] in dets[0]["prompts"]


def test_the_winning_word_is_the_most_confident_one():
    a = {"kind": "fake", "name": "a",
         "dets": [{"cls_name": "wooden_box_mine", "conf": 0.3, "xyxy": list(BOX), "prompt": "plank"}]}
    b = {"kind": "fake", "name": "b",
         "dets": [{"cls_name": "wooden_box_mine", "conf": 0.8, "xyxy": list(BOX), "prompt": "plywood"}]}
    _info, dets = vi.RiverRunner(cfg([a, b], min_arms=1), "cpu")(IMG)
    assert dets[0]["lead"] == "plywood"
    assert dets[0]["prompts"] == ["plank", "plywood"]


# ------------------------------------------------- 클래스별 근거 요구 (class_rules)
# 부탄 오검출은 **작고 흐릿하다** — 실측 최소변 중앙 32 px(실내)·14 px(하천)에
# 89~100% 가 64 px 미만. 확신값만 올리면 큰 오검출은 남고 작은 진짜가 죽는다.
TINY = (10.0, 10.0, 14.0, 14.0)         # 최소변 4 px
MID = (10.0, 10.0, 34.0, 34.0)          # 최소변 24 px (면적 상한 0.08 안)


def test_a_class_can_require_a_minimum_size():
    c = cfg([arm("a", [det("butane_canister", 0.9, TINY), det("butane_canister", 0.9, MID)])],
            min_arms=1, class_rules={"butane_canister": {"min_short_px": 20}})
    info, dets = vi.RiverRunner(c, "cpu")(IMG)
    assert [d["xyxy"] for d in dets] == [list(MID)]
    assert info["dropped_by_class_rule"] == 1


def test_a_class_can_require_two_arms_while_others_need_one():
    """위험물만 엄하게 한다 — 나머지 클래스의 작동점은 그대로다."""
    c = cfg([arm("a", [det("butane_canister", 0.9, BOX), det("water_jug", 0.9, MID)]),
             arm("b", [det("water_jug", 0.8, MID)])],
            min_arms=1, class_rules={"butane_canister": {"min_arms": 2}})
    _info, dets = vi.RiverRunner(c, "cpu")(IMG)
    assert [d["name"] for d in dets] == ["water_jug"]


def test_a_class_rule_does_not_touch_other_classes():
    c = cfg([arm("a", [det("water_jug", 0.9, TINY)])], min_arms=1,
            class_rules={"butane_canister": {"min_short_px": 20, "min_arms": 2}})
    _info, dets = vi.RiverRunner(c, "cpu")(IMG)
    assert len(dets) == 1 and dets[0]["name"] == "water_jug"


def test_a_class_rule_cannot_own_the_confidence_floor():
    """하한의 주인은 class_conf 다. 둘이 가지면 어느 쪽이 이겼는지 설명할 수 없다."""
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(cfg([arm("a", [])], min_arms=1,
                           class_rules={"butane_canister": {"conf": 0.3}}), "cpu")
    assert "class_conf" in str(e.value)


def test_an_unknown_class_in_class_rules_is_refused():
    with pytest.raises(SystemExit) as e:
        vi.RiverRunner(cfg([arm("a", [])], min_arms=1,
                           class_rules={"nope": {"min_arms": 2}}), "cpu")
    assert "없는 클래스" in str(e.value)


def test_the_shipped_config_only_tightens_the_hazard_it_measured():
    """부탄에만 건 규칙이 다른 클래스로 새지 않았는지. 재현율을 위해 **의도적으로
    비워 둔** 나머지 넷의 하한이 조용히 생기면 앞 측정들이 전부 다른 작동점의 것이 된다."""
    import yaml
    river = yaml.safe_load(Path(__file__).with_name("vision_config.yaml")
                           .read_text(encoding="utf-8"))["models"]["river"]
    assert set(river["class_rules"]) == {"butane_canister"}
    assert set(river["class_conf"]) == {"unconfirmed", "butane_canister"}
    # 측정이 min_arms 2 를 재현율 0.029 로 떨어뜨렸다. 되살아나면 시험이 막는다
    assert not river["class_rules"]["butane_canister"].get("min_arms")
    assert river["merge_classes"] == {"beverage_can": "butane_canister"}


def test_a_detection_keeps_what_each_word_was_worth():
    """집합만 남기면 "무엇이 얼마나 그렇게 불렀는지" 가 사라진다 — 사람이 후보를
    견줄 수 없다. 확신 내림차순으로 들고 간다."""
    a = {"kind": "fake", "name": "a",
         "dets": [{"cls_name": "wooden_box_mine", "conf": 0.31, "xyxy": list(BOX), "prompt": "plank"}]}
    b = {"kind": "fake", "name": "b",
         "dets": [{"cls_name": "wooden_box_mine", "conf": 0.72, "xyxy": list(BOX), "prompt": "plywood"}]}
    _info, dets = vi.RiverRunner(cfg([a, b], min_arms=1), "cpu")(IMG)
    assert list(dets[0]["votes"]) == ["plywood", "plank"]        # 높은 것이 먼저
    assert dets[0]["votes"]["plywood"] == 0.72
