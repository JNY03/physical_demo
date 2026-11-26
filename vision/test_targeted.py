"""지정 클래스 탐지 — 카탈로그 · 프로필 검증 · 판정 역할 (2026-11-23).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_targeted.py -q

모델을 적재하지 않는다(CLIP 이 필요한 근거는 빼고 잰다) — 빠르게 도는 시험이다.
"""
import copy
import json
from pathlib import Path

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


# ---------------------------------------------------------------- 객체 재식별 (2026-11-23)
class _FakeClip:
    """crop 평균색으로 임베딩 — 같은 색 상자는 같은 물체로 본다."""
    def images(self, crops):
        v = np.array([c.reshape(-1, 3).mean(0) for c in crops], np.float32) + 1.0
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class _FakeMatcher:
    """같은 색이면 매칭 많이, 다르면 적게."""
    def count(self, pairs):
        return [40 if np.abs(a.astype(int).mean((0, 1)) - b.astype(int).mean((0, 1))).max() < 20 else 5 for a, b in pairs]


def _scene(red_at=None, blue_at=None):
    img = np.zeros((200, 300, 3), np.uint8)
    dets = []
    for colour, at, tid in (((0, 0, 220), red_at, 1), ((220, 0, 0), blue_at, 2)):
        if at is not None:
            x = at
            img[50:150, x:x + 40] = colour
            dets.append({"name": "person", "xyxy": [x, 50, x + 40, 150], "track_id": tid, "conf": 0.9})
    return img, dets


def test_reid_select_track_reacquire_and_other_camera():
    from reid import MemStore, ReidEngine
    eng = ReidEngine(MemStore(), _FakeClip(), _FakeMatcher(), log=None)
    item = [{"id": 1, "device": "cam1", "model": "targeted", "bbox": [20, 50, 60, 150], "label": "person"}]
    img, dets = _scene(red_at=20, blue_at=200)
    eng.step(item, "cam1", "targeted", img, dets, 1)
    assert dets[0]["reid"]["how"] == "selected" and "reid" not in dets[1]
    img, dets = _scene(red_at=24, blue_at=200)
    eng.step(item, "cam1", "targeted", img, dets, 2)
    assert dets[0]["reid"]["how"] == "tracked"
    for n in range(3, 8):                             # 사라졌다
        img, dets = _scene(blue_at=200)
        eng.step(item, "cam1", "targeted", img, dets, n)
        assert all("reid" not in d for d in dets)     # 다른 물체를 고르지 않는다
    hows = []
    for n in range(8, 11):                            # 멀리 다시 나타났다 — 확인 두 번 뒤에 다시 찾는다
        img, dets = _scene(red_at=120, blue_at=200)
        dets[0]["track_id"] = 7
        eng.step(item, "cam1", "targeted", img, dets, n)
        hows.append((dets[0].get("reid") or {}).get("how"))
    assert hows[0] is None and "reacquired" in hows
    img, dets = _scene(red_at=60)                    # 다른 카메라
    other = []
    for n in range(11, 14):
        img, dets = _scene(red_at=60)
        eng.step(item, "cam2", "targeted", img, dets, n)
        other.append((dets[0].get("reid") or {}).get("how"))
    assert "other_camera" in other
    import json as _j
    seen = _j.loads(eng.store.get("vision_reid_seen:1"))
    assert [t["device"] for t in seen["trail"]] == ["cam1", "cam2"]


# ---------------------------------------------------------------- 트랙 근거 누적 (2026-11-23)
def _tedet(sem, cls=80.0, gate=True, tid=1, box=(0, 0, 40, 80), local=0.2):
    cv = {"class": {"pct": cls, "role": "score", "passed": None, "w": 0.5},
          "semantic": {"pct": sem, "role": "score", "passed": None, "w": 0.5},
          "aspect": {"pct": 100.0 if gate else 0.0, "role": "gate", "passed": gate, "w": 0.0}}
    return {"name": "person", "track_id": tid, "xyxy": list(box),
            "assess": {"suspect": None, "scores": {"person": 0.0}, "min_score": 50.0, "min_evidence": 0.45,
                       "cue_values": cv, "cards": {"person": {"rows": [["class", cls, "x"], ["semantic", sem, "y"]]}},
                       "axes": {"global": False, "local": local, "local_declared": True, "geometry": None}}}


def test_track_evidence_averages_cues_over_frames():
    import vision_infer as vi
    te = vi.TrackEvidence()
    d = te.update([_tedet(10.0)])[0]                  # 첫 장: (80+10)/2 = 45 < 50
    assert d["assess"]["suspect"] is None and abs(d["assess"]["score"] - 45.0) < 1e-6
    for _ in range(3):
        d = te.update([_tedet(60.0)])[0]              # 쌓이면 평균이 올라 판정된다
    assert d["assess"]["suspect"] == "person" and d["assess"]["track_frames"] == 4
    assert any(r[0] == "track" and "4 frames" in r[2] for r in d["assess"]["cards"]["person"]["rows"])


def test_track_evidence_gate_majority_ignores_single_failure():
    import vision_infer as vi
    te = vi.TrackEvidence()
    for g in (True, True, False, True):               # 한 장의 우연한 탈락은 판정을 뒤집지 않는다
        d = te.update([_tedet(80.0, gate=g)])[0]
    assert d["assess"]["suspect"] == "person"
    te2 = vi.TrackEvidence()
    for g in (False, False, True):
        d = te2.update([_tedet(80.0, gate=g)])[0]
    assert d["assess"]["suspect"] is None and "aspect" in d["assess"]["gates_failed"]


# ---------------------------------------------------------------- 사람이 넣는 단서 ① 글(프롬프트) (2026-11-26)
def test_prompt_rejects_unusable_text():
    """조용히 다듬지 않는다 — 왜 못 쓰는지 한 줄로 돌려준다(런처 상태 줄에 그대로 나간다)."""
    for bad, why in (("", "비었"), ("  ", "비었"), ("ab", "짧"),
                     ("노란 조끼를 입은 사람", "영어"),            # CLIP 글 인코더는 영어로 배웠다
                     ("a box {with} braces", "쓸 수 없는 글자"),
                     ("a vest person" * 20, "깁니다"),
                     ("123 456", "낱말")):
        with pytest.raises(cc.ProfileError, match=why):
            cc.check_prompts(bad)
    with pytest.raises(cc.ProfileError, match="개까지"):
        cc.check_prompts(["a red box %d" % i for i in range(cc.PROMPT_MAX_COUNT + 1)])


def test_prompt_normalises_and_dedupes():
    got = cc.check_prompts(["  a person  in a\tyellow vest ", "a person in a yellow vest", "a yellow helmet"])
    assert got == ["a person in a yellow vest", "a yellow helmet"]
    assert cc.check_prompts("a red box") == ["a red box"]       # 글 하나도 받는다
    assert cc.check_prompts(None) == []                         # 안 넣은 것은 오류가 아니다


def test_prompt_turns_on_semantic_and_replaces_descriptions():
    t = cc._direct_target("vest", {"prompt": "a person in a yellow safety vest"})
    assert t["user_descriptions"] == ["a person in a yellow safety vest"]
    assert t["descriptions"] == t["user_descriptions"]          # 사람이 적은 글이 자동 문장을 대신한다
    assert t["_descriptions_default"][0].startswith("a photo of a vest")
    assert t["cues"]["semantic"] == cc.CUES["semantic"]["role"]
    # 자동 반대 문장이 사람이 적은 낱말과 부딪히면 그것만 뺀다 — 자기 자신과 다투지 않게
    assert "a person" not in t["negatives"] and "a wall" in t["negatives"]
    plain = cc._direct_target("vest", {})
    assert "user_descriptions" not in plain and "a person" in plain["negatives"]


def test_prompt_in_target_arg_and_profile():
    s = cc.parse_target_arg("vest:prompt=a person in a yellow vest:colour=yellow")
    assert s["prompt"] == ["a person in a yellow vest"] and s["colours"] == ["yellow"]
    with pytest.raises(cc.ProfileError, match="영어"):
        cc.parse_target_arg("vest:prompt=노란 조끼")
    with pytest.raises(cc.ProfileError, match="모르는 칸"):
        _resolve({"models": {"targeted": {"targets": [{"name": "vest", "prompts": "a vest"}]}}})
    with pytest.raises(cc.ProfileError, match="글 단서"):       # 프로필의 틀린 글도 거절한다
        _resolve({"models": {"targeted": {"targets": [{"name": "vest", "prompt": ""}]}}})


@pytest.mark.skipif(not cc.have("yoloe_text", cc.model_files({})), reason="YOLOE 텍스트 가중치 없음")
def test_profile_prompt_reaches_resolved_target():
    r = _resolve({"models": {"targeted": {"targets": [
        {"name": "door", "prompt": "a wooden door with a metal handle"}]}}})
    t = r["targeted"]["targets"][0]
    assert t["descriptions"] == ["a wooden door with a metal handle"]
    assert t["cues"].get("semantic") or "semantic" in t["unavailable"]


# ---------------------------------------------------------------- 사람이 넣는 단서 ② 기준 사진 (2026-11-26)
def _ref_zip(tmp_path, n=3, junk=True):
    """기준 사진 zip 하나 — 그림 아닌 것과 폴더 밖으로 나가려는 이름을 섞어 둔다."""
    import zipfile
    import cv2
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    z = tmp_path / "refs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for i in range(n):
            img = np.full((300, 400, 3), 20 + 30 * i, np.uint8)
            img[50:200, 80:300] = (40, 200, 240)
            f = src / ("ref%d.png" % i)
            cv2.imwrite(str(f), img)
            zf.write(f, "photos/ref%d.png" % i)
        if junk:
            zf.writestr("photos/notes.txt", b"not a photo")
            zf.writestr("../evil.png", b"not a photo either")
    return z


NO_CLIP = {"models": {"targeted": {"files": {"clip": ["models/nope/vision_model_quantized.onnx"]}}}}


def test_reference_zip_kept_even_when_weights_missing(tmp_path):
    """가중치가 없어도 **사진은 남는다** — 뒤에 받으면 그때 만든다. 조용히 비우지 않고 왜 못 했는지 돌려준다."""
    cc.save_direct_target("cone", target_dir=tmp_path)
    out = cc.attach_reference_images("cone", [_ref_zip(tmp_path)], target_dir=tmp_path, cfg=NO_CLIP)
    assert out["state"] == "pending" and "vision_model_quantized.onnx" in out["why"]
    assert out["files"] == ["01.png", "02.png", "03.png"]       # 이름은 우리가 붙인다 (zip 이름을 믿지 않는다)
    assert [a for a, _b in out["skipped"]] == ["photos/notes.txt", "evil.png"]
    tf, ref_dir, npz = cc.ref_paths("cone", tmp_path)
    assert sorted(p.name for p in ref_dir.glob("*")) == ["01.png", "02.png", "03.png"]
    assert not npz.exists()
    t = json.loads(tf.read_text(encoding="utf-8"))
    assert t["reference_images"]["count"] == 3 and t["reference_images"]["state"] == "pending"
    assert "ref_match" not in t["cues"]                         # 못 만든 근거는 켜지 않는다
    # 가중치가 없는 기계에서도 상태는 읽힌다 — **없음(None)과 pending 은 다르다**
    assert cc.ref_status("cone", tmp_path)["state"] == "pending"
    cc.save_direct_target("plain", target_dir=tmp_path)
    assert cc.ref_status("plain", tmp_path) is None


_ENC = cc.ref_encoder_file({})


@pytest.mark.skipif(not (_ENC and _ENC.exists()), reason="CLIP 비전 인코더 없음")
def test_reference_zip_embeds_with_runtime_preprocess(tmp_path):
    cc.save_direct_target("cone", target_dir=tmp_path)
    out = cc.attach_reference_images("cone", [_ref_zip(tmp_path)], target_dir=tmp_path)
    assert out["state"] == "ready" and out["why"] is None
    tf, _d, npz = cc.ref_paths("cone", tmp_path)
    z = np.load(npz)
    emb = z["%s/cone" % cc.REF_ENCODER]
    assert emb.shape == (3, 512) and emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb[0])) - 1.0) < 1e-3     # L2 정규화 — 코사인으로 바로 견준다
    t = json.loads(tf.read_text(encoding="utf-8"))
    # 식이 붙은 뒤로 기본 역할은 score 다 (target_assess 가 기준 사진과의 닮은 정도를 %로 낸다)
    assert t["cues"]["ref_match"] == "score" and t["reference_images"]["embeddings"] == npz.name
    # 같은 사진을 다시 임베딩하면 같은 자리에 온다. 다만 **양자화 ONNX 는 묶음 크기에 따라 조금
    # 달라진다** — 3장 묶음의 한 줄 대 1장은 코사인 0.990 (잰 값, _embed_refs 주석). 같은 묶음끼리는 1.000
    one, _prov = cc._embed_refs([_d / "01.png"], _ENC)
    again, _prov = cc._embed_refs([_d / "01.png"], _ENC)
    assert float(one[0] @ again[0]) > 0.9999
    assert 0.98 < float(one[0] @ emb[0]) < 1.0


def test_reference_zip_without_images_is_rejected(tmp_path):
    import zipfile
    cc.save_direct_target("cone", target_dir=tmp_path)
    z = tmp_path / "empty.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", b"no photos here")
    with pytest.raises(cc.ProfileError, match="하나도 못 읽"):
        cc.attach_reference_images("cone", [z], target_dir=tmp_path, cfg=NO_CLIP)
    with pytest.raises(cc.ProfileError, match="등록된 대상이 아님"):
        cc.attach_reference_images("ghost", [z], target_dir=tmp_path, cfg=NO_CLIP)
    with pytest.raises(cc.ProfileError, match="못 찾음"):
        cc.attach_reference_images("cone", [tmp_path / "nothing.zip"], target_dir=tmp_path, cfg=NO_CLIP)


def test_reference_rebuild_after_weights_arrive(tmp_path):
    """사진만 보관해 둔 대상 — 가중치가 생기면 `--build-refs` 가 그때 만든다."""
    cc.save_direct_target("cone", target_dir=tmp_path)
    cc.attach_reference_images("cone", [_ref_zip(tmp_path, junk=False)], target_dir=tmp_path, cfg=NO_CLIP)
    again = cc.build_ref_embeddings("cone", tmp_path, NO_CLIP)
    assert again["state"] == "pending" and again["images"] == 3
    cc.save_direct_target("plain", target_dir=tmp_path)
    with pytest.raises(cc.ProfileError, match="보관된 기준 사진이 없"):
        cc.build_ref_embeddings("plain", tmp_path, NO_CLIP)


def test_catalog_carries_prompt_guide_for_launcher():
    cat = cc.catalog(CFG)
    g = cat["prompt_guide"]
    assert 2 <= len(g["lines"]) <= 5 and g["good"] and g["bad"]
    assert g["limits"]["max_chars"] == cc.PROMPT_MAX_CHARS
    assert cat["ref_images"]["max_images"] == cc.REF_MAX_IMAGES and cat["ref_images"]["guide"]
    for t in cat["targets"]:
        assert t["prompt"] == [] and t["refs"] is None          # 등록 대상에는 아직 사람이 넣은 것이 없다
