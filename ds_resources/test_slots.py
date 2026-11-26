"""역할 슬롯 — 기능 → 단계 → 역할 슬롯 → 모델 (slots/1, 2026-11-27).

옛 칸(options · cue_model_choices)이 슬롯으로 바뀌는지, 모델 하나를 등록하면 능력 · 도는 곳이 맞는 슬롯에
저절로 뜨는지, 프로필 functions 칸 검사 · cue_setup 별칭 · places JSON · 비용이 그대로인지를 본다.
"""
import copy
import json
import sys
import textwrap
from pathlib import Path

import pytest

from ds_resources import functions as fnc
from ds_resources import slots as slt

REPO = Path(__file__).resolve().parent.parent


def _reg():
    return fnc._registry(None)


def _with_models(**models):
    """실제 레지스트리에 모델을 더한 사본 — 다른 시험에 새지 않게 깊은 사본"""
    reg = copy.deepcopy(_reg())
    reg["models"].update(models)
    return reg


# ---------------------------------------------------------------- 옛 칸 → 슬롯
def test_old_options_become_server_model_slot():
    """options: [...] 는 서버 단계 model 슬롯이 된다 — choices = options, 기본 = 첫째. 옛 options 칸도 그대로 남는다"""
    reg = {"functions": {"f": {"pattern": "server_only", "criticality": "mission", "trigger": "on_request",
                               "note": "옛 모양", "options": ["unidepth", "moge2_aerial"],
                               "stages": {"server": {"cpu_m": 1, "mem_mb": 1}}}},
           "options": _reg()["options"], "models": _reg()["models"]}
    f = fnc.load(reg)["f"]
    assert f["options"] == ["unidepth", "moge2_aerial"]
    (sl,) = f["stages"]["server"]["slots"]
    assert sl["role"] == "model" and sl["choices"] == ["unidepth", "moge2_aerial"] and sl["default"] == "unidepth"
    assert sl["origin"] == "options" and sl["auto"] is False and sl["optional"] is False
    rows = slt.function_slots(reg, have=lambda k: True)["f"]["server"]
    assert [c["key"] for c in rows[0]["choices"]] == ["unidepth", "moge2_aerial"]


def test_real_registry_keeps_options_and_gains_slots():
    f = fnc.load()
    assert f["object_detect"]["options"] == ["yoloe"]                  # 런처 function_options 는 그대로
    roles = {(s["role"]) for s in f["object_detect"]["stages"]["server"]["slots"]}
    assert roles == {"model"}
    assert [s["role"] for s in f["object_detect"]["stages"]["device"]["slots"]] == ["detector"]
    assert f["single_cam_track"]["options"] == []


def test_cue_model_choices_become_targeted_slots():
    """cue_model_choices → targeted_detect 서버 단계의 ovd · depth 슬롯 (선택, multi 는 그대로)"""
    st = fnc.load()["targeted_detect"]["stages"]["server"]
    by = {s["role"]: s for s in st["slots"]}
    assert by["model"]["choices"] == ["targeted"]
    assert by["ovd"]["choices"] == ["yolo_world", "grounding_dino"] and by["ovd"]["multi"] is True
    assert by["ovd"]["default"] == ["yolo_world"] and by["ovd"]["optional"] is True
    assert by["depth"]["choices"] == ["depth", "unidepth_l"] and by["depth"]["multi"] is False
    assert by["depth"]["origin"] == "cue_model_choices"


def test_registry_slots_are_consistent():
    reg = _reg()
    assert slt.check_registry(reg)


def test_bad_slot_declaration_is_refused():
    reg = {"functions": {"f": {"pattern": "server_only", "criticality": "mission", "trigger": "on_request",
                               "note": "x", "stages": {"server": {"cpu_m": 1, "mem_mb": 1, "slots": {
                                   "model": {"needs": ["detect"], "size": "huge"}}}}}}}
    with pytest.raises(ValueError, match="size"):
        fnc.load(reg)
    reg["functions"]["f"]["stages"]["server"]["slots"] = {"model": {"wat": 1}}
    with pytest.raises(ValueError, match="모르는 칸"):
        fnc.load(reg)


# ---------------------------------------------------------------- 선택지 계산
def test_new_model_appears_where_its_capabilities_fit():
    """한 번 등록하면 needs 가 맞고 그 단계에서 돌 수 있는 슬롯마다 뜬다 — 기본값이 맨 앞"""
    reg = _with_models(
        big_det={"runner": "big:Big", "provides": ["detect", "segment", "open_vocab"], "runs_on": ["gpu"],
                 "gpu_mb": 800, "files": []},
        pi_det={"runner": "pi:Pi", "provides": ["detect"], "runs_on": ["cpu_arm64", "cpu_x86"], "size": "small",
                "files": []},
        vlm_x={"runner": "v:V", "provides": ["vlm"], "runs_on": ["gpu"], "files": []},
        comp_only={"provides": ["detect"], "runs_on": ["gpu"], "files": []})        # runner 없음 → 구성품
    fs = slt.function_slots(reg, have=lambda k: True)
    keys = lambda fn, side, role: [c["key"] for r in fs[fn][side] if r["role"] == role for c in r["choices"]]
    assert keys("object_detect", "server", "model") == ["yoloe", "big_det"]          # 기본 먼저, gpu 만
    # 기기 슬롯은 실제 레지스트리의 CPU 모델(yolo26n_seg_cpu …)도 뜬다 — 여기서 더한 것만 본다
    added = lambda ks: [k for k in ks if k in ("big_det", "pi_det", "vlm_x", "comp_only")]
    assert added(keys("object_detect", "device", "detector")) == ["pi_det"]          # 파이 · small
    assert keys("zeroshot_detect", "server", "model") == ["yoloe", "big_det"]
    assert added(keys("zeroshot_detect", "device", "detector")) == []                # open_vocab 이 없다
    assert keys("segmentation", "server", "model") == ["yoloe", "big_det"]
    assert keys("vlm", "server", "model") == ["vlm_x"]
    assert "comp_only" not in keys("object_detect", "server", "model")              # 혼자 못 도는 것은 안 뜬다
    assert keys("river_detect", "server", "model") == ["river"]                      # 옛 슬롯은 명시만
    row = next(c for r in fs["object_detect"]["server"] for c in r["choices"] if c["key"] == "big_det")
    assert row["gpu_mb"] == 800 and row["runs_on"] == ["gpu"] and row["available"] is True
    assert set(row) >= {"key", "shown", "provides", "runs_on", "size", "gpu_mb", "p90_ms", "license",
                        "available", "note"}


def test_fits_rules():
    slot = {"needs": ["detect"], "size": "small", "auto": True}
    small_pi = {"runnable": True, "provides": ["detect"], "runs_on": ["cpu_arm64"], "size": "small"}
    assert slt.fits(small_pi, slot, "device")
    assert not slt.fits(dict(small_pi, size=None), slot, "device")                   # small 슬롯
    assert not slt.fits(small_pi, slot, "server", {"needs": {"accel": "gpu"}})       # 파이 전용은 서버에 안 뜬다
    x86 = {"runnable": True, "provides": ["detect"], "runs_on": ["cpu_x86"]}
    assert slt.fits(x86, dict(slot, size="any"), "server", {"needs": {"accel": "none"}})
    assert not slt.fits(x86, dict(slot, size="any"), "server", {"needs": {"accel": "gpu"}})
    assert not slt.fits(dict(x86, runnable=False), dict(slot, size="any"), "server")


def test_availability_follows_files():
    fs = slt.function_slots(_reg(), have=lambda k: k != "yoloe_pf_l")
    yo = next(c for c in fs["object_detect"]["server"][0]["choices"] if c["key"] == "yoloe")
    assert yo["available"] is False and "가중치 없음" in yo["note"]


# ---------------------------------------------------------------- 프로필 (M4)
def test_profile_picks_good_and_bad():
    reg = _with_models(big_det={"runner": "big:Big", "provides": ["detect", "segment", "open_vocab"],
                                "runs_on": ["gpu"], "files": []})
    funcs = fnc.load(reg)
    ok = slt.check_picks(reg, funcs, {"object_detect": {"server": {"model": "big_det"}},
                                      "targeted_detect": {"server": {"ovd": ["grounding_dino"], "depth": "unidepth_l"}},
                                      "single_cam_track": {"device": {"tracker": None}}})
    assert ok["object_detect"]["server"]["model"] == "big_det"
    assert ok["targeted_detect"]["server"]["ovd"] == ["grounding_dino"]
    assert ok["single_cam_track"]["device"]["tracker"] is None
    for bad, msg in [({"nope": {}}, "모르는 기능"),
                     ({"object_detect": {"cloud": {}}}, "단계가 없다"),
                     ({"object_detect": {"server": {"tracker": "x"}}}, "역할이 없다"),
                     ({"object_detect": {"server": {"model": "river"}}}, "고를 수 없는"),
                     ({"object_detect": {"server": {"model": None}}}, "비울 수 없다"),
                     ({"object_detect": {"server": {"model": ["yoloe", "big_det"]}}}, "하나만"),
                     ({"targeted_detect": {"server": {"depth": "moge2_aerial"}}}, "고를 수 없는")]:
        with pytest.raises(ValueError, match=msg):
            slt.check_picks(reg, funcs, bad)
    eff = slt.effective(reg, funcs, ok, {"object_detect", "depth_estimate"})
    assert eff["object_detect"] == {"device": {"detector": None}, "server": {"model": "big_det"}}
    assert eff["depth_estimate"]["server"]["model"] == "unidepth"


def _cc():
    sys.path.insert(0, str(REPO / "vision"))
    import cue_catalog as cc
    return cc


def test_resolve_profile_carries_slots_and_aliases_cue_setup():
    cc = _cc()
    prof = {"functions": {"targeted_detect": {"server": {"ovd": ["grounding_dino"], "depth": "unidepth_l"}}},
            "models": {"targeted": {"targets": ["door"]}}}
    r = cc.resolve_profile(prof, {}, ["targeted"])
    assert r["slots"]["targeted_detect"]["server"]["depth"] == "unidepth_l"
    assert r["targeted"]["cue_models"] == {"ovd": ["grounding_dino"], "depth": "unidepth_l"}   # = cue_setup.models
    # 옛 cue_setup 만 적은 프로필은 그대로 — slots 는 빈 사전
    old = {"models": {"targeted": {"targets": ["door"], "cue_setup": {"models": {"depth": "unidepth_l"}}}}}
    r2 = cc.resolve_profile(old, {}, ["targeted"])
    assert r2["slots"] == {} and r2["targeted"]["cue_models"] == {"depth": "unidepth_l"}
    # 같은 것을 두 곳에 같게 적으면 된다 · 다르게 적으면 거절
    same = copy.deepcopy(old)
    same["functions"] = {"targeted_detect": {"server": {"depth": "unidepth_l"}}}
    assert cc.resolve_profile(same, {}, ["targeted"])["targeted"]["cue_models"]["depth"] == "unidepth_l"
    same["functions"]["targeted_detect"]["server"]["depth"] = "depth"
    with pytest.raises(cc.ProfileError, match="두 곳"):
        cc.resolve_profile(same, {}, ["targeted"])
    with pytest.raises(cc.ProfileError, match="고를 수 없는"):
        cc.resolve_profile({"functions": {"object_detect": {"server": {"model": "river"}}}}, {}, ["yoloe"])


def test_old_profiles_still_validate():
    cc = _cc()
    for prof in ({}, {"label": "x", "models": {"yoloe": {"overlay": {"count": True}}}},
                 {"models": {"targeted": {"targets": ["door"], "cue_setup": {"off": ["aspect"],
                                                                             "models": {"ovd": ["yolo_world"]}}}}}):
        r = cc.resolve_profile(prof, {}, ["yoloe", "targeted"])
        assert r["slots"] == {}


# ---------------------------------------------------------------- 비용
def test_costs_unchanged_for_old_choices():
    cc = _cc()
    assert cc.gpu_mb(["yoloe"]) == 1300 and cc.gpu_mb(["river"]) == 2800
    assert cc.gpu_mb(["targeted"], ["yoloe_text", "clip"]) == 400 + 900 + 350


def test_runner_model_cost_is_counted_once(monkeypatch):
    cc = _cc()
    monkeypatch.setitem(cc.RUNNER_MODELS, "big_det", "detect")
    monkeypatch.setitem(cc.GPU_MB, "big_det", 800)
    assert cc.gpu_mb(["yoloe", "big_det"]) == 400 + 900 + 800
    assert cc.gpu_mb(["big_det"], ["big_det", "clip"]) == 400 + 800 + 350          # 슬롯과 근거가 같은 모델 — 한 번


def test_stage_costs_dedupe_across_functions():
    reg = _with_models(big_det={"runner": "big:Big", "provides": ["detect", "segment", "open_vocab"],
                                "runs_on": ["gpu"], "gpu_mb": 800, "p90_ms": {"s": 40, "m": 60}, "files": []})
    sel = {"object_detect": {"server": {"model": "big_det"}}, "segmentation": {"server": {"model": "big_det"}},
           "depth_estimate": {"server": {"model": "unidepth"}},
           "targeted_detect": {"server": {"ovd": ["yolo_world", "grounding_dino"]}}}
    c = slt.stage_costs(reg, sel)["server"]
    assert c["models"] == ["big_det", "grounding_dino", "unidepth", "yolo_world"]
    assert c["gpu_mb"] == 800 + 1300 + 1500 + 500
    assert c["p90_ms"]["s"] == 40 + 370                                              # 같은 모델은 한 번


def test_controller_p90_unchanged_and_runner_models(tmp_path):
    from ds_resources import test_controller as tc
    k, _r, _c = tc.make(tmp_path)
    src = {"a": {"w": 1280, "h": 720}}
    assert k.costs(["yoloe"], src)["p90_ms"]["a"] == 135.0
    assert k.costs(["unidepth", "yoloe"], src)["p90_ms"]["a"] == 135.0 + 400.0
    k.reg = _with_models(big_det={"runner": "big:Big", "provides": ["detect"], "p90_ms": {"m": 70}})
    assert k._p90("big_det", "m") == 70 and k._p90("big_det", "xl") == 500 and k._p90("nope", "s") == 500


# ---------------------------------------------------------------- places JSON
def test_places_json_has_function_slots(tmp_path):
    from ds_resources import test_controller as tc
    k, _r, _c = tc.make(tmp_path)
    pl = k.places()
    assert "function_options" in pl and "model_info" in pl                 # 런처가 읽는 옛 칸은 그대로
    fs = pl["function_slots"]
    assert pl["function_options"]["object_detect"] == ["yoloe"]
    row = fs["object_detect"]["server"][0]
    assert set(row) >= {"role", "label", "label_en", "needs", "optional", "multi", "choices", "default"}
    assert row["role"] == "model" and row["default"] == "yoloe" and row["choices"][0]["key"] == "yoloe"
    assert {r["role"] for r in fs["targeted_detect"]["server"]} == {"model", "ovd", "depth"}
    json.dumps(pl, ensure_ascii=False)                                     # 그대로 JSON 이 된다
    assert pl["slot_selection"]["depth_estimate"]["server"]["model"] == "unidepth"


# ---------------------------------------------------------------- 실행기 적재 (M2)
def test_load_runner_with_dummy_module(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "vision"))
    import runners as rn
    pkg = tmp_path / "dummy_runner_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "det.py").write_text(textwrap.dedent('''
        class Det:
            name = "dummy_det"
            kind = "detect"

            def __init__(self, cfg, device):
                self.cfg, self.device = cfg, device

            def __call__(self, bgr):
                return {"infer_ms": 1.0}, [{"cls": 0, "name": "x", "conf": 0.9, "xyxy": [0, 0, 1, 1]}]
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = {"runner": "dummy_runner_pkg.det:Det", "provides": ["detect"], "config": {"conf": 0.3}}
    r = rn.load_runner("dummy_det", spec, "cpu")
    assert r.cfg == {"conf": 0.3} and r.device == "cpu" and r.kind == "detect"
    info, dets = r(None)
    assert dets[0]["xyxy"] == [0, 0, 1, 1]
    assert rn.load_runner("dummy_det", spec, "cpu", {"conf": 0.5}).cfg == {"conf": 0.5}
    with pytest.raises(rn.RunnerError, match="못 읽음"):
        rn.load_runner("x", {"runner": "no_such_module_zz:X"}, "cpu")
    with pytest.raises(rn.RunnerError, match="없다"):
        rn.load_runner("x", {"runner": "dummy_runner_pkg.det:Nope"}, "cpu")
    with pytest.raises(rn.RunnerError, match="모듈:클래스"):
        rn.load_runner("x", {"runner": "yoloe"}, "cpu")
    assert rn.runner_kind({"provides": ["depth_metric"]}) == "depth"
    reg = {"options": {"yoloe": {}}, "models": {"a": {"runner": "a:A"}, "b": {}, "yoloe": {"runner": "y:Y"}}}
    assert set(rn.runner_models(reg)) == {"a"}


# ---------------------------------------------------------------- preflight
def test_preflight_required_slot_without_model_blocks(monkeypatch):
    """필수 슬롯에 맞는 모델이 하나도 없으면 그 기능은 안 된다 — 까닭에 역할과 능력을 적는다"""
    from ds_resources import preflight as pf
    monkeypatch.setattr(pf, "_weight_state", lambda models, dest=None: {m: (True, "") for m in models})
    mach = {"gpu": "x", "kubectl": True}
    by = {r["key"]: r for r in pf.features(mach, fnc.load())[0]}
    assert by["vlm"]["usable"] is False and "vlm" in by["vlm"]["why"] and "weights" in by["vlm"]["needs"]
    assert by["object_detect"]["usable"] is True                     # 기기 슬롯은 선택 — 막지 않는다
    reg = _with_models(vlm_x={"runner": "v:V", "provides": ["vlm"], "runs_on": ["gpu"], "files": []})
    by = {r["key"]: r for r in pf.features(mach, fnc.load(reg), reg=reg)[0]}
    assert by["vlm"]["usable"] is True and by["vlm"]["runnable"] is True
