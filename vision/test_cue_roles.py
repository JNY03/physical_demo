"""지정 물체 탐지 — 근거마다 모델 고르기 (근거 모델 역할, 2026-11-27).

역할 슬롯 detector · free_arm · embed (+ 옛 ovd · depth) 가 레지스트리 능력(provides)으로 선택지를 갖는지,
프로필 functions.targeted_detect.server.<역할> 검사(좋은 것 · 모르는 모델 · 능력이 다른 모델 · 가중치 없음),
**아무것도 고르지 않으면 예전과 같은지**(골든), 고른 것이 needed_models · GPU · 쓸 수 있는 근거 · 카탈로그 ·
TargetedRunner 까지 가는지를 본다. GPU 시험은 가중치 · CUDA 가 없으면 건너뛴다.
"""
import copy
import os
from pathlib import Path

import pytest

os.environ.setdefault("YOLO_AUTOINSTALL", "False")      # ultralytics 가 패키지를 몰래 바꾸지 않게 (onnxruntime-gpu 사건)
import cue_catalog as cc  # noqa: E402
from ds_resources import functions as fnc  # noqa: E402
from ds_resources import slots as slt  # noqa: E402

CFG = {"models": {"targeted": {}}}
NEW = {"detector": "yoloe26_text_s", "free_arm": "yoloe26_pf_s", "embed": "clip_fp32"}


def _resolve(profile, models=("targeted",)):
    return cc.resolve_profile(profile, CFG, list(models))


def _prof(roles=None, targets=("person", "chair", "door"), **targeted):
    p = {"models": {"targeted": dict({"targets": list(targets)}, **targeted)}}
    if roles is not None:
        p["functions"] = {"targeted_detect": {"server": dict(roles)}}
    return p


def _files_ok(*keys):
    files = cc.model_files({})
    return all(cc.have_model(k, files) for k in keys)


# ---------------------------------------------------------------- 레지스트리 · 슬롯
def test_role_slots_choices_come_from_capabilities():
    rs = cc.role_slots()
    assert set(rs) == {"detector", "free_arm", "embed"}
    assert rs["detector"]["choices"][0] == "yoloe_text"
    assert set(rs["detector"]["choices"]) == {"yoloe_text", "yoloe26_text_l", "yoloe26_text_s"}
    assert rs["free_arm"]["choices"][0] == "yoloe_free"
    assert set(rs["free_arm"]["choices"]) == {"yoloe_free", "yoloe26_pf_s", "yoloe26_pf_l"}
    assert rs["embed"]["choices"] == ["clip", "clip_fp32"]
    # mobileclip_s0_cpu 는 글 인코더가 없고 임베딩 공간이 달라 embed 에 뜨지 않는다 · COCO 닫힌 어휘(rf_detr)도 아니다
    allc = {k for r in rs.values() for k in r["choices"]}
    assert "mobileclip_s0_cpu" not in allc and not any(k.startswith("rf_detr") for k in allc)


def test_every_cue_names_a_known_role():
    roles = set(cc.role_slots()) | set(cc.CUE_CHOICES) | {"none"}
    for k, c in cc.CUES.items():
        assert c["model_role"] in roles, k
    assert cc.CUES["class"]["model_role"] == "detector" and cc.CUES["parts"]["model_role"] == "detector"
    assert cc.CUES["free_arm"]["model_role"] == "free_arm"
    for k in ("semantic", "dict_features", "clip_colour", "clip_shape", "lookalike", "ref_match"):
        assert cc.CUES[k]["model_role"] == "embed"
    assert cc.CUES["ovd_pair_iou"]["model_role"] == "ovd" and cc.CUES["size_veto"]["model_role"] == "depth"
    # needs 에 적힌 기본 모델 키는 그 역할의 기본값이다 (고르면 바뀌는 자리)
    for k, c in cc.CUES.items():
        for n in c["needs"]:
            if cc.need_role(n):
                assert cc.need_role(n) == c["model_role"], (k, n)


def test_new_model_with_capability_appears_in_role(tmp_path):
    """능력만 적으면 저절로 뜬다 — runner 가 없어도(component 슬롯)"""
    reg = copy.deepcopy(fnc._registry(None))
    reg["models"]["my_clip"] = {"provides": ["embed_image_text"], "device": "gpu", "files": []}
    reg["models"]["my_pf_cpu"] = {"provides": ["detect_free"], "device": "cpu", "files": []}   # cpu 만 — GPU 단계엔 안 뜬다
    st = fnc.load(reg)["targeted_detect"]["stages"]["server"]
    table = slt.parts(reg)
    by = {s["role"]: slt.choices(s, "server", st, table) for s in st["slots"]}
    assert "my_clip" in by["embed"] and by["embed"][0] == "clip"
    assert "my_pf_cpu" not in by["free_arm"]
    # component 가 아닌 슬롯에는 runner 없는 모델이 여전히 뜨지 않는다
    od = fnc.load(reg)["object_detect"]["stages"]["server"]["slots"][0]
    assert "yoloe_text" not in slt.choices(od, "server", fnc.load(reg)["object_detect"]["stages"]["server"], table)


def test_function_slots_rows_for_launcher():
    fs = slt.function_slots(fnc._registry(None))["targeted_detect"]["server"]
    by = {r["role"]: r for r in fs}
    assert [r["role"] for r in fs] == ["model", "detector", "free_arm", "embed", "ovd", "depth"]
    assert by["detector"]["component"] is True and by["ovd"]["component"] is False
    assert by["detector"]["optional"] is False and by["embed"]["optional"] is True
    assert by["embed"]["default"] == "clip"


def test_embed_availability_needs_text_encoder_files():
    """clip_fp32 는 river 에는 비전 인코더만, embed 로는 글 인코더 · 토크나이저까지 있어야 한다 (files_for)"""
    reg = fnc._registry(None)
    plain = slt.role_files("clip_fp32", reg)
    full = slt.role_files("clip_fp32", reg, ["embed_image_text"])
    assert len(plain) == 1 and len(full) == 3 and plain[0] == full[0]
    assert any(f.endswith("text_model.onnx") for f in full) and any(f.endswith("tokenizer.json") for f in full)


# ---------------------------------------------------------------- 기본값 = 예전 (골든)
#: 이 변경 전 HEAD(0165c53) 의 cue_catalog 로 잰 값 — 아무것도 고르지 않으면 이것과 같아야 한다
GOLDEN = [
    (_prof(), ["clip", "depth", "yolo_world", "yoloe_free", "yoloe_text"], 3950),
    (_prof(targets=("person", "door", "chair", "desk", "demo_door", "demo_box")),
     ["clip", "depth", "grounding_dino", "trained", "yolo_world", "yoloe_free", "yoloe_text"], 5370),
    (_prof(targets=("door",), distance=False), ["clip", "yolo_world", "yoloe_free", "yoloe_text"], 2650),
]


@pytest.mark.skipif(not _files_ok("yoloe_text", "yoloe_free", "clip", "yolo_world", "depth", "grounding_dino"),
                    reason="기본 가중치가 다 있어야 골든과 같다")
@pytest.mark.parametrize("prof,need,mb", GOLDEN)
def test_defaults_unchanged_golden(prof, need, mb):
    r = _resolve(prof)["targeted"]
    got = sorted(cc.needed_models(r, {}))
    assert got == need and cc.gpu_mb(["targeted"], got) == mb
    assert r["cue_models"] == {}
    assert r["role_models"] == {"detector": "yoloe_text", "free_arm": "yoloe_free", "embed": "clip",
                                "ovd": "auto", "depth": "depth"}


def test_picking_defaults_explicitly_is_the_same_as_not_picking():
    a = _resolve(_prof())["targeted"]
    b = _resolve(_prof({"detector": "yoloe_text", "free_arm": "yoloe_free", "embed": "clip"}))["targeted"]
    assert cc.needed_models(a, {}) == cc.needed_models(b, {})
    assert [t["cues"] for t in a["targets"]] == [t["cues"] for t in b["targets"]]


# ---------------------------------------------------------------- 고른 것이 따라간다
@pytest.mark.skipif(not _files_ok(*NEW.values()), reason="YOLOE-26 · CLIP fp32 가중치 없음")
def test_needed_models_and_gpu_follow_choice():
    r = _resolve(_prof(NEW))["targeted"]
    need = cc.needed_models(r, {})
    assert {"yoloe26_text_s", "yoloe26_pf_s", "clip_fp32"} <= need
    assert not need & {"yoloe_text", "yoloe_free", "clip"}
    assert r["cue_models"] == NEW and r["role_models"]["embed"] == "clip_fp32"
    g = cc.GPU_MB
    base = cc.gpu_mb(["targeted"], cc.needed_models(_resolve(_prof())["targeted"], {}))
    assert cc.gpu_mb(["targeted"], need) == base - g["yoloe_text"] - g["yoloe_free"] - g["clip"] \
        + g["yoloe26_text_s"] + g["yoloe26_pf_s"] + g["clip_fp32"]
    # 하나만 바꾸면 그것만
    one = cc.needed_models(_resolve(_prof({"embed": "clip_fp32"}))["targeted"], {})
    assert "clip_fp32" in one and "clip" not in one and "yoloe_text" in one


def test_profile_rejects_unknown_and_wrong_capability():
    for roles, msg in [({"detector": "no_such_model"}, "고를 수 없는"),
                       ({"detector": "yoloe26_pf_s"}, "고를 수 없는"),         # prompt-free 는 detect_text 가 아니다
                       ({"free_arm": "yoloe26_text_s"}, "고를 수 없는"),
                       ({"embed": "mobileclip_s0_cpu"}, "고를 수 없는"),
                       ({"detector": "rf_detr_s"}, "고를 수 없는"),
                       ({"detector": None}, "비울 수 없다"),
                       ({"embed": ["clip", "clip_fp32"]}, "하나만")]:
        with pytest.raises(cc.ProfileError, match=msg):
            _resolve(_prof(roles))
    with pytest.raises(cc.ProfileError, match="바꿀 수 없는"):          # 옛 cue_setup 별칭은 ovd · depth 만
        _resolve({"models": {"targeted": {"targets": ["door"], "cue_setup": {"models": {"detector": "yoloe26_text_s"}}}}})


def test_optional_role_null_means_default():
    r = _resolve(_prof({"free_arm": None, "embed": None}))["targeted"]
    assert r["role_models"]["free_arm"] == "yoloe_free" and r["role_models"]["embed"] == "clip"


def test_missing_weights_rejected_and_cue_unavailable(monkeypatch, tmp_path):
    """고른 모델의 파일이 없으면 프로필을 거절하고, 카탈로그 쪽 판정에서는 그 역할의 근거가 이유와 함께 빠진다"""
    gone = tmp_path / "gone.onnx"
    real = cc.role_files

    def fake(key, files, role=None):
        out = real(key, files, role)
        return [gone] + out[1:] if key == "clip_fp32" else out
    monkeypatch.setattr(cc, "role_files", fake)
    with pytest.raises(cc.ProfileError, match="가중치 없음"):
        _resolve(_prof({"embed": "clip_fp32"}))
    t = cc.load_targets()["door"]
    av = cc.available_cues(t, {}, cc.model_files({}), True, {"embed": "clip_fp32"})
    for k in ("semantic", "dict_features", "clip_shape"):
        assert av[k] == "clip_fp32 없음"
    assert av["class"] is None or "clip" not in str(av["class"])
    ch = {c["key"]: c for c in cc.cue_choices()["embed"]["choices"]}
    assert ch["clip_fp32"]["available"] is False


def test_missing_detector_makes_class_unavailable(monkeypatch, tmp_path):
    real = cc.role_files
    monkeypatch.setattr(cc, "role_files", lambda key, files, role=None:
                        [tmp_path / "x.pt", tmp_path / "y.ts"] if key == "yoloe26_text_s" else real(key, files, role))
    t = cc.load_targets()["person"]
    av = cc.available_cues(t, {}, cc.model_files({}), True, {"detector": "yoloe26_text_s"})
    assert av["class"] == "YOLOE 텍스트 가중치 없음 (yoloe26_text_s)" and av["parts"] == "yoloe26_text_s 없음"
    with pytest.raises(cc.ProfileError, match="가중치 없음"):
        _resolve(_prof({"detector": "yoloe26_text_s"}))


# ---------------------------------------------------------------- 카탈로그 (런처)
def test_catalog_shape_for_launcher():
    cat = cc.catalog({})
    ch = cat["cue_choices"]
    assert set(ch) == {"detector", "free_arm", "embed", "ovd", "depth"}
    for role in ("detector", "free_arm", "embed"):
        row = ch[role]
        assert row["multi"] is False and row["default"] == [cc.role_slots()[role]["default"]]
        assert row["label"] and row["label_en"] and row["cues"]
        for c in row["choices"]:
            assert set(c) >= {"key", "shown", "note", "available", "gpu_mb", "p90_ms"}
    assert ch["embed"]["cues"] == ["semantic", "dict_features", "clip_colour", "clip_shape", "lookalike", "ref_match"]
    assert ch["ovd"]["cues"] == ["ovd_pair_iou"] and ch["depth"]["cues"] == ["size_veto"]
    gm = {c["key"]: c["gpu_mb"] for c in ch["detector"]["choices"]}
    assert gm["yoloe_text"] == 900 and gm["yoloe26_text_s"] == 200
    rows = {r["key"]: r for r in cat["options"]["targeted"]}
    assert rows["semantic"]["model_role"] == "embed" and rows["class"]["model_role"] == "detector"
    assert rows["colour"]["model_role"] == "none"


def test_launcher_profile_shape_resolves():
    """런처가 쓰는 꼴 — functions.targeted_detect.server.<역할> + (ovd · depth 는) 같은 값의 cue_setup.models"""
    p = {"functions": {"targeted_detect": {"server": {"embed": "clip_fp32", "detector": "yoloe26_text_s",
                                                       "depth": "unidepth_l", "ovd": ["yolo_world"]}}},
         "models": {"targeted": {"cue_setup": {"models": {"depth": "unidepth_l", "ovd": ["yolo_world"]}},
                                 "distance": True, "targets": [{"name": "person"}, {"name": "chair"}]}},
         "profile_version": 1}
    r = _resolve(p)["targeted"]
    assert r["role_models"] == {"detector": "yoloe26_text_s", "free_arm": "yoloe_free", "embed": "clip_fp32",
                                "ovd": ["yolo_world"], "depth": "unidepth_l"}
    # 두 곳이 다르면 거절 (옛 별칭 규칙 그대로)
    p["models"]["targeted"]["cue_setup"]["models"]["depth"] = "depth"
    with pytest.raises(cc.ProfileError, match="두 곳에서 다르게"):
        _resolve(p)


def test_gpu_executor_estimate_follows_depth_choice():
    from ds_resources import gpu_executor as ge
    mb_d, need_d = ge.estimate_mb(["targeted"], _prof(targets=("door",)), {})
    mb_u, need_u = ge.estimate_mb(["targeted"], _prof({"depth": "da3_metric_l"}, targets=("door",)), {})
    assert "depth" in need_d and "da3_metric_l" in need_u and "depth" not in need_u


# ---------------------------------------------------------------- 비전 (GPU)
def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not (_gpu() and _files_ok(*NEW.values(), "yolo_world")), reason="CUDA · 가중치 없음")
def test_targeted_runner_builds_with_chosen_models(monkeypatch):
    monkeypatch.setenv("YOLO_AUTOINSTALL", "False")
    import vision_infer as vi
    import target_assess as ta
    import numpy as np
    cfg = vi.load_config(Path(vi.HERE) / "vision_config.yaml")
    res = cc.resolve_profile(_prof(NEW, targets=("person", "chair"), distance=False), cfg, ["targeted"])
    vi.apply_profile(cfg, res)
    r = vi.TargetedRunner(cfg["models"]["targeted"], "cuda:0")
    names = dict((l, a.name) for l, a in r.arms)
    assert names["yoloe"] == "yoloe-26s-seg.pt" and names["yoloe_free"] == "yoloe-26s-seg-pf.pt"
    assert r.arms[0][1].encoder == "mobileclip2_b.ts"
    assert r.role_models == NEW
    files = [Path(p).name for p in cc.role_files("clip_fp32", cc.model_files(cfg["models"]["targeted"]), "embed")]
    assert files == ["vision_model.onnx", "text_model.onnx", "tokenizer.json"]
    assert r.evidence.clip is ta._shared_clip(cc.role_files("clip_fp32", cc.model_files(cfg["models"]["targeted"]),
                                                             "embed"))
    info, dets = r(np.zeros((480, 640, 3), np.uint8))
    assert info["missing_arms"] == []
    # 되돌리면(기본) 옛 모델로 — 같은 프로세스에서 다시 세워도 깨지지 않는다(reconfigure)
    res0 = cc.resolve_profile(_prof(targets=("person", "chair"), distance=False), cfg, ["targeted"])
    r.prewarm(res0["targeted"])
    r.reconfigure(res0["targeted"])
    assert dict((l, a.name) for l, a in r.arms)["yoloe"] == "yoloe-11l-seg.pt"
    assert r.arms[0][1].encoder == "mobileclip_blt.ts"

