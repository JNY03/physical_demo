"""model_registry.yaml — 선택지 · 단서 · 모델을 한 파일에서 정의한다 (2026-11-24).

레지스트리로 옮기기 전 cue_catalog.py 의 상수 값을 골든 값으로 둔다. 레지스트리를 고치면 이 값이 바뀌는 것이
의도인지 여기서 드러난다.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cue_catalog as cc  # noqa: E402

GOLD_GPU_MB = {"base": 400, "yoloe_text": 900, "yoloe_free": 500, "clip": 350, "yolo_world": 500,
               "grounding_dino": 1300, "gdino": 1300, "reid": 700, "depth": 1300, "trained": 120}
GOLD_FILE_KEYS = {"yoloe_text", "yoloe_free", "clip", "yolo_world", "gdino", "depth", "trained", "reid"}


def test_options_and_kinds():
    assert cc.MODELS == ("yoloe", "river", "targeted", "unidepth", "moge2_aerial")
    assert cc.DETECT_MODELS == ("yoloe", "river", "targeted")
    assert cc.DEPTH_MODELS == ("unidepth", "moge2_aerial")
    assert cc.MODEL_SHOWN["targeted"] == "지정 클래스 탐지"


def test_gpu_estimates_unchanged():
    assert cc.GPU_MB == GOLD_GPU_MB
    assert cc.gpu_mb(["yoloe"]) == 1300 and cc.gpu_mb(["river"]) == 2800
    assert cc.gpu_mb(["unidepth"]) == 1900 and cc.gpu_mb(["moge2_aerial"]) == 1900
    assert cc.gpu_mb(["targeted"], ["yoloe_text", "clip"]) == 400 + 900 + 350


def test_files_cover_old_keys():
    assert GOLD_FILE_KEYS <= set(cc.DEFAULT_FILES)
    assert cc.DEFAULT_FILES["gdino"][0].startswith("models/grounding-dino-tiny")


def test_cues_shape():
    assert list(cc.CUES)[:3] == ["class", "free_arm", "semantic"]
    assert len(cc.CUES) == 18
    c = cc.CUES["size_veto"]
    assert c["roles"] == ("veto",) and c["needs"] == ["depth"] and c["detector"] == ("yoloe_text", "trained")
    assert cc.CUE_EN["free_arm"] == "wider candidates" and cc.CUE_EN["track"] == "track"


def test_every_reference_resolves():
    reg = cc.load_registry()
    for k, c in reg["cues"].items():
        for m in c.get("needs") or []:
            assert m in reg["models"] or m in ("ovd", "depth"), (k, m)
    for opt, rows in cc.option_cues().items():
        assert opt in cc.MODELS
        for r in rows:
            for m in r["models"]:
                assert m["key"] in reg["models"], (opt, r["key"], m["key"])


def test_catalog_carries_options_and_budget():
    cat = cc.catalog({})
    assert set(cat["options"]) == set(cc.MODELS)
    river = {r["key"]: [m["key"] for m in r["models"]] for r in cat["options"]["river"]}
    assert river["class"] == ["yoloe_text", "yoloe_free"] and river["distance"] == ["depth"]
    assert cat["gpu_mb_base"] == 400 and cat["gpu_mb_option"]["river"] == 2400
    tgt = {r["key"]: [m["key"] for m in r["models"]] for r in cat["options"]["targeted"]}
    assert tgt["ovd_pair_iou"] == ["yolo_world", "grounding_dino"]


@pytest.mark.parametrize("bad, msg", [
    ("options: {}\ncues: {a: {}}\nmodels: {m: {}}\n", "options"),
    ("options: {o: {cue_models: {c: [nope]}}}\ncues: {a: {needs: []}}\nmodels: {m: {}}\n", "nope"),
    ("options: {o: {}}\ncues: {a: {needs: [ghost]}}\nmodels: {m: {}}\n", "ghost"),
])
def test_bad_registry_is_refused(tmp_path, bad, msg):
    f = tmp_path / "reg.yaml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cc.load_registry(f)
    assert msg in str(e.value)
