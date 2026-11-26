"""vision/runners/rf_detr.py — RF-DETR runner (가중치 · rfdetr · CUDA 가 없으면 건너뛴다)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runners"))
import rf_detr  # noqa: E402

DS = Path(os.environ.get("DS_HOME") or HERE.parent)


def _need(variant):
    pytest.importorskip("rfdetr")
    torch = pytest.importorskip("torch")
    p = rf_detr.weight_path({"variant": variant})
    if not p.is_file():
        pytest.skip("가중치 없음: %s" % p)
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def test_unknown_variant_refused():
    with pytest.raises(SystemExit):
        rf_detr.weight_path({"variant": "huge"})


def test_weight_path_default_under_ds_home(monkeypatch):
    monkeypatch.setenv("DS_HOME", "/x")
    assert str(rf_detr.weight_path({"variant": "medium"})) == "/x/models/rf-detr/rf-detr-medium.pth"
    assert str(rf_detr.weight_path({"weight": "/a/b.pth"})) == "/a/b.pth"
    assert rf_detr.variant_of({"weight": "models/rf-detr/rf-detr-seg-medium.pt"}) == "seg-medium"
    assert rf_detr.variant_of({}) == "small"


def test_missing_weight_is_systemexit(tmp_path):
    pytest.importorskip("rfdetr")
    with pytest.raises(SystemExit):
        rf_detr.Runner({"variant": "small", "weight": str(tmp_path / "none.pth")}, "cpu")


def _frame():
    import glob
    import cv2
    fs = sorted(glob.glob(str(DS / "stream_data/pi12/2*/frames/*.jpg")))[:1]
    return cv2.imread(fs[0]) if fs else np.zeros((720, 1280, 3), np.uint8)


@pytest.mark.parametrize("variant", ["small", "seg-small"])
def test_runner_protocol(variant):
    dev = _need(variant)
    r = rf_detr.Runner({"variant": variant, "conf": 0.3}, dev)
    try:
        assert r.name == "rf_detr" and r.kind == "detect"
        bgr = _frame()
        info, dets = r(bgr)
        assert info["infer_ms"] > 0 and info["variant"] == variant
        h, w = bgr.shape[:2]
        for d in dets:
            assert set(d) >= {"cls", "name", "conf", "xyxy"}
            assert 0 <= d["cls"] < 80 and r.names[d["cls"]] == d["name"]
            assert d["conf"] >= 0.3
            x1, y1, x2, y2 = d["xyxy"]
            assert -1 <= x1 <= x2 <= w + 1 and -1 <= y1 <= y2 <= h + 1
        if variant.startswith("seg"):
            polys = [d["_poly"] for d in dets if "_poly" in d]
            assert polys or not dets
            for p, d in zip(polys, [d for d in dets if "_poly" in d]):  # 외곽선이 원래 화면 좌표로 되돌아왔나
                assert p.shape[1] == 2
                x1, y1, x2, y2 = d["xyxy"]
                assert x1 - 8 <= p[:, 0].min() and p[:, 0].max() <= x2 + 8
                assert y1 - 8 <= p[:, 1].min() and p[:, 1].max() <= y2 + 8
        assert [d["conf"] for d in dets] == sorted((d["conf"] for d in dets), reverse=True)
        # 빈 화면에서도 돈다
        r(np.zeros((480, 640, 3), np.uint8))
    finally:
        r.close()


def test_include_filters():
    dev = _need("small")
    r = rf_detr.Runner({"variant": "small", "conf": 0.05, "include": ["person"]}, dev)
    try:
        _, dets = r(_frame())
        assert all(d["name"] == "person" for d in dets)
    finally:
        r.close()
