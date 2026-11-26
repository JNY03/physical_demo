"""YOLOE-26 실행기 (vision/runners/yoloe26.py) — 가중치가 없으면 건너뛴다 (2026-11-26).

  DS_HOME=… PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vision/test_yoloe26.py -q

GPU 가 있으면 cuda:0, 없으면 cpu 로 돈다. 한 장(실 프레임 있으면 그것, 없으면 합성)으로 규약만 본다.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pytest

HOME = Path(os.environ.get("DS_HOME") or Path(__file__).resolve().parent.parent)
M = HOME / "models"
os.environ.setdefault("YOLO_AUTOINSTALL", "False")       # 빠진 패키지를 몰래 설치하지 않는다
os.environ.setdefault("YOLO_OFFLINE", "1")

pytest.importorskip("ultralytics")
from runners import yoloe26  # noqa: E402


def _device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _frame():
    fs = sorted(glob.glob(str(HOME / "stream_data/pi12/**/frames/*.jpg"), recursive=True))[:1]
    if fs:
        import cv2
        im = cv2.imread(fs[0])
        if im is not None:
            return im
    im = np.full((480, 640, 3), 90, np.uint8)
    im[100:400, 200:300] = 200
    return im


def _need(*names):
    miss = [n for n in names if not (M / n).is_file()]
    if miss:
        pytest.skip("가중치 없음: %s" % ", ".join(miss))


def _check(info, dets):
    assert "infer_ms" in info
    for d in dets:
        assert {"cls", "name", "conf", "xyxy"} <= set(d)
        assert len(d["xyxy"]) == 4 and 0 <= d["conf"] <= 1


@pytest.mark.parametrize("w", ["yoloe-26s-seg-pf.pt", "yoloe-26l-seg-pf.pt"])
def test_free(w):
    _need(w)
    r = yoloe26.Runner({"weight": str(M / w), "mode": "free"}, _device())
    assert r.kind == "detect" and set(r.provides) == {"detect", "segment", "open_vocab"}
    assert len(r.model.names) > 4000                   # 내장 어휘
    _check(*r(_frame()))
    r.close()


@pytest.mark.parametrize("w", ["yoloe-26s-seg.pt", "yoloe-26l-seg.pt"])
def test_text(w):
    _need(w, "mobileclip2_b.ts")
    classes = ["person", "door", "chair"]
    r = yoloe26.Runner({"weight": str(M / w), "mode": "text", "classes": classes,
                        "text_encoder_dir": str(M)}, _device())
    assert set(r.provides) == {"detect", "segment", "open_vocab"} and r.encoder == "mobileclip2_b.ts"
    info, dets = r(_frame())
    _check(info, dets)
    assert all(d["name"] in classes for d in dets)
    r.close()


def test_text_rejects_missing_encoder(tmp_path):
    _need("yoloe-26s-seg.pt")
    with pytest.raises(FileNotFoundError, match="mobileclip2_b.ts"):
        yoloe26.TextRunner({"weight": str(M / "yoloe-26s-seg.pt"), "classes": ["door"],
                            "text_encoder_dir": str(tmp_path)}, "cpu")


def test_text_rejects_pf_weight():
    _need("yoloe-26s-seg-pf.pt", "mobileclip2_b.ts")
    with pytest.raises(ValueError, match="-pf"):
        yoloe26.TextRunner({"weight": str(M / "yoloe-26s-seg-pf.pt"), "classes": ["door"],
                            "text_encoder_dir": str(M)}, "cpu")
