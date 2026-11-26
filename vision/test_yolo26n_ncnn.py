"""YOLO26n 기기 실행기 (vision/runners/yolo26n_ncnn.py) — 모델이 없으면 건너뛴다 (2026-11-26).

  DS_HOME=… PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vision/test_yolo26n_ncnn.py -q

NCNN(one-to-many 머리 + NMS) 과 ONNX end2end 두 길이 같은 프레임에서 비슷한 답을 내는지 본다.
"""
import ast
import glob
import os
from pathlib import Path

import numpy as np
import pytest

HOME = Path(os.environ.get("DS_HOME") or Path(__file__).resolve().parent.parent)
D = HOME / "models" / "yolo26n_ncnn"
SRC = Path(__file__).resolve().parent / "runners" / "yolo26n_ncnn.py"

from runners import yolo26n_ncnn as yn  # noqa: E402


def test_standalone_imports_only():
    """기기 컨테이너에 혼자 들어간다 — 저장소 모듈 · torch · ultralytics 를 맨 위에서 import 하지 않는다."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert top <= {"__future__", "os", "time", "pathlib", "numpy"}, top


def test_letterbox_roundtrip():
    x, r, left, top = yn.letterbox(np.zeros((400, 464, 3), np.uint8), 320)
    assert x.shape == (3, 320, 320) and x.dtype == np.float32
    assert abs(r - 320 / 464) < 1e-6 and left == 0 and top > 0


def _frames(n=4):
    import cv2
    fs = sorted(glob.glob(str(HOME / "stream_data/pi12/**/frames/*.jpg"), recursive=True))[:n]
    ims = [cv2.imread(f) for f in fs]
    ims = [i for i in ims if i is not None]
    if not ims:
        pytest.skip("실 프레임 없음 (stream_data/pi12)")
    return ims


@pytest.mark.parametrize("model", ["yolo26n_ncnn_320", "yolo26n_ncnn_640", "yolo26n_320.onnx", "yolo26n_640.onnx"])
def test_runner(model):
    p = D / model
    if not p.exists():
        pytest.skip("모델 없음: %s" % p)
    if model.endswith(".onnx"):
        pytest.importorskip("onnxruntime")
    else:
        pytest.importorskip("ncnn")
    pytest.importorskip("cv2")
    r = yn.Runner({"model": str(p), "threads": 4}, "cpu")
    assert r.kind == "detect" and r.provides == ("detect",) and r.imgsz in (320, 640)
    for im in _frames():
        info, dets = r(im)
        assert info["backend"] == ("onnx" if model.endswith(".onnx") else "ncnn")
        h, w = im.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h and d["conf"] >= 0.25
            assert d["name"] == r.names[d["cls"]]
    r.close()


def test_ncnn_and_onnx_agree():
    a, b = D / "yolo26n_ncnn_640", D / "yolo26n_640.onnx"
    if not (a.exists() and b.exists()):
        pytest.skip("모델 없음")
    pytest.importorskip("ncnn")
    pytest.importorskip("onnxruntime")
    ra, rb = yn.Runner({"model": str(a)}), yn.Runner({"model": str(b)})
    na = nb = 0
    for im in _frames(6):
        na += len(ra(im)[1])
        nb += len(rb(im)[1])
    assert abs(na - nb) <= max(3, 0.5 * max(na, nb))
