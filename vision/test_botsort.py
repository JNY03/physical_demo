"""BoT-SORT 어댑터 (vision/runners/botsort.py) — 가중치 없이 돈다 (ReID 시험만 파일이 있어야) (2026-11-26).

  DS_HOME=… PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vision/test_botsort.py -q
"""
import os
from pathlib import Path

import numpy as np
import pytest

HOME = Path(os.environ.get("DS_HOME") or Path(__file__).resolve().parent.parent)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

pytest.importorskip("ultralytics")
pytest.importorskip("lap")
from runners import botsort  # noqa: E402


def _scene(i, w=640, h=480):
    """오른쪽으로 움직이는 상자 둘 + 배경 무늬 (GMC 가 볼 것)."""
    rng = np.random.default_rng(0)
    im = (rng.integers(0, 255, (h, w, 3))).astype(np.uint8)
    dets = [{"cls": 0, "name": "person", "conf": 0.9, "xyxy": [50 + 5 * i, 100, 120 + 5 * i, 300]},
            {"cls": 2, "name": "car", "conf": 0.8, "xyxy": [400, 200 + 2 * i, 560, 300 + 2 * i]}]
    return im, dets


@pytest.mark.parametrize("gmc", ["sparseOptFlow", "none"])
def test_ids_stable(gmc):
    tr = botsort.Runner({"gmc_method": gmc}, "cpu")
    assert tr.kind == "track" and tr.provides == ("track",)
    seen = []
    for i in range(10):
        im, dets = _scene(i)
        info, out = tr(im, dets=dets, stream="a")
        assert len(out) == 2 and info["tracker"] == "botsort"
        seen.append(tuple(d.get("track_id") for d in out))
    assert seen[-1][0] is not None and seen[-1][0] != seen[-1][1]
    assert len(set(seen[3:])) == 1                         # 번호가 바뀌지 않는다
    assert out[0]["xyxy"] == _scene(9)[1][0]["xyxy"]        # 박스는 탐지기 것 그대로 (keep_boxes: detection)


def test_streams_are_separate():
    tr = botsort.Runner({}, "cpu")
    for i in range(5):
        im, dets = _scene(i)
        a = tr(im, dets=dets, stream="a")[1]
        b = tr(im, dets=dets[:1], stream="b")[1]
    assert len(a) == 2 and len(b) == 1
    assert set(tr._streams) == {"a", "b"}


def test_reset_gap_restarts():
    tr = botsort.Runner({"reset_gap_s": 1.0}, "cpu")
    for i in range(5):
        im, dets = _scene(i)
        tr.update(dets, float(i) * 0.1, im, stream="a")
    first = tr._streams["a"][0]
    tr.update(_scene(5)[1], 10.0, _scene(5)[0], stream="a")
    assert tr._streams["a"][0] is not first


def test_untracked_keep_and_drop():
    im, _ = _scene(0)
    weak = [{"cls": 0, "name": "x", "conf": 0.12, "xyxy": [10, 10, 50, 50]}]     # new_track_thresh 아래
    assert botsort.Runner({}, "cpu").update(weak, None, im)[0].get("track_id") is None
    assert botsort.Runner({"on_untracked": "drop"}, "cpu").update(weak, None, im) == []


def test_rejects_unknown_and_remote_reid():
    with pytest.raises(ValueError, match="모르는"):
        botsort.Runner({"tarck_buffer": 3}, "cpu")
    with pytest.raises(FileNotFoundError):
        botsort.Runner({"with_reid": True, "reid_model": "yolo26n-reid.onnx"}, "cpu")   # 이름만 → 받으러 나간다


def test_reid_local_file():
    f = HOME / "models" / "yolo26n-reid.onnx"
    if not f.is_file():
        pytest.skip("ReID 가중치 없음")
    pytest.importorskip("onnxruntime")
    tr = botsort.Runner({"with_reid": True, "reid_model": str(f)}, "cpu")
    for i in range(4):
        im, dets = _scene(i)
        out = tr(im, dets=dets, stream="a")[1]
    assert out[0].get("track_id") is not None
