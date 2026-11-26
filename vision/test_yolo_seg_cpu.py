"""YOLO 분할 CPU 실행기 (yolo26n_seg_cpu · yoloe26n_vocab_cpu) — 가중치가 없으면 실제 추론 시험은 건너뛴다"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from runners import yolo_seg_cpu as ysc  # noqa: E402
from runners import load_runner, runner_models  # noqa: E402

ASSET = None
try:
    import ultralytics
    ASSET = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
except ImportError:
    pass


def _have(*rel):
    return all(Path(ysc.resolve_path(r)).exists() for r in rel)


def test_letterbox_keeps_ratio_and_pads():
    img = np.zeros((360, 640, 3), np.uint8)
    out, r, (px, py) = ysc.letterbox(img, 320)
    assert out.shape == (320, 320, 3) and r == 0.5 and px == 0 and py == 70
    assert out[0, 0, 0] == 114 and out[100, 100, 0] == 0


def test_nms_raw_drops_overlap_and_low_score():
    nm = 2
    # 앵커 3 개: 둘은 같은 자리(점수 0.9 · 0.8) · 하나는 낮은 점수. 클래스 2 개
    cols = [[50, 50, 20, 20, 0.9, 0.1, 1, 1],
            [51, 50, 20, 20, 0.8, 0.1, 1, 1],
            [150, 150, 20, 20, 0.05, 0.1, 1, 1]]
    pred = np.array(cols, np.float32).T                  # (4+nc+nm, A)
    rows = ysc.nms_raw(pred, 0.25, 0.5, 10, nm)
    assert rows.shape == (1, 6 + nm)
    assert rows[0, :4].tolist() == [40, 40, 60, 60] and abs(rows[0, 4] - 0.9) < 1e-6 and rows[0, 5] == 0


def test_names_from_meta_and_yaml(tmp_path):
    assert ysc._names_from_meta("{0: 'person', 1: 'door'}") == ["person", "door"]
    assert ysc._names_from_meta("") is None
    p = tmp_path / "metadata.yaml"
    p.write_text("task: segment\nnames:\n  0: person\n  1: 'traffic cone'\nargs: {}\n", encoding="utf-8")
    assert ysc._names_from_yaml(p) == ["person", "traffic cone"]


def test_registry_entries_point_here():
    reg = runner_models()
    for key, cls in (("yolo26n_seg_cpu", "Yolo26nSegCpuRunner"), ("yoloe26n_vocab_cpu", "Yoloe26nVocabCpuRunner")):
        spec = reg[key]
        assert spec["runner"] == "yolo_seg_cpu:%s" % cls
        assert {"cpu_arm64", "cpu_x86"} <= set(spec["runs_on"]) and spec["size"] == "small"
        assert getattr(ysc, cls).name == key
    assert "open_vocab" in reg["yoloe26n_vocab_cpu"]["provides"]


@pytest.mark.parametrize("key,imgsz", [("yolo26n_seg_cpu", 320), ("yoloe26n_vocab_cpu", 320)])
def test_runs_on_bus(key, imgsz):
    spec = runner_models()[key]
    if not _have(*spec["files"]) or ASSET is None or not ASSET.exists():
        pytest.skip("가중치 또는 시험 그림 없음")
    import cv2
    r = load_runner(key, spec, "cpu", {"imgsz": imgsz, "threads": 2})
    info, dets = r(cv2.imread(str(ASSET)))
    assert info["infer_ms"] > 0 and info["imgsz"] == imgsz
    names = [d["name"] for d in dets]
    assert names.count("person") >= 3 and "bus" in names
    d = dets[0]
    assert set(d) >= {"cls", "name", "conf", "xyxy"} and 0 < d["conf"] <= 1
    assert d["_poly"].shape[1] == 2 and len(d["polygon"]) >= 3
    x1, y1, x2, y2 = d["xyxy"]
    px, py = d["_poly"][:, 0], d["_poly"][:, 1]
    assert px.min() >= x1 - 15 and px.max() <= x2 + 15 and py.min() >= y1 - 15 and py.max() <= y2 + 15
