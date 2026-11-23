"""demo_box — Box_Detection 원본 수신기(mqtt_box_receiver.BoxDetector)와 **같은 검출 · 관문 · 거리**가 나오는가.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python -m pytest test_demo_box.py -q

원본 저장소(/home/khw67/Box_Detection)와 GPU 가 없으면 건너뛴다. 원본의 on_frame 은 try_box/ 에 파일을
쓰므로 부르지 않고, 같은 순서(crop → predict(conf=CANDIDATE_CONF) → distance → gates)를 원본 메서드로 밟는다.
"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import pytest

BOX = Path("/home/khw67/Box_Detection")
HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(not (BOX / "demo/test/mqtt_box_receiver.py").exists(),
                                reason="Box_Detection 저장소 없음")


def _original():
    sys.path.insert(0, str(BOX / "demo/test"))
    spec = importlib.util.spec_from_file_location("mqtt_box_receiver", BOX / "demo/test/mqtt_box_receiver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _frames():
    out = sorted(glob.glob(str(BOX / "box_meter/*.jpg")))
    for d in ("box_2m", "box_2.5m", "box_3m", "sidebox_2m"):
        fs = sorted(glob.glob(str(BOX / d / "*/frames/*.jpg")) + glob.glob(str(BOX / d / "frames/*.jpg")))
        out += fs[::max(1, len(fs) // 6)][:6]
    return out


def test_same_as_original():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU 없음")
    orig = _original()
    det = orig.BoxDetector(orig.DEFAULT_RUN, orig.DEFAULT_CONF)
    target = json.loads((HERE / "assets/targets/demo_box.json").read_text(encoding="utf-8"))
    from demo_box import TrainedBoxArm
    arm = TrainedBoxArm(target, "cuda:0")
    frames = _frames()
    assert len(frames) >= 14
    x0, y0, x1, y1 = orig.BORDER_CROP_BOX
    n_det = 0
    for f in frames:
        img = cv2.imread(f)
        assert img.shape[:2] == (400, 464)
        crop = img[y0:y1, x0:x1]
        res = det.model.predict(crop, imgsz=det.imgsz, conf=orig.CANDIDATE_CONF, verbose=False)[0]
        want = []
        for b in sorted(res.boxes, key=lambda b: -float(b.conf[0])):
            box = [float(v) for v in b.xyxy[0].tolist()]
            score = float(b.conf[0])
            if score < det.conf:
                continue
            d = {"score": score}
            d.update(det.distance(box))
            gates = det.gates(box, score, d)
            failed = sorted(n for n, g in gates.items() if g.get("required") and not g["passed"])
            want.append((box, score, d["distance_cm"], failed, gates))
        got_ok, _rej, got_bad, _ms = arm(img)
        got = [(g, []) for g in got_ok] + [(b, sorted(b["failed_gates"])) for b in got_bad]
        assert len(got) == len(want), f
        for (wbox, wscore, wdist, wfail, wgates), (g, gfail) in zip(want, sorted(
                got, key=lambda t: -(t[0].get("conf") if "conf" in t[0] else t[0]["score"]))):
            gbox = g.get("xyxy") or g["bbox"]
            assert abs(gbox[0] - (wbox[0] + x0)) < 0.2 and abs(gbox[3] - (wbox[3] + y0)) < 0.2, f
            assert gfail == wfail, f
            if "box_distance" in g:
                assert g["box_distance"]["distance_cm"] == wdist, f
                for name in ("ground_gate", "shape_gate", "height_gate"):
                    assert g["box_gates"][name]["passed"] == wgates[name]["passed"], (f, name)
                    assert g["box_gates"][name]["words"] == wgates[name]["words"], (f, name)
                n_det += 1
    assert n_det >= 10          # box_meter 14장은 원본에서 전부 검출됐다(학습보고서 v9)


def test_other_camera_defers_ground():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU 없음")
    import numpy as np
    target = json.loads((HERE / "assets/targets/demo_box.json").read_text(encoding="utf-8"))
    from demo_box import TrainedBoxArm
    arm = TrainedBoxArm(target, "cuda:0")
    img = cv2.imread(sorted(glob.glob(str(BOX / "box_meter/*.jpg")))[0])
    big = cv2.resize(img, (928, 800))             # Go1 이 아닌 크기
    ok, _r, _b, _ms = arm(big)
    for d in ok:
        assert d["camera_profile"] == "other"
        assert d["box_distance"]["distance_cm"] is None
        assert d["box_gates"]["ground_gate"]["applicable"] is False
        assert d["box_gates"]["ground_gate"]["passed"] is True
    assert isinstance(np.zeros(1), np.ndarray)
