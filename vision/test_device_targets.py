"""기기마다 다른 대상 (2026-11-27) — 한 번 찾고 그 기기의 대상만 남긴다 (vision_infer._detect_one)."""
import numpy as np

import vision_infer as vi


class FakeTargeted:
    def __call__(self, bgr, roi_crop=None):
        return {}, [{"name": "person", "conf": 0.9, "xyxy": [0, 0, 10, 10]},
                    {"name": "chair", "conf": 0.8, "xyxy": [20, 20, 40, 40]}]


def _run(cfg, device):
    cfg.setdefault("stabilize", {"enabled": False, "targeted": {"enabled": False}})
    out = vi._detect_one("targeted", np.zeros((64, 64, 3), np.uint8), {"run": "t"}, {}, device,
                         {"targeted": FakeTargeted()}, cfg, {}, None, True)
    return out


def _names(out):
    for x in out if isinstance(out, (list, tuple)) else [out]:
        if isinstance(x, list) and x and isinstance(x[0], dict) and "name" in x[0]:
            return sorted(d["name"] for d in x)
        if isinstance(x, (list, tuple)):
            n = _names(x)
            if n is not None:
                return n
    return None


def test_device_targets_filter_only_that_device():
    cfg = {"models": {}, "_targets_dev": {"pi12": ["chair"]}}
    assert _names(_run(dict(cfg), "pi12")) == ["chair"]
    assert _names(_run(dict(cfg), "robot1")) == ["chair", "person"]      # 적지 않은 기기는 전부
