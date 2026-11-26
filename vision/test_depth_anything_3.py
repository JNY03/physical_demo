"""Depth Anything 3 runner · 거리 단계 연결.

  DS_HOME=/home/khw67/data_stream python -m pytest vision/test_depth_anything_3.py -q

가중치(models/da3-*)나 외부 코드(external/Depth-Anything-3)가 없으면 GPU 시험은 건너뛴다.
가벼운 시험(입력 크기 · DA3 판별 · kind 고르기)은 torch 없이 돈다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


da3 = _load("da3_under_test", HERE / "runners" / "da3_metric.py")
dt = _load("depth_track_da3_test", HERE / "depth_track.py")

METRIC_W = "models/da3-metric-large"
SMALL_W = "models/da3-small"


def _have(*paths):
    return all((da3.ds_path(p) / "model.safetensors").is_file() for p in paths) and \
        (da3.ds_path("external/Depth-Anything-3") / "src" / "depth_anything_3").is_dir()


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------- 가벼운 것

def test_proc_size_is_patch_multiple_and_long_side():
    for h, w in [(400, 464), (720, 1280), (1080, 1920), (1920, 3840)]:
        nh, nw = da3.proc_size(h, w, 504)
        assert nh % 14 == 0 and nw % 14 == 0
        assert abs(max(nh, nw) - 504) <= 7


def test_is_da3_reads_config(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_name": "da3metric-large", "config": {}}))
    assert da3.is_da3(d) and da3.model_name(d) == "da3metric-large"
    (d / "config.json").write_text(json.dumps({"model_type": "unidepth"}))
    assert not da3.is_da3(d)
    assert not da3.is_da3(tmp_path / "없음")


def test_kind_switches_only_for_da3_folder(tmp_path):
    d = tmp_path / "da3"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_name": "da3metric-large"}))
    # cue_model_choices.depth 는 model_id 만 바꾼다 — kind 가 unidepth 여도 DA3 로 돈다
    assert dt._kind({"kind": "unidepth", "model_id": str(d)}) == "depth_anything_3"
    # UniDepth 기본은 그대로
    assert dt._kind({"kind": "unidepth", "model_id": "lpiccinelli/unidepth-v2-vitb14"}) == "unidepth"
    assert dt._kind({"kind": "unidepth", "model_id": str(tmp_path / "unidepth")}) == "unidepth"
    assert dt._kind({"kind": "hf_metric", "model_id": str(d)}) == "hf_metric"
    assert "depth_anything_3" in dt.KINDS


# ---------------------------------------------------------------- GPU (가중치가 있을 때만)

needs_metric = pytest.mark.skipif(not (_have(METRIC_W, SMALL_W) and _cuda()),
                                  reason="DA3 가중치/외부 코드/GPU 없음")
needs_small = pytest.mark.skipif(not (_have(SMALL_W) and _cuda()), reason="DA3-Small 가중치/GPU 없음")


def _scene(h=360, w=640):
    """바닥 기울기가 있는 단순한 장면 (위 = 멀다)."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (60 + 150 * y).astype(np.uint8)[..., None]
    img[h // 3: h // 2, w // 3: w // 2] = (30, 30, 200)
    return img


@needs_metric
def test_metric_runner_contract():
    r = da3.Runner({"model_path": METRIC_W}, "cuda:0")
    img = _scene()
    info, d, K = r(img)
    assert d.shape == img.shape[:2] and d.dtype == np.float32
    assert info["space"] == "metric" and info["focal_source"] == "da3_small"
    assert K.shape == (3, 3) and K[0, 0] > 0
    assert np.isfinite(d).all() and (d >= 0).all() and (d > 0).mean() > 0.5
    # 준 화각이 먼저다. fx 가 두 배면 미터도 두 배 (metric = fx·net/300)
    i1, d1, K1 = r(img, fov_x=90)
    i2, d2, K2 = r(img, fov_x=2 * np.degrees(np.arctan(0.5)))     # fx = w  (90° 의 두 배)
    assert i1["focal_source"] == "given"
    m = (d1 > 0) & (d2 > 0)
    assert np.median(d2[m] / d1[m]) == pytest.approx(2.0, rel=0.02)


@needs_metric
def test_metric_without_focal_refuses():
    r = da3.Runner({"model_path": METRIC_W, "focal_model": False}, "cuda:0")
    with pytest.raises(ValueError):
        r(_scene())
    info, _d, _K = r(_scene(), fov_x=70)
    assert info["focal_source"] == "given"


@needs_small
def test_small_is_relative_with_intrinsics():
    r = da3.Runner({"model_path": SMALL_W}, "cuda:0")
    info, d, K = r(_scene())
    assert info["space"] == "relative" and info["focal_source"] == "self" and K[0, 0] > 0


@needs_metric
def test_tracker_uses_da3_and_gives_real_mm():
    t = dt.DepthTracker({"enabled": True, "kind": "unidepth", "model_id": str(da3.ds_path(METRIC_W)),
                         "period": 1}, log=None)
    assert t.ready and t.space == dt.METRIC, t.why
    dets = [{"name": "box", "xyxy": [213, 120, 320, 180]}]
    ms = t(_scene(), dets)
    assert ms > 0 and dets[0]["dist"]["source"] == "measured" and dets[0]["dist"]["value"] > 0
    assert "real_mm" in dets[0]


@needs_small
def test_tracker_refuses_relative_weights():
    t = dt.DepthTracker({"enabled": True, "kind": "depth_anything_3", "model_id": str(da3.ds_path(SMALL_W))},
                        log=None)
    assert not t.ready and "상대" in (t.why or "")
