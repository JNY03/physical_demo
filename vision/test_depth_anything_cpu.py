"""Depth Anything 상대 깊이 CPU 실행기 (da3_small_cpu · dav2_small_cpu) — 가중치가 없으면 추론 시험은 건너뛴다"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runners import depth_anything_cpu as dac  # noqa: E402
from runners import load_runner, runner_models  # noqa: E402


def test_k_from_fov():
    K = dac.k_from_fov(640, 480, 90.0)
    assert abs(K[0, 0] - 320.0) < 1e-6 and K[0, 2] == 320.0 and K[1, 2] == 240.0


def test_registry_entries_are_relative_cpu_depth():
    reg = runner_models()
    for key in ("da3_small_cpu", "dav2_small_cpu"):
        spec = reg[key]
        assert spec["provides"] == ["depth_relative"] and spec["size"] == "small"
        assert {"cpu_arm64", "cpu_x86"} <= set(spec["runs_on"])


@pytest.mark.parametrize("key", ["da3_small_cpu", "dav2_small_cpu"])
def test_depth_is_relative_and_far_is_bigger(key):
    spec = runner_models()[key]
    if not all(Path(dac.resolve_path(f)).exists() for f in spec["files"]):
        pytest.skip("가중치 없음")
    # 위는 멀고(작은 무늬) 아래는 가까운(큰 무늬) 바닥 — 아래쪽이 더 가깝게 나와야 한다
    h, w = 240, 320
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        period = 4 + y // 6
        img[y, :, :] = ((np.arange(w) // period) % 2 * 200 + 30)[:, None]
    img[: h // 3] = (235, 206, 135)                       # 하늘
    r = load_runner(key, spec, "cpu", {"input": 196, "threads": 2})
    info, depth, K = r(img)
    assert info["relative"] is True and info["units"] == "relative"
    assert depth.shape == (h, w) and depth.dtype == np.float32
    assert K.shape == (3, 3) and K[0, 0] > 0
    assert abs(float(np.median(depth)) - 1.0) < 0.35
    assert float(np.median(depth[-30:])) < float(np.median(depth[h // 3: h // 3 + 30]))
    _, _, K2 = r(img, fov_x=90.0)
    assert abs(K2[0, 0] - w / 2) < 1e-6
