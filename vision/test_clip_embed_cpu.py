"""CLIP 계열 임베딩 CPU 실행기 (mobileclip_s0_cpu) — 가중치가 없으면 추론 시험은 건너뛴다"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runners import clip_embed_cpu as cec  # noqa: E402
from runners import load_runner, runner_models  # noqa: E402


def test_registry_entry():
    spec = runner_models()["mobileclip_s0_cpu"]
    assert spec["provides"] == ["embed"] and spec["size"] == "small"
    assert {"cpu_arm64", "cpu_x86"} <= set(spec["runs_on"])
    assert cec.MobileClipS0EmbedRunner.name == "mobileclip_s0_cpu"


def test_embeds_are_normalised_and_similar_crops_match():
    spec = runner_models()["mobileclip_s0_cpu"]
    if not all(Path(cec.resolve_path(f)).exists() for f in spec["files"]):
        pytest.skip("가중치 없음")
    rng = np.random.default_rng(0)
    img = np.zeros((240, 480, 3), np.uint8)
    img[:, :240] = (30, 30, 220)                                 # 왼쪽 빨강 판
    img[:, 240:] = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)   # 오른쪽 잡음
    r = load_runner("mobileclip_s0_cpu", spec, "cpu", {"threads": 2})
    info, e = r(img, [[10, 10, 110, 110], [120, 120, 230, 230], [250, 10, 470, 230], [5, 5, 5.5, 5.5]])
    assert e.shape == (4, info["dim"]) and info["skipped"] == 1
    assert np.allclose(np.linalg.norm(e[:3], axis=1), 1.0, atol=1e-4) and not e[3].any()
    assert e[0] @ e[1] > e[0] @ e[2]                              # 빨강끼리가 빨강-잡음보다 닮았다
    _, whole = r(img, [])
    assert whole.shape == (1, info["dim"])
