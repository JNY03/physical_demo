# -*- coding: utf-8 -*-
"""vision/runners/qwen3_vl.py 시험 — 답 글 → 상자 풀이는 늘 돌고, 실제 모델은 가중치 · GPU 가 있을 때만 돈다."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "runners"))
import qwen3_vl as Q  # noqa: E402  (torch 는 Runner 를 만들 때만 import 한다)


def test_parse_json_fence_rel1000():
    txt = '```json\n[\n {"bbox_2d": [100, 200, 300, 400], "label": "chair"},\n {"bbox_2d": [500, 0, 1000, 1000], "label": "person"}\n]\n```'
    d = Q.parse_boxes(txt, 2000, 1000)
    assert [x["name"] for x in d] == ["chair", "person"]
    assert d[0]["xyxy"] == [200.0, 200.0, 600.0, 400.0]
    assert d[1]["xyxy"] == [1000.0, 0.0, 2000.0, 1000.0]
    assert all(x["cls"] == -1 and x["conf"] == 1.0 for x in d)


def test_parse_truncated_answer_keeps_complete_items():
    # max_new_tokens 에서 잘린 답 — 끝난 항목만 살린다
    txt = '```json\n[\n\t{"bbox_2d": [255, 107, 452, 552], "label": "chair"},\n\t{"bbox_2d": [488, 173, 600'
    d = Q.parse_boxes(txt, 1000, 1000)
    assert len(d) == 1 and d[0]["name"] == "chair" and d[0]["xyxy"] == [255.0, 107.0, 452.0, 552.0]


def test_parse_variants():
    # 울타리 없는 한 객체 · 뒤집힌 좌표 · 화면 밖 좌표 · 넓이 0 · 중복
    assert Q.parse_boxes('{"bbox_2d": [400, 400, 100, 100], "label": "door"}', 1000, 1000)[0]["xyxy"] == [100.0, 100.0, 400.0, 400.0]
    assert Q.parse_boxes('[{"bbox": [-50, 0, 1200, 500], "name": "person"}]', 100, 100)[0]["xyxy"] == [0.0, 0.0, 100.0, 50.0]
    assert Q.parse_boxes('[{"bbox_2d": [10, 10, 10, 50], "label": "x"}]', 1000, 1000) == []
    two = '[{"bbox_2d": [1,2,300,400], "label": "chair"}, {"bbox_2d": [1,2,300,400], "label": "chair"}]'
    assert len(Q.parse_boxes(two, 1000, 1000)) == 1
    # 이름이 바깥 키 · 상자가 맨 리스트
    d = Q.parse_boxes('{"chair": [[0, 0, 500, 500]]}', 1000, 1000)
    assert d and d[0]["xyxy"] == [0.0, 0.0, 500.0, 500.0]


def test_parse_abs_mode_and_garbage():
    d = Q.parse_boxes('[{"bbox_2d": [0, 0, 320, 176], "label": "chair"}]', 1280, 704, mode="abs", in_wh=(640, 352))
    assert d[0]["xyxy"] == [0.0, 0.0, 640.0, 352.0]
    assert Q.parse_boxes("상자를 찾지 못했습니다.", 640, 480) == []
    assert Q.parse_boxes("", 640, 480) == []
    # JSON 이 깨진 답 — 숫자 넷을 줄에서 줍는다
    d = Q.parse_boxes('chair: (100, 100, 200, 300)\nperson 500,500,900,900', 1000, 1000)
    assert [x["xyxy"] for x in d] == [[100.0, 100.0, 200.0, 300.0], [500.0, 500.0, 900.0, 900.0]]


def test_prompts_are_korean_templates():
    assert {"describe", "situation", "explain", "ground"} <= set(Q.PROMPTS)
    g = Q.PROMPTS["ground"].format(names="chair, person")
    assert "chair, person" in g and "bbox_2d" in g and Q._GROUND_HINT.search(g)
    assert not Q._GROUND_HINT.search(Q.PROMPTS["describe"])
    assert not Q._GROUND_HINT.search("앞에 박스가 있어?")      # 글 질문은 상자 풀이로 가지 않는다


def _weights():
    home = Path(os.environ.get("DS_HOME") or Path(__file__).resolve().parents[1])
    return home / "models" / "qwen3-vl-2b-instruct"


@pytest.mark.skipif(not (_weights() / "model.safetensors").is_file(), reason="Qwen3-VL 가중치 없음")
def test_runner_real_frame():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU 없음")
    import numpy as np
    img = np.full((480, 640, 3), 200, np.uint8)
    img[100:400, 200:440] = (30, 30, 30)                   # 회색 바탕에 검은 사각형
    r = Q.Runner({"max_new_tokens": 32}, "cuda:0")
    try:
        assert r.kind == "vlm" and r.name == "qwen3_vl"
        txt = r.describe(img)
        assert isinstance(txt, str) and txt
        info, dets = r.ground(img, ["black rectangle"])
        assert "text" in info
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
        assert r.describe(img) == txt                       # 결정적 디코딩
    finally:
        r.close()
    assert torch.cuda.memory_allocated() < 200 * 2**20


def test_parse_drops_full_frame_hallucination():
    txt = '[{"bbox_2d": [265, 0, 1000, 1000], "label": "chair"}, {"bbox_2d": [0, 0, 1000, 1000], "label": "door"}]'
    assert len(Q.parse_boxes(txt, 1280, 720)) == 2
    d = Q.parse_boxes(txt, 1280, 720, max_area=0.95)
    assert [x["name"] for x in d] == ["chair"]
