"""위험물 검증 — 라벨 없이 도는 부분의 경계.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv/bin/python -m pytest test_hazard_verify.py -q
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "hazard_verify_under_test", Path(__file__).with_name("hazard_verify.py"))
hv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hv)


def stub(classes=("butane_canister",), **over):
    """인코더도 프로토타입도 없는 검증기 — **비활성이어야 한다.**"""
    spec = {"classes": list(classes), "models": [], "prototypes": "/nope.npz"}
    spec.update(over)
    return hv.HazardVerifier(spec, log=None)


def test_without_models_or_prototypes_it_stands_down_quietly():
    """가중치가 없는 기계에서 **세우기를 막지 않는다** — 검증만 빠진다(AI-C-05)."""
    v = stub()
    assert v.ready is False
    dets = [{"name": "butane_canister", "conf": 0.5, "xyxy": [0, 0, 20, 60]}]
    assert v(np.zeros((100, 100, 3), np.uint8), dets) == 0.0
    assert "verify" not in dets[0]          # 점수를 **지어내지 않는다**


def test_a_frame_with_no_hazard_costs_nothing():
    v = stub()
    v.ready = True                          # 준비됐다고 쳐도 후보가 없으면 0
    assert v(np.zeros((100, 100, 3), np.uint8),
             [{"name": "styrofoam_box", "conf": 0.9, "xyxy": [0, 0, 10, 10]}]) == 0.0


# ---------------------------------------------------------- 구조 서명
def _can(cap_bgr, body_bgr, w=60, h=200):
    """뚜껑 → 무채색 어깨 → 채도 몸통. 기준 사진에서 읽은 구조."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 8] = cap_bgr
    img[h // 8: h // 4] = (200, 200, 200)   # 어깨(무채색 밝음)
    img[h // 4:] = body_bgr
    return img


def test_the_structure_cue_prefers_a_can_over_a_plain_block():
    can = _can((200, 60, 40), (40, 40, 220))
    plain = np.full((200, 60, 3), (40, 40, 220), np.uint8)   # 몸통뿐, 뚜껑·어깨 없음
    assert hv.HazardVerifier._structure(can, "butane_canister") > \
           hv.HazardVerifier._structure(plain, "butane_canister")


def test_the_structure_cue_reads_a_lying_can_too():
    """강에 뜬 통은 서 있지 않다. 긴 축을 찾아 본다."""
    up = _can((200, 60, 40), (40, 40, 220))
    lying = cv2.rotate(up, cv2.ROTATE_90_CLOCKWISE)
    a = hv.HazardVerifier._structure(up, "butane_canister")
    b = hv.HazardVerifier._structure(lying, "butane_canister")
    assert abs(a - b) < 0.12


def test_wood_cue_wants_low_saturation():
    wood = np.full((120, 60, 3), (150, 190, 225), np.uint8)   # 옅은 나무색
    vivid = np.full((120, 60, 3), (30, 30, 240), np.uint8)
    assert hv.HazardVerifier._structure(wood, "wooden_box_mine") > \
           hv.HazardVerifier._structure(vivid, "wooden_box_mine")


def test_a_crop_too_small_is_not_scored():
    """**0 점이 아니라 '재지 않음'** 이다. 둘을 섞으면 작은 것이 전부 나쁜 점수가 된다."""
    assert hv.HazardVerifier._structure(np.zeros((4, 4, 3), np.uint8), "wooden_box_mine") is None
    assert hv.HazardVerifier._structure(None, "butane_canister") is None


def test_an_unknown_class_gets_no_structure_score():
    assert hv.HazardVerifier._structure(np.zeros((80, 40, 3), np.uint8), "styrofoam_box") is None
