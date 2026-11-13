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


def test_without_models_or_prototypes_only_the_learned_half_stands_down():
    """가중치가 없는 기계에서 **세우기를 막지 않고**, 쓸 수 있는 근거는 계속 쓴다.

    예전에는 준비 판정이 하나여서 프로토타입이 없으면 **프로토타입이 필요 없는**
    구조 단서까지 같이 꺼졌다. 의존하지 않는 것이 의존하는 것 때문에 죽던 자리다.
    """
    v = stub()
    assert v.learned is False               # 배운 쪽(match)은 못 선다
    assert v.ready is True                  # 안 배운 쪽(structure)은 선다
    dets = [{"name": "butane_canister", "conf": 0.5, "xyxy": [0, 0, 20, 60]}]
    v(np.zeros((100, 100, 3), np.uint8), dets)
    per = (dets[0].get("verify") or {}).get("per_model", {})
    assert "clip_b32" not in per            # 기준이 없으면 match 를 **지어내지 않는다**


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


def test_the_mine_gets_no_structure_cue_because_none_was_found():
    """**갈리지 않는 수를 근거로 내지 않는다.**

    전에는 여기서 목질색을 셌는데 `material` 이 같은 것을 보고 있어 같은 사실이
    평균에 두 번 들어갔다. 색을 안 보는 형상 단서로 바꿔 보려고 명암 기울기 셋을
    쟀고 **셋 다 갈리지 않았다** — 하늘이 진짜 상자보다 '네모났다'(0.515 vs 0.440).
    그래서 내지 않는다. 증거처럼 보이는데 증거가 아닌 수가 중복보다 나쁘다.
    """
    wood = np.full((120, 60, 3), (150, 190, 225), np.uint8)
    vivid = np.full((120, 60, 3), (30, 30, 240), np.uint8)
    assert hv.HazardVerifier._structure(wood, "wooden_box_mine") is None
    assert hv.HazardVerifier._structure(vivid, "wooden_box_mine") is None
    # 부탄은 구성이 색 말고도 **순서**(뚜껑→어깨→몸통)를 보므로 그대로 둔다
    assert hv.HazardVerifier._structure(_can((200, 60, 40), (40, 40, 220)),
                                        "butane_canister") is not None


def test_a_crop_too_small_is_not_scored():
    """**0 점이 아니라 '재지 않음'** 이다. 둘을 섞으면 작은 것이 전부 나쁜 점수가 된다."""
    assert hv.HazardVerifier._structure(np.zeros((4, 4, 3), np.uint8), "wooden_box_mine") is None
    assert hv.HazardVerifier._structure(None, "butane_canister") is None


def test_an_unknown_class_gets_no_structure_score():
    assert hv.HazardVerifier._structure(np.zeros((80, 40, 3), np.uint8), "styrofoam_box") is None


# ------------------------------------------- 프로토타입이 없어도 structure 는 산다
def test_structure_survives_without_a_prototype_bank():
    """`structure` 는 기준 사진에서 읽은 규칙이라 프로토타입이 **필요 없다**.
    준비 판정을 하나로 두면 npz 없는 기계에서 쓸 수 있는 근거까지 꺼진다(AI-C-05)."""
    v = hv.HazardVerifier({"classes": ["butane_canister"], "models": [],
                           "prototypes": "/nope.npz", "structure": True}, log=None)
    assert v.learned is False and v.ready is True
    img = np.zeros((200, 60, 3), np.uint8)
    img[:25] = (200, 60, 40); img[25:50] = (200, 200, 200); img[50:] = (40, 40, 220)
    d = {"name": "butane_canister", "conf": 0.5, "xyxy": [0, 0, 60, 200]}
    v(img, [d])
    assert d["verify"] is not None
    assert set(d["verify"]["per_model"]) == {"structure"}
    assert d["verify"]["per_model"]["structure"]["learned"] is False


def test_without_prototypes_there_is_no_match_row():
    """match 는 기준과의 거리다 — 기준이 없으면 **정의 자체가 없다.** 지어내지 않는다."""
    v = hv.HazardVerifier({"classes": ["butane_canister"], "models": [],
                           "prototypes": "/nope.npz", "structure": True}, log=None)
    d = {"name": "butane_canister", "conf": 0.5, "xyxy": [0, 0, 60, 200]}
    v(np.full((200, 60, 3), 90, np.uint8), [d])
    assert "clip_b32" not in (d["verify"] or {}).get("per_model", {})


def test_turning_structure_off_without_a_bank_disables_the_stage():
    v = hv.HazardVerifier({"classes": ["butane_canister"], "models": [],
                           "prototypes": "/nope.npz", "structure": False}, log=None)
    assert v.ready is False
    d = {"name": "butane_canister", "conf": 0.5, "xyxy": [0, 0, 60, 200]}
    assert v(np.zeros((200, 60, 3), np.uint8), [d]) == 0.0
    assert "verify" not in d


def test_a_missing_declared_bank_falls_back_to_the_bundled_one():
    """막 clone 한 기계에서도 `match` 가 선다. **어느 쪽을 썼는지는 기록에 남는다** —
    조용히 바뀌면 숫자가 어느 프로토타입의 것인지 알 수 없다."""
    v = hv.HazardVerifier({"classes": ["butane_canister"], "models": [],
                           "prototypes": "/nope/does-not-exist.npz"}, log=None)
    assert v.bank_bundled is True
    assert v.bank, "동봉본을 읽지 못했다"
    assert any(k.endswith("/_negative") for k in v.bank), "음성 묶음이 없다"


def test_a_declared_bank_wins_over_the_bundled_one(tmp_path):
    """현장이 자기 프로토타입을 적었으면 그것을 쓴다 — 동봉본은 **차선**이다."""
    import numpy as _np
    mine = tmp_path / "mine.npz"
    _np.savez(mine, **{"clip_b32/butane_canister": _np.zeros((1, 512), _np.float32)})
    v = hv.HazardVerifier({"classes": ["butane_canister"], "models": [],
                           "prototypes": str(mine)}, log=None)
    assert v.bank_bundled is False
    assert set(v.bank) == {"clip_b32/butane_canister"}
