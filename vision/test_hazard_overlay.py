"""hazard_overlay 카드 배치 — 카드가 **그려진 박스·라벨을 가리지 않는다**.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python -m pytest test_hazard_overlay.py -q
"""
import numpy as np
from PIL import Image

import hazard_overlay as ho


def _overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _fake_card(n, box, w=300, h=200):
    return {"n": n, "box": box, "w": w, "h": h}


def _run(cards, obstacles, size=(1280, 720)):
    """`_cards_on` 을 돌리되 그리기 대신 놓인 자리를 모은다."""
    got = {}
    orig = ho._paint_card
    ho._paint_card = lambda pil, dr, pen, k, cx, cy: got.__setitem__(k["n"], (cx, cy))
    try:
        pil = ho._cards_on(Image.new("RGB", size), None, cards, obstacles)
    finally:
        ho._paint_card = orig
    return pil, got


def test_cards_avoid_boxes_labels_and_each_other():
    boxes = [(100, 300, 400, 650), (420, 280, 700, 640), (800, 100, 1000, 300)]
    labels = [(b[0], b[1] - 26, b[0] + 200, b[1]) for b in boxes]
    cards = [_fake_card(i + 1, b) for i, b in enumerate(boxes)]
    pil, got = _run(cards, boxes + labels)
    rects = {n: (x, y, x + 300, y + 200) for n, (x, y) in got.items()}
    assert set(rects) == {1, 2, 3}                     # 하나도 빠지지 않는다
    for r in rects.values():
        assert not any(_overlap(r, o) for o in boxes + labels)
    rs = list(rects.values())
    assert not any(_overlap(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs)))


def test_no_room_spills_to_side_panel_instead_of_covering():
    full = [(0, 0, 1280, 720)]                         # 화면 전체가 박스다
    cards = [_fake_card(1, full[0]), _fake_card(2, full[0])]
    pil, got = _run(cards, full)
    assert pil.size[0] > 1280                          # 패널이 덧대졌다
    assert all(x >= 1280 for x, _y in got.values())    # 카드는 전부 화면 밖(패널)에


def test_free_spot_prefers_near_box():
    occ = np.zeros((720, 1280), np.uint8)
    occ[200:500, 500:800] = 1                          # 물체
    x, y = ho._free_spot(occ, 200, 100, (500, 200, 800, 500))
    gap = max(0, 500 - (x + 200), x - 800) + max(0, 200 - (y + 100), y - 500)
    assert gap <= ho.CARD_STEP                         # 바로 옆에 붙는다


# ---------------------------------------------------------------- 점수 판정 (assess)
def test_score_is_weighted_mean_of_measured_cues_only():
    ho.load_cues()
    w = ho.weights()
    assert "size" not in w and not ho.cue_on("size")                   # size 단서는 뺐다
    # 환경이 바뀌어도 덜 흔들리는 단서를 높게: 글 설명·기준 사진 > 이름 > 색 규칙
    assert w["semantic"] >= w["match"] > w["class"] > w["material"]
    det = {"cls": "butane_canister", "conf": 0.6, "xyxy": [0, 0, 100, 300], "lead": "beverage can",
           "verify": None}                                           # match·structure 는 못 잰다
    colour = {"red": 0.2, "blue": 0.0, "both": 0.0, "wood": 0.0, "chrome": 0.3, "lit": 0.2}
    sc, rows, _src = ho.grounds(det, 1280, 720, None, None, colour=colour)
    measured = [(k, p) for k, p, *_ in rows if p is not None]
    assert {k for k, _ in measured} >= {"class", "material"}
    assert "match" not in {k for k, _ in measured}                   # 못 잰 단서는 빠진다(0 이 아니다)
    assert "size" not in {k for k, _ in measured}
    want = sum(p * w[k] for k, p in measured) / sum(w[k] for k, p in measured)
    assert abs(sc - want) < 1e-9


def test_assess_compares_every_hazard_class_and_uses_min_score():
    ho.load_cues()
    img = np.full((200, 300, 3), (120, 170, 200), np.uint8)          # 따뜻한 저채도(원목 비슷)
    det = {"name": "unconfirmed", "conf": 0.4, "xyxy": [50, 50, 150, 150], "lead": "storage box",
           "verify_by": {"wooden_box_mine": {"per_model": {"clip_b32": {"margin": 0.1}}}}}
    a = ho.assess(det, img)
    assert set(a["scores"]) == set(ho.HAZARD)                        # 이름이 없어도 둘 다 견준다
    assert a["min_score"] == ho.min_score()
    best = max(a["scores"], key=lambda c: a["scores"][c] or -1)
    assert a["suspect"] == (best if a["scores"][best] >= a["min_score"] else None)
    keys = [r[0] for r in a["cards"][best]["rows"] if r[1] is not None]
    order = sorted(keys, key=lambda k: -ho.weights().get(k, 0))
    assert keys == order                                             # 무게 순으로 늘어선다


def test_card_scale_follows_photo_width():
    # 카드는 사진 폭에 비례한다 — 상한이 없어 8000 px 사진도 1280 px 와 같은 비율이다
    for w in (1280, 3840, 8000):
        assert abs(ho.card_scale(w) - w / ho.CARD_BASE_W) < 1e-9
    assert ho.card_scale(320) == ho.CARD_MIN_SCALE                    # 작은 그림은 하한까지만


def test_shape_cue_separates_box_can_and_irregular():
    import cv2
    def rect(w, h):
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    box = ho.shape_cue(rect(160, 100), "wooden_box_mine")[0]         # 1.6:1 볼록 사각형
    can = ho.shape_cue(rect(70, 200), "butane_canister")[0]          # 2.9:1 원통 옆모습
    star = np.array([[50, 0], [62, 38], [100, 38], [69, 62], [81, 100], [50, 76],
                     [19, 100], [31, 62], [0, 38], [38, 38]], np.float32)   # 오목한 묶음
    assert box > 90 and can > 90
    assert ho.shape_cue(star, "wooden_box_mine")[0] < 60
    assert ho.shape_cue(rect(100, 100), "butane_canister")[0] < 80   # 정사각은 통 옆모습이 아니다


def test_small_crop_does_not_score_appearance():
    ho.load_cues()
    det = {"cls": "butane_canister", "conf": 0.6, "xyxy": [0, 0, 20, 30], "lead": "can",
           "verify": {"semantic": 0.99, "per_model": {"clip_b32": {"margin": 0.2}}}}
    colour = {"red": 0.5, "blue": 0.0, "both": 0.0, "wood": 0.0, "chrome": 0.3, "lit": 0.5}
    _sc, rows, _src = ho.grounds(det, 1280, 720, None, None, colour=colour)
    got = {k for k, p, *_ in rows if p is not None}
    assert not got & {"semantic", "match", "material"}               # 20 px: 묻지 않는다(0 점 아님)
