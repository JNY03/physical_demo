"""river 의 부품 접기(_drop_nested) — 큰 상자 안의 작은 상자는 지우고, 다른 위험물은 남긴다.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python -m pytest test_river_nested.py -q
"""
from types import SimpleNamespace

from vision_infer import RiverRunner


def _d(box, sus=None):
    return {"xyxy": box, "assess": {"suspect": sus} if sus is not None else {"suspect": None}}


def _run(dets, ratio=0.7):
    return RiverRunner._drop_nested(SimpleNamespace(nest_ratio=ratio), dets)


def test_parts_inside_whole_are_dropped():
    can = _d([100, 100, 200, 400], "butane_canister")
    cap = _d([130, 100, 170, 140], "butane_canister")               # 같은 클래스로 의심된 뚜껑
    label = _d([110, 200, 190, 300])                                 # 의심 아님
    kept, n = _run([cap, can, label])
    assert kept == [can] and n == 2


def test_other_hazard_inside_big_box_is_kept():
    box = _d([0, 0, 400, 400], "wooden_box_mine")
    can = _d([50, 50, 150, 350], "butane_canister")                  # 큰 상자에 덮였지만 다른 위험물
    kept, n = _run([box, can])
    assert kept == [box, can] and n == 0


def test_side_by_side_overlap_is_not_nesting():
    a, b = _d([0, 0, 200, 200]), _d([150, 0, 350, 200])              # 25% 만 겹친다
    kept, n = _run([a, b])
    assert len(kept) == 2 and n == 0
