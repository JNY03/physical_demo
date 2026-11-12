# -*- coding: utf-8 -*-
"""
slice360_common.py — 360 슬라이스 공통 함수 (sweep_latest.py · slice360_live.py 가 같이 쓴다)
================================================================================
2026-11-11

자르는 식은 e2p.py · e2p_sweep.py · 서버 anchor360_live.py 와 같은 py360convert.e2p 다.
  확인: 360camera_work/image backup/br40_live.jpg 를 u=230, v=-36.35 로 자르면 같은 폴더의 br40_live_slice.jpg
        (서버 anchor360_live 가 자른 것)와 픽셀 평균 차이 0.68 (JPEG 압축 차이 수준). u 를 1도만 바꿔도 11 이상.
  그래서 sweep 에서 고른 각도 = 실시간 슬라이스 각도 = 예전 서버 슬라이스 각도다.

각도
  u  가로. 0 = 파노라마 가운데, + 는 오른쪽 (위에서 보면 시계 방향). 파노라마 x = ((u/360 + 0.5) mod 1) × 가로
  v  세로. 0 = 수평, + 위, - 아래.                                    파노라마 y = (0.5 - v/180) × 세로
  ※ 사진 방식(ProStitcher)과 실시간 스트림(인카메라 스티칭)은 u 가 180도 다르다(261015 카메라 최종정리2).
    각도는 실시간 스트림 파노라마로 고른다.

화각
  가로 시야 fov(기본 90도). 세로 시야는 가로:세로 비율에서 정한다(픽셀이 정사각이 되게) —
  1:1 이면 90도, 4:3 이면 약 73.7도 (multicam_match/mast3r_pair_test2.py make_slice 와 같은 식).

py360convert 1.0 이상은 같은 각도·크기의 좌표표를 기억해 두므로(lru_cache) 같은 각도로 계속 자르면 빠르다
(3840×1920 → 960×960 약 20~25ms, Claude 작업 공간 CPU 측정).
"""
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

KST = timezone(timedelta(hours=9))


def iso_ms(t=None):
    """epoch 초 → '2026-11-11T14:10:31.086+09:00' (통합규약 §2-2 와 같은 꼴)"""
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")


def parse_aspect(text):
    a, b = str(text).split(":")
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        raise ValueError("비율은 양수여야 합니다: %s" % text)
    return a, b


def slice_size(width, aspect):
    aw, ah = parse_aspect(aspect)
    return int(width), int(round(width * ah / aw))


def fov_v_of(fov, width, height):
    """가로 시야 fov(도)와 출력 크기로 세로 시야(도) — 픽셀이 정사각이 되게"""
    return math.degrees(2 * math.atan(math.tan(math.radians(fov) / 2) * height / float(width)))


def slice_pano(equ_bgr, u, v, fov=90.0, width=960, aspect="1:1"):
    """등장방형 BGR → (u, v) 방향 원근 사진. (사진 uint8, 세로 시야 도) 를 돌려준다."""
    import py360convert
    w, h = slice_size(width, aspect)
    fv = fov_v_of(fov, w, h)
    p = py360convert.e2p(equ_bgr, fov_deg=(float(fov), fv), u_deg=float(u), v_deg=float(v),
                         out_hw=(h, w), mode="bilinear")
    if p.dtype != np.uint8:
        p = np.clip(p, 0, 255).astype(np.uint8)
    return p, fv


def pano_xy(u, v, w, h):
    """각도 → 파노라마 픽셀 좌표 (방향 표시용)"""
    x = ((u / 360.0 + 0.5) % 1.0) * w
    y = (0.5 - v / 180.0) * h
    return x, y


def is_pano(shape, tol=0.02):
    h, w = shape[:2]
    return h > 0 and abs(w / float(h) - 2.0) <= tol


# ---------------------------------------------------------------- 각도 파일 (실행 중에 각도 바꾸기)
def read_angle_file(path):
    """{"u": .., "v": .., "fov": .., "width": .., "aspect": ".."} — 없거나 못 읽으면 None"""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    try:
        out = {"u": float(d["u"]), "v": float(d.get("v", 0.0))}
        if d.get("fov") is not None:
            out["fov"] = float(d["fov"])
        if d.get("width") is not None:
            out["width"] = int(d["width"])
        if d.get("aspect") is not None:
            parse_aspect(d["aspect"])
            out["aspect"] = str(d["aspect"])
    except (KeyError, TypeError, ValueError):
        return None
    return out


def write_angle_file(path, angle, by="cli"):
    """한 번에 바꿔 쓴다 (읽는 쪽이 반쯤 쓴 파일을 읽지 않게: 임시 파일 → 이름 바꾸기)"""
    d = dict(angle)
    d["updated_at"] = iso_ms()
    d["updated_by"] = by
    tmp = "%s.tmp%d" % (path, os.getpid())
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
