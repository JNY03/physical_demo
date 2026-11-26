# -*- coding: utf-8 -*-
"""bytetrack.py — ByteTrack 트래커 어댑터 (2026-11-27). provides [track]

물체 추적 기능(object_track)의 트래커 슬롯 선택지. ultralytics 의 BYTETracker 를 botsort.py 와 **같은 어댑터**로
감싼다(설정 파일 · 클래스만 다르다). vision_infer.VendorTracker(backend: bytetrack — 탐지기 기록 안의 번호)와
같은 트래커지만, 이것은 **기능 하나로서** 고른 탐지 기능의 검출을 받아 vision_stream:<소스>:track 에 궤적을 낸다.

BoT-SORT 와 다른 점: GMC(전역 움직임 보정)가 없다 — 프레임을 넘겨도 쓰지 않는다. 고정 카메라 · CPU 가 빠듯한
기기에 맞다(실측 p90 1.1 ms, 검출 7개/장, i7 4코어).

cfg: bytetrack.yaml 의 값 (track_high_thresh · new_track_thresh · track_buffer · match_thresh …) +
     on_untracked (keep|drop) · keep_boxes (detection|tracker) · reset_gap_s
의존: ultralytics · lap · numpy. 같은 폴더의 botsort.py 를 쓴다(기기 컨테이너에는 두 파일을 함께 복사한다).
"""
from __future__ import annotations

try:                                                    # runners 꾸러미로 읽을 때 (load_runner · 시험)
    from .botsort import Runner as _Adapter
except ImportError:                                     # 파일 둘만 복사한 기기 컨테이너
    from botsort import Runner as _Adapter  # type: ignore


class Runner(_Adapter):
    name = "bytetrack"
    kind = "track"
    provides = ("track",)
    TRACKER_YAML = "bytetrack.yaml"
    TRACKER_MODULE = "ultralytics.trackers.byte_tracker"
    TRACKER_CLASS = "BYTETracker"
