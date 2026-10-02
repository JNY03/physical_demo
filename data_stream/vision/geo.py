# -*- coding: utf-8 -*-
"""
geo.py — 검출 픽셀 + depth + 드론 위치·자세 → 검출 객체의 GPS 좌표
================================================================================
2026-10-02

단위 (data_stream/GPS_단위_정리_261002.md, 브리지 MQTT state 계약과 같음)
  위도·경도  도 (WGS84 십진수, float64 — float32 금지)
  고도      m  (amsl_m 해발 / relative_m 이륙 지점 기준)
  자세      도 (roll/pitch/yaw, NED 기준, yaw 0 = 진북 · 시계방향 +). 삼각함수 직전에만 rad 로 바꾼다
  gps.fix_type < min_fix_type (기본 2) 이면 위도·경도를 내지 않는다 (null)

좌표계
  카메라  OpenCV  x 오른쪽 · y 아래 · z 광축
  기체    FRD     x 앞 · y 오른쪽 · z 아래
  지역    NED     x 북 · y 동 · z 아래   (드론 위치 기준 오프셋, m)

수식
  depth 는 광축 방향 Z 다 (UniDepth·MoGe 모두 points[..., 2]). 직선거리가 아니다.
  P_c = Z · [(u-cx)/fx, (v-cy)/fy, 1]                          카메라 좌표 (m), 직선거리 = |P_c|
  R_bc = Rz(mount_yaw) · Ry(-pitch_down) · CAM_TO_FRD · Rz(mount_roll)    카메라 → 기체 (장착각)
  R_nb = Rz(yaw) · Ry(pitch) · Rx(roll)                        기체 → NED (PX4 ZYX 오일러)
  [N, E, D] = R_nb · (R_bc · P_c + lever_arm)
  위도 = lat0 + N / (R_M + h) · 180/π,  경도 = lon0 + E / ((R_N + h) · cos lat0) · 180/π,  고도 = amsl0 - D
    R_M = a(1-e²)/(1-e² sin² lat0)^1.5,  R_N = a/√(1-e² sin² lat0)   (WGS84, 수 km 이내 cm 수준)

위치·자세 찾기
  드론 브리지는 프레임마다 state 를 한 건 낸다 (reason "frame", ts = 그 프레임 sender.ts).
  프레임 sender.ts 와 ts 가 가장 가까운 state 를 쓴다 (둘 다 Pi 시계라 PC 시계와 어긋나도 상관없다).
  live            Redis state_stream:<기기> 의 최근 항목
  저장된 이미지    <실행>/states.jsonl
"""
import json
import math

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
CAM_TO_FRD = np.array([[0.0, 0.0, 1.0],     # 기체 x(앞)    = 카메라 z(광축)
                       [1.0, 0.0, 0.0],     # 기체 y(오른쪽) = 카메라 x
                       [0.0, 1.0, 0.0]])    # 기체 z(아래)  = 카메라 y


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mount_matrix(mount):
    """카메라 → 기체(FRD) 회전. pitch_down_deg: 정면에서 아래로 숙인 각 (수직 하방 = 90)
    yaw_deg: 오른쪽으로 돌린 각, roll_deg: 광축 둘레로 시계방향 돌린 각"""
    m = mount or {}
    p = math.radians(float(m.get("pitch_down_deg", 0.0)))
    y = math.radians(float(m.get("yaw_deg", 0.0)))
    r = math.radians(float(m.get("roll_deg", 0.0)))
    return rot_z(y) @ rot_y(-p) @ CAM_TO_FRD @ rot_z(r)


def body_to_ned(roll_deg, pitch_deg, yaw_deg):
    return rot_z(math.radians(yaw_deg)) @ rot_y(math.radians(pitch_deg)) @ rot_x(math.radians(roll_deg))


def ned_to_geodetic(lat0, lon0, h0, n, e, d):
    """드론 위치(lat0, lon0 도, h0 해발 m 또는 None) 에서 N·E·D(m) 만큼 떨어진 점의 (위도, 경도, 해발)"""
    phi = math.radians(lat0)
    w = math.sqrt(1.0 - WGS84_E2 * math.sin(phi) ** 2)
    h = h0 or 0.0
    rm = WGS84_A * (1.0 - WGS84_E2) / w ** 3 + h
    rn = WGS84_A / w + h
    lat = lat0 + math.degrees(n / rm)
    lon = lon0 + math.degrees(e / (rn * math.cos(phi)))
    return lat, lon, (None if h0 is None else h0 - d)


# ---------------------------------------------------------------- 위치·자세 (MQTT state)
def pose_from_state(p):
    """브리지 MQTT state payload → 계산에 쓰는 값만. 없는 값은 None"""
    g = p.get("gps") or {}
    a = p.get("attitude") or {}
    al = p.get("altitude") or {}
    return {"ts": p.get("ts"),
            "lat": g.get("lat"), "lon": g.get("lon"), "fix_type": g.get("fix_type"),
            "satellites": g.get("satellites"),
            "amsl_m": al.get("amsl_m"), "relative_m": al.get("relative_m"),
            "roll_deg": a.get("roll_deg"), "pitch_deg": a.get("pitch_deg"), "yaw_deg": a.get("yaw_deg"),
            "att_age_s": a.get("age_s"), "gps_age_s": g.get("age_s")}


def state_payload(topic, payload):
    """state 채널이고 ts 가 있는 payload 만 dict 로. 아니면 None"""
    if isinstance(topic, bytes):
        topic = topic.decode("utf-8", "replace")
    if not topic.endswith("/state"):
        return None
    try:
        p = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(p, dict) or not isinstance(p.get("ts"), (int, float)):
        return None
    return p


def nearest(states, ts):
    """[(ts, pose)] 에서 ts 에 가장 가까운 것. ts 가 없으면 가장 최근 것"""
    if not states:
        return None, None
    if ts is None:
        t, pose = max(states, key=lambda s: s[0])
        return pose, None
    t, pose = min(states, key=lambda s: abs(s[0] - ts))
    return pose, t - ts


class DiskStates:
    """<실행>/states.jsonl — 늘어나는 파일을 이어 읽는다 (--watch 와 live 저장 중인 실행 폴더)"""

    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.ts = []        # 정렬된 ts
        self.poses = []

    def refresh(self):
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            for line in f:
                if not line.endswith(b"\n"):
                    break                               # 쓰는 중인 마지막 줄
                self.pos += len(line)
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                p = state_payload(d.get("topic") or "", d.get("payload"))
                if p is not None:
                    self.ts.append(float(p["ts"]))
                    self.poses.append(pose_from_state(p))
        order = np.argsort(self.ts, kind="stable")
        if len(order) and np.any(order != np.arange(len(order))):
            self.ts = [self.ts[i] for i in order]
            self.poses = [self.poses[i] for i in order]

    def lookup(self, ts):
        self.refresh()
        if not self.ts:
            return None, None
        if ts is None:
            return self.poses[-1], None
        i = int(np.searchsorted(self.ts, ts))
        cand = [j for j in (i - 1, i) if 0 <= j < len(self.ts)]
        j = min(cand, key=lambda k: abs(self.ts[k] - ts))
        return self.poses[j], self.ts[j] - ts


def redis_lookup(r, key, ts, scan=64):
    """Redis state_stream:<기기> 최근 scan 개 중 ts 가 가장 가까운 state"""
    states = []
    for _, f in r.xrevrange(key, "+", "-", count=scan):
        p = state_payload(f.get(b"topic") or b"", f.get(b"payload"))
        if p is not None:
            states.append((float(p["ts"]), pose_from_state(p)))
    return nearest(states, ts)


def frame_ts(header):
    s = (header or {}).get("sender") or {}
    for k in ("ts", "captured_at", "observed_at"):
        if isinstance(s.get(k), (int, float)):
            return float(s[k])
    return None


# ---------------------------------------------------------------- 계산
class GeoLocator:
    """기기 하나의 장착 정보로 검출마다 카메라 좌표 → NED 오프셋 → 위도·경도를 붙인다"""

    def __init__(self, dev_geo, geo_cfg):
        self.R_bc = mount_matrix(dev_geo)
        self.lever = np.array(dev_geo.get("lever_arm_m") or [0.0, 0.0, 0.0], float)
        self.min_fix = int(geo_cfg.get("min_fix_type", 2))

    def locate(self, dets, K, pose):
        """dets 각각에 range_m · cam_xyz_m · ned_m · lat · lon · alt_m 를 붙이고 프레임 상태를 돌려준다.
        상태: ok | no_state (맞는 state 없음) | no_attitude | no_fix (오프셋 ned_m 까지만)"""
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
        status = "ok"
        R = arm = None
        if pose is None:
            status = "no_state"
        elif None in (pose.get("roll_deg"), pose.get("pitch_deg"), pose.get("yaw_deg")):
            status = "no_attitude"
        else:
            R_nb = body_to_ned(pose["roll_deg"], pose["pitch_deg"], pose["yaw_deg"])
            R, arm = R_nb @ self.R_bc, R_nb @ self.lever
            if (pose.get("fix_type") or 0) < self.min_fix or pose.get("lat") is None or pose.get("lon") is None:
                status = "no_fix"
        for d in dets:
            d.update(range_m=None, ned_m=None, lat=None, lon=None, alt_m=None)
            z, uv = d.get("depth_m"), d.get("uv")
            if z is None or uv is None:
                continue
            pc = z * np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0])
            d["range_m"] = round(float(np.linalg.norm(pc)), 3)
            d["cam_xyz_m"] = [round(float(v), 3) for v in pc]
            if R is None:
                continue
            n, e, dd = R @ pc + arm
            d["ned_m"] = [round(float(n), 3), round(float(e), 3), round(float(dd), 3)]
            if status == "ok":
                lat, lon, alt = ned_to_geodetic(float(pose["lat"]), float(pose["lon"]), pose.get("amsl_m"),
                                                float(n), float(e), float(dd))
                d["lat"], d["lon"] = round(lat, 8), round(lon, 8)
                d["alt_m"] = None if alt is None else round(alt, 2)
        return status
