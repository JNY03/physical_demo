"""
피지컬팀 mk2 — Unitree Go1 제어기 링크 (HW-R-01)
==================================================
`ControllerLink` 의 Go1 구현. **읽기 전용이다** — 로봇에 명령을 보내지 않는다.

## 왜 SDK(UDP)를 쓰지 않는가

Go1 의 고수준 SDK 경로(`192.168.123.161:8082`)는 **요청/응답 구조**라 가만히 듣기만
해서는 아무것도 오지 않는다. 상태를 받으려면 `HighCmd` 를 보내야 하는데, 그 순간
sport mode 에서 **제어권을 가져온다.** 서 있는 로봇이 힘이 빠져 주저앉을 수 있다.

대신 Go1 은 자체 MQTT 브로커(`192.168.123.161:1883`)로 텔레메트리를 **이미 발행하고
있다.** 구독만 하면 되고, 구독은 로봇을 움직일 수 없다. 그래서 이 경로를 쓴다.

| 토픽 | 내용 | 주기 |
|---|---|---|
| `robot/state` | 84바이트 — 몸통 자세·12관절·높이·속도 | 12.5 Hz |
| `bms/state` | 34바이트 — SDK `BmsState` 구조체 그대로 | 0.5 Hz |
| `usys/version/*` | 각 노드 버전 (retained) | — |

## 페이로드 레이아웃 (pi7 에서 역공학, 2026-08-31)

`bms/state` 는 SDK `comm.h` 의 `BmsState` 와 정확히 일치했다 — 값이 전부 물리적으로
말이 됐다(SOC 43%, 방전 4.93A, 124사이클, BQ 32/31°C).

`robot/state` 는 공개 규격이 없어 **관측으로 추정**했다. 확신도를 구분해 둔다.

| 오프셋 | 형식 | 해석 | 확신도 |
|---|---|---|---|
| 0~5 | int16 ×3 | 몸통 roll·pitch·yaw (도) | **높음** — 기립 (0,0,-25) + 주행 중 회전과 함께 변화 확인 |
| 6~29 | int16 ×12 | 12관절 각도 (도), 4다리 × (hip,thigh,calf) | **높음** — 기립 자세 (0,46,-92)×4 일치 |
| 30~51 | int16 ×11 | 미상 (보행 중 크게 요동 — 발 접지력·위상 추정) | 낮음 |
| 52~59 | float32 ×2 | **위치 x·y (m, 오도메트리)** | **확정** — 2026-09-01 실주행: 전진 시 x 단조 증가(0.008→0.54), 정지 시 고정 |
| 60~67 | float32 ×2 | 몸통 높이 (m) | **높음** — 0.29~0.33, 보행 중 미세 요동 |
| 68~75 | float32 ×2 | **속도 x·y (m/s)** | **확정** — 주행 중만 비영(최대 0.41), 방향 부호 일치, 정지 시 0 |
| 76~83 | float32 ×2 | 각속도 계열 — f80 은 요 각속도(rad/s)로 추정 | 중간 — 회전 구간에서만 비영, 요 변화율과 부합 |

**확정 방법 (2026-09-01).** 사용자가 리모컨으로 주행(전진 0.5m·횡이동·소회전)하는
동안 이 스트림을 기록만 했다(`/tmp/drive_rec.py`, 211초 2,637표본). 명령은 한 건도
보내지 않았다 — 구독은 로봇을 움직일 수 없다.

## 명령(HW-R-06)은 아직 없다

`send_command()` 는 의도적으로 거부한다. 명령 경로는 로봇이 **엎드린 상태 또는 거치대에서**,
사람이 지켜보는 가운데 별도로 붙인다.
"""
import struct
import threading
import time

import paho.mqtt.client as mqtt

from common import config
from robot.controller_link import ControllerLink, RobotState

# --- robot/state 레이아웃 상수 (위 표 참조) ---
# ⚠ 전부 **바이트 오프셋**이다. int16 인덱스와 헷갈리면 값이 조용히 깨진다
# (실제로 JOINTS 를 인덱스 3 으로 뒀다가 관절이 쓰레기값으로 나왔다).
BODY_RPY = 0            # int16 ×3, 도   — 바이트 0,2,4
JOINTS = 6              # int16 ×12, 도  — 바이트 6~29
UNKNOWN = 30            # int16 ×11 (미상) — 바이트 30~51
UNKNOWN_N = 11
F_POS = 52              # float32 ×2 (추정)
F_HEIGHT = 60           # float32, m
F_VEL = 68              # float32 ×2, m/s
F_TAIL = 76             # float32 ×2
STATE_LEN = 84
BMS_LEN = 34

# --- bms/state 레이아웃 — SDK comm.h `BmsState` 와 일치함을 실측 확인했다 ---
# 2026-09-30 pi7: ver 1.4 / status 1 / SOC 18% / -5644mA / 144사이클 /
# BQ 31·31°C / MCU 32·35°C. 값이 전부 물리적으로 말이 된다.
BMS_CURRENT = 4         # int32, mA (음수 = 방전)
BMS_CYCLE = 8           # uint16
BMS_BQ_NTC = 10         # int8 ×2, °C
BMS_MCU_NTC = 12        # int8 ×2, °C
BMS_CELLS = 14          # uint16 ×10, mV

# 셀 전압 10칸 중 **실제로 물린 칸만** 값이다. Go1 은 6S 인데 나머지 네 칸에
# 32mV 같은 빈 값이 들어온다(실측 2026-09-30: 3424×6 + 32×4). 그걸 셀 전압으로
# 내보내면 "셀 하나가 완전히 죽었다"로 보인다 — 빈 칸은 측정값이 아니다.
CELL_MIN_MV = 1000


class Go1Link(ControllerLink):
    """Go1 내부 MQTT 를 구독해 상태만 읽는다. 명령은 보내지 않는다."""

    def __init__(self, host=None, port=1883):
        self.host = host or config.GO1_MQTT_HOST
        self.port = port
        self._lock = threading.Lock()
        self._state = None          # 최신 robot/state 원본
        self._bms = None            # 최신 bms/state 원본
        self._last_state_at = 0.0
        self._last_bms_at = 0.0
        self._versions = {}
        self._connected = False

        kw = {"callback_api_version": mqtt.CallbackAPIVersion.VERSION2,
              "client_id": f"hw-go1-{int(time.time())}"}
        self._c = mqtt.Client(**kw)
        self._c.on_connect = self._on_connect
        self._c.on_disconnect = self._on_disconnect
        self._c.on_message = self._on_message
        self._c.reconnect_delay_set(min_delay=1, max_delay=config.RECONNECT_MAX_DELAY)
        self._c.connect_async(self.host, self.port, keepalive=20)
        self._c.loop_start()

    # ---------- MQTT ----------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            print(f"[Go1] 접속 실패 {reason_code}")
            return
        self._connected = True
        print(f"[Go1] 내부 MQTT 접속 — {self.host}:{self.port}")
        # 구독만 한다. 발행하지 않는다 — 로봇을 움직일 수 있는 경로를 열지 않는다.
        for t in ("robot/state", "bms/state", "usys/version/#"):
            client.subscribe(t, qos=0)

    def _on_disconnect(self, client, userdata, flags=None, reason_code=None, properties=None):
        self._connected = False
        print(f"[Go1] 내부 MQTT 단절 (rc={getattr(reason_code, 'value', reason_code)})")

    def _on_message(self, client, userdata, msg):
        now = time.time()
        with self._lock:
            if msg.topic == "robot/state" and len(msg.payload) >= STATE_LEN:
                self._state, self._last_state_at = msg.payload, now
            elif msg.topic == "bms/state" and len(msg.payload) >= BMS_LEN:
                self._bms, self._last_bms_at = msg.payload, now
            elif msg.topic.startswith("usys/version/"):
                self._versions[msg.topic.rsplit("/", 1)[-1]] = \
                    msg.payload.decode("utf-8", "replace")[:200]

    # ---------- ControllerLink 구현 ----------
    def read_state(self) -> RobotState:
        with self._lock:
            st, bms = self._state, self._bms
            now = time.time()
            st_age = now - self._last_state_at if self._state else None
            bms_age = now - self._last_bms_at if self._bms else None

        if st is None:
            # 아직 한 건도 못 받았다. 값을 지어내지 않는다 — 모르는 것은 모른다고 낸다.
            raise RuntimeError("go1_state_unavailable")

        # 전부 0 인 프레임은 **측정값이 아니라 데이터 부재**다. 로봇 전원 직후나
        # sport mode 초기화 전에 퍼블리셔가 빈 버퍼를 그대로 내보내는 것을 실측했다
        # (2026-09-10: robot/state 84B·bms/state 34B 가 85초간 전부 0). 이걸 그대로 읽으면
        # 자세 0도·위치 원점·배터리 0% 라는 '있어 보이는 거짓값'이 상위로 올라간다.
        if not any(st):
            raise RuntimeError("go1_state_zeroed")

        rpy = struct.unpack_from("<3h", st, BODY_RPY)          # 도
        joints = struct.unpack_from("<12h", st, JOINTS)        # 도
        height = struct.unpack_from("<f", st, F_HEIGHT)[0]     # m
        pos = struct.unpack_from("<2f", st, F_POS)             # 추정
        vel = struct.unpack_from("<2f", st, F_VEL)             # m/s
        tail = struct.unpack_from("<2f", st, F_TAIL)           # 각속도 계열(추정)

        bat = self._bms_detail(bms)
        speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5

        # mode: Go1 의 동작 모드를 이 스트림에서 확정하지 못했다. 관측 가능한 것으로
        # 대신한다 — 스트림이 최근에 왔고 높이가 기립 범위면 서 있는 것으로 본다.
        mode = self._infer_mode(st_age, height, speed)

        return RobotState(
            # 모르면 None 이다. 0.0 으로 바꾸면 "방전 직전"과 구별되지 않아
            # 배터리 경보가 오발동하고 임무가 battery_too_low 로 거부된다.
            battery_pct=bat["soc_pct"] if bat else None,
            x=round(pos[0], 3), y=round(pos[1], 3),
            heading_deg=float(rpy[2]),
            speed_mps=round(speed, 3),
            mode=mode,
            # --- 여기부터는 예전에 언팩해 놓고도 버리던 값들이다 ---
            roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]),
            body_height_m=round(height, 4),
            vx=round(vel[0], 4), vy=round(vel[1], 4),
            # F_TAIL 두 칸 중 뒤쪽이 요 각속도로 보인다(회전 구간에서만 비영).
            # **확정이 아니다** — 앞쪽 칸은 이름을 붙이지 않고 raw_unknown 에 남긴다.
            yaw_rate=round(tail[1], 4),
            joints_deg=list(joints),
            battery_current_ma=bat["current_ma"] if bat else None,
            battery_cycles=bat["cycles"] if bat else None,
            battery_voltage_v=bat["voltage_v"] if bat else None,
            battery_temp_c=bat["temp_c"] if bat else None,
            battery_cells_mv=bat["cells_mv"] if bat else None,
            battery_status=bat["status"] if bat else None,
            state_age_s=round(st_age, 3) if st_age is not None else None,
            bms_age_s=round(bms_age, 2) if bms_age is not None else None,
            # 해독 못 한 바이트 30~51 + F_TAIL 앞칸. 이름 없이 hex 로만 남긴다.
            raw_unknown=st[UNKNOWN:UNKNOWN + UNKNOWN_N * 2].hex()
                        + struct.pack("<f", tail[0]).hex(),
        )

    @staticmethod
    def _bms_detail(bms):
        """SDK BmsState 34바이트를 **전부** 읽는다. 레이아웃은 실측 확인했다(상단 상수).

        예전에는 SOC 한 칸만 읽었다. 같은 프레임에 전류·사이클·온도 4점·셀 전압이
        이미 들어 있는데 버리고 있었던 것이다 — 잔량 하나로는 남은 시간도, 과열도,
        셀 불균형도 알 수 없다.
        """
        if bms is None or len(bms) < BMS_LEN:
            return None
        if not any(bms):
            return None            # 전부 0 = 아직 안 채워진 프레임(데이터 부재)
        cells = struct.unpack_from("<10H", bms, BMS_CELLS)
        live = [v for v in cells if v >= CELL_MIN_MV]      # 빈 칸은 싣지 않는다
        return {
            "version": f"{bms[0]}.{bms[1]}",
            "status": bms[2],
            "soc_pct": float(bms[3]),
            "current_ma": struct.unpack_from("<i", bms, BMS_CURRENT)[0],
            "cycles": struct.unpack_from("<H", bms, BMS_CYCLE)[0],
            "temp_c": list(struct.unpack_from("<2b", bms, BMS_BQ_NTC))
                      + list(struct.unpack_from("<2b", bms, BMS_MCU_NTC)),
            "cells_mv": live,
            "cell_spread_mv": (max(live) - min(live)) if live else None,
            "voltage_v": round(sum(live) / 1000.0, 3) if live else None,
        }

    @staticmethod
    def _infer_mode(age, height, speed):
        """관측으로 추정하는 동작 모드. **Go1 의 실제 mode 필드가 아니다.**
        확정하려면 `controller/current_action` 해독이나 SDK HighState 가 필요하다."""
        if age is None or age > config.GO1_STALE_S:
            return "unknown"
        if height < 0.15:
            return "idle"            # 엎드림
        return "mission" if speed > 0.05 else "idle"

    def send_command(self, action, params):
        """HW-R-06. **의도적으로 막아 둔다.**

        Go1 에 명령을 보내려면 SDK 의 UDP 경로로 HighCmd 를 실어야 하는데, 그 순간
        sport mode 에서 제어권을 가져와 서 있던 로봇이 주저앉을 수 있다. 명령 경로는
        로봇이 엎드린 상태에서 사람이 지켜보는 가운데 따로 붙인다."""
        raise NotImplementedError(
            "go1_command_not_enabled — 명령 경로는 안전 조건 확립 후 별도로 활성화한다")

    def link_health(self):
        with self._lock:
            age = time.time() - self._last_state_at if self._state else None
            bms_age = time.time() - self._last_bms_at if self._bms else None
        if not self._connected or age is None:
            return "fault"
        if age > config.GO1_STALE_S:
            return "fault"           # 붙어는 있는데 값이 안 온다
        if bms_age is None or bms_age > config.GO1_BMS_STALE_S:
            return "degraded"        # 자세는 오는데 배터리가 안 온다
        return "ok"

    # ---------- 펌웨어 ----------
    def versions(self):
        """`usys/version/*` (retained) 에 로봇이 스스로 올려 둔 노드별 펌웨어 목록.

        구독하는 순간 이미 받아 놓고 진단에서만 한 줄 쓰고 버렸다. 로봇 쪽 구성이
        바뀌었는지는 이 값으로만 알 수 있다 — 상태가 갑자기 달라졌을 때 "로봇이
        바뀐 것인가"를 가릴 근거다. 상태(20Hz)가 아니라 요약(10초)에 싣는다.
        """
        with self._lock:
            versions = dict(self._versions)
        out = {"app": versions.get("app"), "sport_mode": None, "nodes": {}}
        for name, raw in versions.items():
            if name == "app":
                continue
            parts = dict(p.split(":", 1) for p in raw.split(";") if ":" in p)
            out["nodes"][name] = parts
            if name == "raspi" and "sportMode" in parts:
                out["sport_mode"] = parts["sportMode"]
        return out

    # ---------- 진단 ----------
    def diagnostics(self):
        with self._lock:
            st, bms = self._state, self._bms
            age = time.time() - self._last_state_at if self._state else None
        fw = self.versions()
        d = {"connected": self._connected, "host": self.host,
             "state_age_s": round(age, 2) if age is not None else None,
             "sport_mode_version": fw["sport_mode"],
             "firmware": fw}
        if st:
            d["joints_deg"] = list(struct.unpack_from("<12h", st, JOINTS))
            d["body_height_m"] = round(struct.unpack_from("<f", st, F_HEIGHT)[0], 3)
        # 전부 0 인 프레임은 측정값이 아니다 — 판정은 _bms_detail 안에 한 번만 둔다.
        d["battery"] = self._bms_detail(bms)
        return d
