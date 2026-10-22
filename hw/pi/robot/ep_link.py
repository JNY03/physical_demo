"""
피지컬팀 mk2 — RoboMaster EP 내부 링크 (HW-R-01 / HW-R-06)
============================================================
`ControllerLink` 구현체. Go1Link 와 같은 자리에 꽂히고, 위쪽(robot_node·봉투·
spool·순번·임무 4단계)은 기종을 전혀 모른다.

  CONTROLLER_LINK=ep  →  이 파일
  CONTROLLER_LINK=go1 →  go1_link.Go1Link

## 벤더 SDK 는 여기서 끝난다

`robomaster` 패키지 import 는 **이 파일 하나에만** 있다. 위쪽으로 새는 타입이
없으므로 기종 종속은 이 경계를 넘지 않는다. 나중에 ROS 2 로 갈아타더라도
`Ros2Link` 를 하나 더 만들면 되고 이 파일은 그대로 둔다.

## Go1 과 다른 점 — 명령을 연다

Go1Link 는 `send_command()` 를 의도적으로 막아 두었다. sport mode 에서 제어권을
가져오는 순간 서 있던 로봇이 주저앉기 때문이다. EP 는 바퀴형이라 그 위험이 없고,
제어권 개념도 없다. 그래서 명령 경로를 연다 — 다만 안전을 위해 둘을 둔다.

  1) 속도 상한(`EP_MAX_SPEED`) 을 링크에서 클램프한다. 상위가 무엇을 보내든
     이 값을 넘지 않는다.
  2) 워치독. 마지막 명령 후 `EP_CMD_TIMEOUT_S` 안에 갱신이 없으면 정지한다.
     상위가 죽어도 로봇이 계속 달리지 않는다.

## 결측 표현 — 0 으로 채우지 않는다

`common/schema.py` 의 규칙을 그대로 진다. 배터리를 아직 못 받았으면 `None` 이지
`0.0` 이 아니다. 0 을 넣으면 "방전 직전"과 구별되지 않아 배터리 경보가 오발동하고
임무가 `battery_too_low` 로 거부된다(Go1 에서 실제로 겪은 실패다).
"""
import threading
import time

from common import config
from robot.controller_link import ControllerLink, RobotState


class EpLink(ControllerLink):
    """RoboMaster EP 를 SDK 구독으로 읽고, 섀시 속도로 제어한다."""

    def __init__(self, conn_type=None):
        # import 를 __init__ 안에 둔다 — CONTROLLER_LINK 가 ep 가 아닐 때
        # robomaster 패키지가 없어도 노드가 뜬다(go1_link 가 paho 를 다루는 방식과 같다).
        from robomaster import robot as rm_robot

        self.conn_type = conn_type or config.EP_CONN_TYPE
        self._lock = threading.Lock()

        self._pos = None            # (x, y) m
        self._yaw = None            # deg
        self._speed = None          # m/s
        self._battery = None        # %
        self._pos_at = 0.0
        self._att_at = 0.0
        self._vel_at = 0.0
        self._bat_at = 0.0

        self._mission = False       # start_mission 으로 들어온 주행 중인가
        self._last_cmd_at = 0.0
        self._connected = False
        self._version = None
        self._sn = None

        self._ep = rm_robot.Robot()
        self._ep.initialize(conn_type=self.conn_type)
        self._connected = True
        try:
            self._version = self._ep.get_version()
            self._sn = self._ep.get_sn()
        except Exception:
            pass                     # 식별 정보는 있으면 좋고 없어도 동작한다
        print(f"[EP] 접속 — conn_type={self.conn_type} sn={self._sn} fw={self._version}")

        ch = self._ep.chassis
        ch.sub_position(cs=0, freq=config.EP_SUB_FREQ, callback=self._on_position)
        ch.sub_attitude(freq=config.EP_SUB_FREQ, callback=self._on_attitude)
        ch.sub_velocity(freq=config.EP_SUB_FREQ, callback=self._on_velocity)
        self._ep.battery.sub_battery_info(freq=1, callback=self._on_battery)

        self._stop_watchdog = threading.Event()
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    # ---------- SDK 콜백 ----------
    # 콜백 인자 개수가 SDK 버전마다 다를 수 있어 *a 로 받고 필요한 만큼만 쓴다.
    def _on_position(self, *a):
        v = a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a
        if len(v) < 2:
            return
        with self._lock:
            # EP 섀시는 NED 계열이다 - x 전방, **y 오른쪽 +**.
            # 이 코드베이스의 제어기 계약은 ENU 계열 - x 전방, **y 왼쪽 +**.
            # 그래서 여기서 뒤집는다. 어댑터가 계약을 지키게 하는 것이 맞고,
            # 하류(twin_relay_out.to_unity_frame)는 모든 기종이 공유하므로
            # 거기에 EP 사정을 넣으면 Go1 이 틀어진다.
            self._pos = (float(v[0]), -float(v[1]))
            self._pos_at = time.time()

    def _on_attitude(self, *a):
        v = a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a
        if not v:
            return
        with self._lock:
            # yaw, pitch, roll 순. EP 는 **시계 +**, 계약은 **반시계 +** 라 뒤집는다.
            # y 부호와 짝이다 - 둘 중 하나만 뒤집으면 거울상이 되어
            # 직진은 맞는데 회전만 반대로 도는 증상이 난다(실측).
            self._yaw = -float(v[0])
            self._att_at = time.time()

    def _on_velocity(self, *a):
        v = a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a
        if len(v) < 2:
            return
        with self._lock:
            self._speed = (float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5
            self._vel_at = time.time()

    def _on_battery(self, *a):
        v = a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a
        pct = v[0] if isinstance(v, (list, tuple)) else v
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            return
        with self._lock:
            # 0 은 유효한 측정값일 수 있으나 EP 가 미초기화 프레임으로 0 을 내는 것을
            # 구분할 수 없다. Go1 의 전량 0 프레임 사고와 같은 판단으로 버린다.
            self._battery = pct if pct > 0 else None
            self._bat_at = time.time()

    # ---------- 워치독 ----------
    def _watchdog(self):
        """상위가 죽거나 명령이 끊기면 정지시킨다. 바퀴형은 마지막 속도 명령이
        그대로 유지되므로, 이게 없으면 노드가 죽어도 로봇은 계속 달린다."""
        while not self._stop_watchdog.wait(0.1):
            with self._lock:
                stale = (self._mission
                         and time.time() - self._last_cmd_at > config.EP_CMD_TIMEOUT_S)
            if stale:
                print("[EP] 워치독 — 명령 갱신 없음, 정지")
                self._drive(0.0, 0.0, 0.0)
                with self._lock:
                    self._mission = False

    # ---------- ControllerLink 구현 ----------
    def read_state(self) -> RobotState:
        with self._lock:
            pos, yaw, speed, battery = self._pos, self._yaw, self._speed, self._battery
            pos_age = time.time() - self._pos_at if self._pos_at else None
            mission = self._mission

        if pos is None or yaw is None:
            # 아직 한 건도 못 받았다. 값을 지어내지 않는다.
            raise RuntimeError("ep_state_unavailable")

        # 모드는 **명령 사실**로 정한다. 속도로 추정하지 않는다.
        #
        # 처음에는 Go1Link 처럼 속도 임계로 추정했는데, 실측에서 로봇이 진동만 해도
        # 속도 구독값이 임계를 넘어 모드가 계속 뒤집혔다(pi1 + EP, 2026-09-14).
        # 표시만 틀리는 게 아니다 — robot_node 의 heartbeat_enabled() 가 임무 중에는
        # 하트비트를 끄고 state 를 20Hz 로 올리므로, 백엔드는 하트비트가 끊긴 채
        # 20Hz 스트림을 받게 된다.
        #
        # Go1 은 명령 경로가 막혀 있어(제어권 탈취 위험) 추정 말고는 방법이 없었지만,
        # EP 는 명령 경로가 열려 있으니 추정할 이유가 없다. 우리가 시킨 것만 임무다.
        if pos_age is not None and pos_age > config.EP_STALE_S:
            mode = "unknown"
        elif mission:
            mode = "mission"
        else:
            mode = "idle"

        return RobotState(
            battery_pct=battery,      # 모르면 None
            x=round(pos[0], 3), y=round(pos[1], 3),
            heading_deg=round(yaw, 1),
            speed_mps=round(speed, 3) if speed is not None else 0.0,
            mode=mode,
        )

    def send_command(self, action, params) -> None:
        """HW-R-06. 상위의 어휘를 EP 섀시 명령으로 옮긴다."""
        params = params or {}

        if action == "start_mission":
            vx = self._clamp(params.get("vx") or config.EP_MISSION_SPEED)
            vy = self._clamp(params.get("vy") or 0.0)
            wz = float(params.get("wz") or 0.0)
            with self._lock:
                self._mission = True
                self._last_cmd_at = time.time()
            self._drive(vx, vy, wz)
            return

        if action in ("abort_mission", "stop"):
            with self._lock:
                self._mission = False
            self._drive(0.0, 0.0, 0.0)
            return

        if action == "drive":
            # 연속 제어. 워치독을 살리려면 EP_CMD_TIMEOUT_S 안에 계속 불러야 한다.
            vx = self._clamp(params.get("vx") or 0.0)
            vy = self._clamp(params.get("vy") or 0.0)
            wz = float(params.get("wz") or 0.0)
            with self._lock:
                self._mission = abs(vx) + abs(vy) + abs(wz) > 0
                self._last_cmd_at = time.time()
            self._drive(vx, vy, wz)
            return

        raise NotImplementedError(f"ep_action_not_supported: {action}")

    def link_health(self) -> str:
        with self._lock:
            pos_age = time.time() - self._pos_at if self._pos_at else None
            bat_age = time.time() - self._bat_at if self._bat_at else None
            connected = self._connected
        if not connected or pos_age is None:
            return "fault"
        if pos_age > config.EP_STALE_S:
            return "fault"            # 붙어는 있는데 값이 안 온다
        if bat_age is None or bat_age > config.EP_BATTERY_STALE_S:
            return "degraded"         # 자세는 오는데 배터리가 안 온다
        return "ok"

    # ---------- 내부 ----------
    @staticmethod
    def _clamp(v):
        v = float(v)
        lim = config.EP_MAX_SPEED
        return max(-lim, min(lim, v))

    def _drive(self, vx, vy, wz):
        try:
            self._ep.chassis.drive_speed(x=vx, y=vy, z=wz, timeout=config.EP_CMD_TIMEOUT_S)
        except Exception as e:
            print(f"[EP] 구동 명령 실패: {type(e).__name__}: {e}")
            with self._lock:
                self._connected = False

    def diagnostics(self):
        with self._lock:
            now = time.time()
            return {
                "conn_type": self.conn_type,
                "sn": self._sn,
                "fw_version": self._version,
                "connected": self._connected,
                "mission": self._mission,
                "age_s": {
                    "position": round(now - self._pos_at, 2) if self._pos_at else None,
                    "attitude": round(now - self._att_at, 2) if self._att_at else None,
                    "velocity": round(now - self._vel_at, 2) if self._vel_at else None,
                    "battery": round(now - self._bat_at, 2) if self._bat_at else None,
                },
            }

    def close(self):
        self._stop_watchdog.set()
        try:
            self._drive(0.0, 0.0, 0.0)
            self._ep.chassis.unsub_position()
            self._ep.chassis.unsub_attitude()
            self._ep.chassis.unsub_velocity()
            self._ep.battery.unsub_battery_info()
        except Exception:
            pass
        finally:
            try:
                self._ep.close()
            except Exception:
                pass
