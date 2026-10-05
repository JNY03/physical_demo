"""
피지컬팀 mk2 — 로봇 제어기 내부 링크 (HW-R-01 / HW-R-06)
=========================================================
로봇 제어기(구동부)와 온보드 라즈베리파이 사이의 내부 링크를 추상화한다.

역할 분리가 목적이다. 외부 통신(MQTT)·버퍼링·중계는 온보드가 전담하고 제어기는
주행·임무 수행에 연산 자원을 집중한다. 그리고 외부 프로토콜과 구동 인터페이스를
온보드에서 갈라놓아 **통신 장애가 구동 로직에 직접 닿지 않게** 한다(HW-R-06).

제어기 실물이 미정이므로 인터페이스만 고정하고 구현체를 갈아끼운다.

  SimLink   실물 없이 파이프라인을 검증한다. 지금 사용
  Go1Link   Unitree Go1. 내부 MQTT 구독 전용(명령 차단)
  EpLink    RoboMaster EP. SDK 구독 + 섀시 속도 제어
  CanLink   소형 주기 스칼라(배터리·위치·속도·모드). 결정적 지연·프레임 우선순위
  EthLink   가변 크기 인식 결과. 구조화 데이터

소형 주기값은 CAN, 인식 결과는 Ethernet 으로 나누는 하이브리드를 기준으로 둔다
(SRS 9.4). 공유메모리는 제어기와 파이가 물리적으로 별개 보드라 적용되지 않는다.
"""
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class RobotState:
    """제어기가 50Hz로 올려보내는 내부 상태 (HW-R-01).

    ## 앞의 여섯 칸은 건드리지 않는다

    기종을 가리지 않는 최소 공통분모이고, **가시화 웹과 서버가 이미 이 이름을 읽고
    있다**(docs/web-integration.md §2). 이름도 의미도 바꾸지 않는다.

    ## 뒤의 칸들은 기종이 주면 채우고 안 주면 None 이다

    Go1 은 내부 MQTT 84바이트에 자세·관절·속도벡터를 이미 다 실어 보내는데 예전에는
    여섯 칸만 남기고 버렸다. `speed_mps` 하나로 뭉치는 순간 **방향이 사라져** 뒤로
    가는 것과 앞으로 가는 것을 구별할 수 없고, `heading_deg` 만 남기면 로봇이 기울어진
    것을 알 수 없다. 버린 쪽이 손해였다 — 담긴 정보가 많을수록 쓸 데가 넓어진다.

    전부 기본값이 있어서 **다른 구현체(SimLink/EpLink)는 손대지 않아도 그대로 돈다.**
    모르면 0 이 아니라 None 이다(common/schema.py 결측 표현 규칙).
    """
    battery_pct: float | None      # None = 아직 모른다(0% 와 구별해야 한다)
    x: float
    y: float
    heading_deg: float
    speed_mps: float
    mode: str                 # idle | mission | fault

    # --- 자세. heading_deg 는 yaw 하나뿐이라 기울기를 못 봤다 ---
    roll_deg: float | None = None
    pitch_deg: float | None = None
    body_height_m: float | None = None      # 엎드림/기립 판정의 실제 근거값

    # --- 속도 벡터. speed_mps 는 크기뿐이라 방향이 사라졌다 ---
    vx: float | None = None                 # 전진 +
    vy: float | None = None                 # 좌측 +
    yaw_rate: float | None = None           # 회전 각속도

    # --- 관절 12개 (4다리 × hip·thigh·calf), 도 ---
    joints_deg: list | None = None

    # --- 배터리 상세. 잔량 하나로는 남은 시간도 이상도 알 수 없다 ---
    battery_current_ma: int | None = None   # 음수 = 방전
    battery_cycles: int | None = None
    battery_voltage_v: float | None = None
    battery_temp_c: list | None = None
    battery_cells_mv: list | None = None    # 실제로 물린 셀만. 빈 칸은 싣지 않는다
    battery_status: int | None = None

    # --- 값의 나이. 이게 없으면 멈춘 값을 최신값으로 그린다(§6 기지 문제) ---
    state_age_s: float | None = None
    bms_age_s: float | None = None

    # --- 해독하지 못한 바이트. **이름을 붙이지 않는다** ---
    # 이름을 붙이면 받는 쪽이 확정값으로 쓴다. hex 로만 실어서 나중에 서버에 쌓인
    # 기록으로 해독할 수 있게 남긴다 — 지금 버리면 그 기회가 영구히 사라진다.
    raw_unknown: str | None = None

    def as_dict(self):
        return asdict(self)


class ControllerLink(ABC):
    """구현체는 이 셋만 채우면 된다."""

    @abstractmethod
    def read_state(self) -> RobotState:
        """내부 수집. 외부 전송 주기보다 빠르게 돌려 전송 시점에 가장 최신 표본을
        고를 수 있게 한다 (HW-R-01)."""

    @abstractmethod
    def send_command(self, action: str, params: dict) -> None:
        """수신한 서브태스크·제어 명령을 제어기가 실행 가능한 형태로 변환·전달 (HW-R-06)."""

    @abstractmethod
    def link_health(self) -> str:
        """ok | degraded | fault — 내부 링크 자체의 건강 상태."""


class SimLink(ControllerLink):
    """실물 없이 상위 파이프라인(보고 주기·버퍼링·명령 4단계)을 검증하기 위한 구현체.
    제어기가 확보되면 CanLink/EthLink 로 교체하고 이 파일 밖은 손대지 않는다."""

    def __init__(self):
        self.t0 = time.time()
        self.mode = "idle"
        self.battery = 100.0
        self.x = self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.last_cmd = None

    def read_state(self):
        now = time.time()
        dt = 0.02
        if self.mode == "mission":
            # 원을 그리며 주행하는 것으로 둔다 — 위치·속도·방위가 연속적으로 변해야
            # 다운샘플 재전송(SDD 5.4)이 실제로 의미 있는 검증이 된다.
            self.speed = 0.8
            self.heading = (self.heading + 12.0 * dt) % 360.0
            rad = math.radians(self.heading)
            self.x += self.speed * dt * math.cos(rad)
            self.y += self.speed * dt * math.sin(rad)
            self.battery -= 0.02 * dt          # 임무 중 소모
        else:
            self.speed = 0.0
            self.battery -= 0.002 * dt
        self.battery = max(0.0, self.battery)
        return RobotState(
            battery_pct=round(self.battery, 2),
            x=round(self.x, 3), y=round(self.y, 3),
            heading_deg=round(self.heading, 1),
            speed_mps=round(self.speed + random.uniform(-0.01, 0.01), 3),
            mode=self.mode,
        )

    def send_command(self, action, params):
        self.last_cmd = (action, params)
        if action == "start_mission":
            self.mode = "mission"
        elif action in ("abort_mission", "stop"):
            self.mode = "idle"

    def link_health(self):
        return "ok"


def create(kind="sim"):
    if kind == "sim":
        return SimLink()
    if kind == "go1":
        from robot.go1_link import Go1Link      # paho 의존을 여기서만 진다
        return Go1Link()
    if kind == "ep":
        from robot.ep_link import EpLink        # robomaster SDK 의존을 여기서만 진다
        return EpLink()
    raise NotImplementedError(
        f"ControllerLink '{kind}' 미구현 — CanLink/EthLink 는 제어기 실물 확보 후")
