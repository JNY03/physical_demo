# -*- coding: utf-8 -*-
"""트윈 출구 — 로봇 상태를 다중 로봇 릴레이로 한 줄 보낸다.

**기종을 가리지 않는다.** Go1 이든 EP 든 sim 이든 `robot_node` 가 상태를 발행할 때
이 함수를 부르면 된다. 기종별 피더를 따로 만들지 않는 이유다 — 만들면 기종이
늘 때마다 같은 코드가 한 벌씩 늘고, 좌표 규약이 그만큼 갈라진다.

    robot_node ──► 이 모듈 ──(15200)──► robot_state_relay ──(15201)──► Unity

주기는 **업무 평면과 분리**돼 있다. MQTT state 는 대기 중 5초에 한 번이지만
트윈은 기본 10Hz 다. 5초 주기로 화면을 그리면 순간이동처럼 보인다.
`HW_TWIN_RELAY_HZ` 로 바꾼다.

## 보내는 형식

`robot_state_relay.py:42-44` 의 「피더 축약형」 그대로다. 릴레이는 손대지 않는다.

    <robot_type> <robot_id> <seq> <tms> <x> <z> <yaw> <vx> <vy> <wz> <estop> <mode>

`node_id` 는 **릴레이가 찍는다.** 발신자가 무엇을 주장하든 릴레이가 덮어쓴다 —
어느 파이를 거쳐 왔는지는 그 파이 자신만 확실히 안다. 그래서 여기서 보내지 않는다.

## 로봇을 어떻게 구별하나

릴레이를 지나면 `(node_id, robot_type, robot_id)` 가 된다. 세 칸이 모두 있어야
같은 기종 같은 번호가 다른 파이에 붙어 있어도 안 섞인다.

    robot_type <- HW_DEVICE_TYPE   (go1_robot, robomaster_ep, ...)
    robot_id   <- HW_ENTITY_ID     (go1-001, ep-1, ...)

둘 다 **파이의 기존 설정값**이다. 새 설정을 만들지 않았다.

## 끄는 법

`HW_TWIN_RELAY_HOST` 가 비어 있으면 **아무것도 하지 않는다.** 기본이 비어 있으므로
설정을 주지 않은 노드는 지금과 똑같이 동작한다. Go1 은 이미 `go1_sdk_pc` 가
릴레이로 쏘고 있으므로 이 값을 주지 않는다 - 주면 같은 로봇이 두 번 올라간다.

## 좌표 변환은 여기서만 한다

`RobotState` 는 제어기 규약(전진 +x, 왼쪽 +y, 반시계 +yaw)이고
릴레이 아래쪽은 Unity 규약(전진 +z, 오른쪽 +x, 시계 +yaw)이다.
그 변환이 여러 곳에 흩어지면 방향이 틀어졌을 때 찾을 데가 여러 곳이 된다 -
Go1 에서 다섯 군데에 흩어져 있었고, 한 번 고칠 때마다 다른 데가 틀어졌다.
"""
import math
import os
import socket
import time

_DEFAULT_PORT = 15200

# 릴레이가 읽는 mode 어휘. Unity 는 이 값으로 정지/이동을 가른다.
_MODE_IDLE = 1
_MODE_MOVE = 2

_sock = None
_seq = 0
_warned = False
_last_sent = 0.0


def _enabled_host():
    host = os.environ.get("HW_TWIN_RELAY_HOST", "").strip()
    return host or None


def _port():
    try:
        return int(os.environ.get("HW_TWIN_RELAY_PORT", _DEFAULT_PORT))
    except ValueError:
        return _DEFAULT_PORT


def _hz():
    """트윈 전송 주기(Hz).

    **업무 평면과 일부러 다르게 둔다.** MQTT state 는 대기 중 5초에 한 번인데
    (ROBOT_STATE_INTERVAL_IDLE), 그 주기로 화면을 그리면 로봇이 5초마다
    순간이동하는 것처럼 보인다. 두 평면은 목적이 다르다 —
    업무 기록은 드물어도 되지만 화면은 이어져 보여야 한다.

    상한은 내부 수집 주기(EP_SUB_FREQ=10Hz)다. 그보다 빨리 보내 봐야
    같은 표본을 두 번 보내는 것이라 대역만 쓴다.
    """
    try:
        hz = float(os.environ.get("HW_TWIN_RELAY_HZ", 10.0))
    except ValueError:
        hz = 10.0

    return max(0.2, min(50.0, hz))


def to_unity_frame(x, y, heading_deg):
    """제어기 규약 -> Unity 규약. **바꾸는 곳은 여기 하나다.**

        제어기: 전진 +x, 왼쪽 +y, yaw 반시계 +
        Unity : 전진 +z, 오른쪽 +x, yaw 시계 +

    방향이 틀어지면 이 세 줄만 고친다. 다른 파일에 보정을 흩뿌리지 말 것.
    """
    unity_x = -y
    unity_z = x
    unity_yaw = math.radians(-heading_deg)
    return unity_x, unity_z, unity_yaw


def send_state(robot_type, robot_id, state, estop=0):
    """상태 한 장을 릴레이로 보낸다.

    `HW_TWIN_RELAY_HOST` 가 없으면 **조용히 아무것도 하지 않는다** -
    이 경로를 안 쓰는 노드가 로그로 시끄러워지지 않게.

    실패해도 예외를 올리지 않는다. 트윈 표시가 안 된다고 로봇 노드가
    멈추면 안 된다 - 업무 평면이 본체이고 이건 곁가지다.
    """
    global _sock, _seq, _warned, _last_sent

    host = _enabled_host()
    if host is None:
        return False

    # 자체 주기로 솎는다. 호출부(on_tick)는 50Hz 로 부르고 여기서 거른다 —
    # 그래야 호출부가 트윈 주기를 몰라도 된다.
    now = time.monotonic()
    if now - _last_sent < (1.0 / _hz()):
        return False
    _last_sent = now

    if not robot_type or not robot_id:
        if not _warned:
            print("[twin-out] robot_type/robot_id 가 비어 있다 - 보내지 않는다. "
                  "HW_DEVICE_TYPE / HW_ENTITY_ID 를 확인하라.")
            _warned = True
        return False

    try:
        if _sock is None:
            _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        x, z, yaw = to_unity_frame(state.x, state.y, state.heading_deg)

        # 속도는 제어기가 크기만 준다(speed_mps). 방향 성분을 지어내지 않는다 -
        # 0 으로 채우면 "정지"로 읽히므로 전진 성분에만 싣는다.
        vx = float(state.speed_mps or 0.0)
        moving = abs(vx) > 1e-3
        mode = _MODE_MOVE if moving else _MODE_IDLE

        _seq += 1
        line = ("%s %s %d %.1f %.4f %.4f %.4f %.3f %.3f %.3f %d %d"
                % (robot_type, robot_id, _seq, time.time() * 1000.0,
                   x, z, yaw, vx, 0.0, 0.0, int(estop), mode))

        _sock.sendto(line.encode("ascii"), (host, _port()))

        # HW_TWIN_RELAY_DEBUG=1 이면 변환의 **양쪽**을 한 줄에 찍는다.
        # 방향이 틀어졌을 때 부호를 추측으로 뒤집지 않기 위해서다 -
        # 어느 축이 어디로 갔는지는 눈이 아니라 숫자가 말해야 한다.
        if os.environ.get("HW_TWIN_RELAY_DEBUG", "").strip() not in ("", "0"):
            print("[twin-out] 제어기 x=%+.3f y=%+.3f h=%+.1f도  ->  "
                  "Unity x=%+.3f z=%+.3f yaw=%+.1f도"
                  % (state.x, state.y, state.heading_deg,
                     x, z, math.degrees(yaw)), flush=True)

        return True

    except OSError as e:
        if not _warned:
            print("[twin-out] 전송 실패 (%s) - 트윈 표시만 영향, 업무는 계속한다" % e)
            _warned = True
        return False


def describe():
    host = _enabled_host()
    if host is None:
        return "twin-out: 꺼짐 (HW_TWIN_RELAY_HOST 없음)"
    return "twin-out: %s:%d @%.0fHz" % (host, _port(), _hz())
