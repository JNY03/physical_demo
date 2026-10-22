# -*- coding: utf-8 -*-
"""
피지컬팀 mk2 — 규약 teleop → 구동 브리지 속도 스트림 (HW-R-06)
==================================================================
관제 화면의 키보드 수동 조작(`teleop`)을 구동 브리지(go1_sdk_pc)의 텔레옵 입구로 옮긴다.

## 왜 이 노드가 스트림을 다시 만드나

화면은 200ms 마다 같은 값을 다시 보내지만, 브리지는 텔레옵 프레임이 **0.15초** 안 오면
선다(unity_timeout_sec). 화면 주기를 그대로 흘리면 로봇이 걷다 서다를 되풀이한다.
그래서 여기서 마지막 값을 붙들고 20Hz 로 다시 낸다. 붙드는 시한이 hold_ms 다 —
이 시한 안에 다음 teleop 이 안 오면 프레임을 끊고 정지 프레임을 보낸다.
**화면이 죽거나 줄이 끊겨도 로봇이 계속 걷지 않게 하는 것이 이 모듈의 핵심이다.**
이 판정은 MQTT 와 무관한 스레드가 단조 시계로 하므로 브로커가 멎어도 돈다.

  보냄 127.0.0.1:15100   "MODE 1"                세션 시작 + 1초마다(다른 쪽이 MODE 0 으로
                                                 돌려 놓았거나 브리지가 막 떴을 때를 덮는다)
                         "<vx> <vy> <wz> 0"      20Hz
                         "0.000 0.000 0.000 0" ×3, "MODE 0"   세션 끝

축은 ROS REP-103 그대로다(브리지 프레임과 같다): vx 앞 +, vy 왼쪽 +, wz 반시계 +.
"""
import math
import socket
import threading
import time

RATE_HZ = 20.0
MODE_REASSERT_S = 1.0

# 안전 상한. 브리지 자체 상한(V/S 0.4, W 2.0)보다 좁게 둔다 — 화면이 보내는 값
# (±0.3 / ±0.2 / ±0.6)에 여유를 준 정도다. 넘는 값은 **잘라서** 쓴다(스트림이라
# 한 건 거절하면 그 사이 로봇이 멎었다 다시 걷는다).
VX_MAX = 0.40
VY_MAX = 0.30
WZ_MAX = 1.00

HOLD_DEFAULT_MS = 500.0
HOLD_MIN_MS = 250.0       # 화면 재전송 주기(200ms)보다 짧으면 걷다 서다를 되풀이한다
HOLD_MAX_MS = 1000.0
ZERO_EPS = 1e-3


def hold_seconds(hold_ms):
    """hold_ms 해석. 없거나 이상하면(비수치·0 이하) 500ms, 그 밖엔 [250, 1000]ms 로 자른다."""
    if hold_ms is None or not math.isfinite(hold_ms) or hold_ms <= 0:
        hold_ms = HOLD_DEFAULT_MS
    return min(HOLD_MAX_MS, max(HOLD_MIN_MS, hold_ms)) / 1000.0


def clamp_velocity(vx, vy, wz):
    def c(v, lim):
        return max(-lim, min(lim, v))
    return c(vx, VX_MAX), c(vy, VY_MAX), c(wz, WZ_MAX)


class TeleopStreamer:
    def __init__(self, host="127.0.0.1", port=15100, log=print):
        self.addr = (host, port)
        self.log = log
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._target = (0.0, 0.0, 0.0)
        self._deadline = 0.0          # time.monotonic() 기준. 이 시각이 지나면 선다
        self._session = False         # MODE 1 을 보내 프레임을 흘리는 중인가
        self._end_reason = None       # 다음 루프에서 세션을 끝낼 사유
        self.last_reason = None       # 마지막 세션이 끝난 사유(diag 용)
        threading.Thread(target=self._run, name="teleop", daemon=True).start()

    # ---------- 노드가 부르는 쪽 ----------
    @property
    def active(self):
        with self._lock:
            return self._session or self._deadline > time.monotonic()

    def command(self, vx, vy, wz, hold_s):
        """새 목표 속도. 첫 건이거나 값이 바뀌었으면 True(→ Acceptance 를 낸다)."""
        target = (round(vx, 3), round(vy, 3), round(wz, 3))
        with self._lock:
            fresh = not (self._session or self._deadline > time.monotonic())
            changed = fresh or target != self._target
            self._target = target
            self._deadline = time.monotonic() + hold_s
            self._end_reason = None
        self._wake.set()
        return changed

    def stop(self, reason):
        """지금 선다. 세션 중이 아니어도 부를 수 있다(정지 프레임은 세션 중일 때만 나간다)."""
        with self._lock:
            was = self._session or self._deadline > time.monotonic()
            self._target = (0.0, 0.0, 0.0)
            self._deadline = 0.0
            if was:
                self._end_reason = reason
        self._wake.set()
        return was

    # ---------- 송출 스레드 ----------
    def _send(self, text):
        try:
            self._tx.sendto(text.encode(), self.addr)
        except OSError:
            pass                      # 브리지가 없으면 받는 쪽이 없을 뿐이다

    def _run(self):
        period = 1.0 / RATE_HZ
        mode_at = 0.0
        while True:
            self._wake.wait(period)
            self._wake.clear()
            now = time.monotonic()
            with self._lock:
                target, deadline = self._target, self._deadline
                reason = self._end_reason
                live = deadline > now
                if self._session and not live and reason is None:
                    reason = "hold_expired"
                start = live and not self._session
                end = self._session and not live
                if start:
                    self._session = True
                if end:
                    self._session = False
                    self._end_reason = None
                    self.last_reason = reason
            if start:
                self._send("MODE 1")
                mode_at = now
                self.log(f"[텔레옵] 시작 vx={target[0]:+.2f} vy={target[1]:+.2f} "
                         f"wz={target[2]:+.2f}")
            if end:
                for _ in range(3):
                    self._send("0.000 0.000 0.000 0")
                    time.sleep(0.01)
                self._send("MODE 0")
                self.log(f"[텔레옵] 정지 ({reason})")
                continue
            if not live:
                continue
            if now - mode_at >= MODE_REASSERT_S:
                self._send("MODE 1")
                mode_at = now
            self._send("%.3f %.3f %.3f 0" % target)
