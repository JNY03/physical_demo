# -*- coding: utf-8 -*-
"""
피지컬팀 mk2 — teleop · move_relative 검증 (브로커·실물 불필요)
==================================================================
RobotNode 를 그대로 띄우고 규약 봉투를 downlink 로 넣어 uplink 응답과 구동 브리지 쪽에
실제로 흘러간 UDP 를 함께 본다. 브리지는 fake_go1_sdk 를 **다른 포트(25100/25106)**에
띄워 쓰므로 go1-sdk 서비스를 내릴 필요가 없다.

사용: python3 -m bench.teleop_move_test     (pi/ 디렉터리)
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HW_ENTITY_ID", "go1-test")
os.environ.setdefault("HW_NODE_ID", "pitest")
os.environ.setdefault("HW_ZONE_ID", "zoneT")
os.environ["HW_CONTROLLER_LINK"] = "sim"
os.environ["HW_SPOOL_PATH"] = "/tmp/teleop_move_test_spool.jsonl"

from bench.fake_go1_sdk import FakeSdk
from common import physical_command_pb2 as pb
from common.node import BaseNode
from robot import go1_mission
from robot.robot_node import RobotNode

CMD_PORT, ACK_PORT, DEAD_PORT = 25100, 25106, 25199
PB = pb.PhysicalCommandEnvelope


class _Info:
    rc = 0; mid = 1


class FakeClient:
    def __init__(self):
        self.sent = []
        self.lock = threading.Lock()

    def publish(self, t, pl, qos=0, retain=False):
        with self.lock:
            self.sent.append((time.monotonic(), t, pl))
        return _Info()

    def subscribe(self, t, qos=0):
        pass

    def uplink(self, cid=None):
        out = []
        with self.lock:
            items = list(self.sent)
        for _, t, pl in items:
            if not t.startswith("terminal/"):
                continue
            env = PB.FromString(pl)
            which = env.WhichOneof("body")
            body = getattr(env, which)
            if cid is None or getattr(body, "command_id", None) == cid:
                out.append((which, body))
        return out


_n = [0]


def send(node, action, **params):
    _n[0] += 1
    cid = f"t-{_n[0]}"
    env = PB()
    env.command.command_id = cid
    env.command.action = action
    for k, v in params.items():
        env.command.parameters[k] = float(v)
    node.pcmd.on_message(env.SerializeToString())
    return cid


def resend(node, cid, action, **params):
    env = PB()
    env.command.command_id = cid
    env.command.action = action
    for k, v in params.items():
        env.command.parameters[k] = float(v)
    node.pcmd.on_message(env.SerializeToString())


def acceptance(fc, cid):
    acc = [b for w, b in fc.uplink(cid) if w == "acceptance"]
    return acc


def wait_result(fc, cid, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        r = [b for w, b in fc.uplink(cid) if w == "result"]
        if r:
            return r[0]
        time.sleep(0.05)
    raise AssertionError(f"{cid} Result 가 안 왔다")


def wait_until(pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return time.monotonic() - t0
        time.sleep(0.01)
    return None


def ok(msg):
    print(f"  ✓ {msg}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    go1_mission.MissionClient.__init__.__defaults__ = ("127.0.0.1", CMD_PORT, ACK_PORT)
    sdk = FakeSdk(cmd_port=CMD_PORT, ack_port=ACK_PORT, interval=0.6)
    threading.Thread(target=sdk.serve, daemon=True).start()

    BaseNode._connect = lambda self: None
    n = RobotNode()
    n.connected = True
    fc = FakeClient()
    n.client = fc
    n.teleop.addr = ("127.0.0.1", CMD_PORT)
    n.pcmd.start()

    cap = [b for w, b in fc.uplink() if w == "capability"][0]
    assert {"teleop", "move_relative", "move_forward", "turn"} <= set(cap.actions), cap.actions
    ok(f"Capability.actions 에 teleop·move_relative 선언 ({len(cap.actions)}개)")

    # ---------------- teleop ----------------
    print("[teleop]")
    c1 = send(n, "teleop", vx=0.3, vy=0, vyaw=0, hold_ms=500)
    a = acceptance(fc, c1)
    assert len(a) == 1 and a[0].accepted, a
    dt = wait_until(sdk.moving, 0.5)
    assert dt is not None, "누름 → 이동이 안 됐다"
    assert abs(sdk.last_frame[0] - 0.3) < 1e-6
    ok(f"누름 → 이동 ({dt*1000:.0f}ms 안에 브리지가 vx=0.30 수신, MODE 1)")

    # 200ms 마다 같은 값을 새 id 로 — 계속 움직이고, 수락은 첫 건만
    ids = []
    for _ in range(5):
        time.sleep(0.2)
        ids.append(send(n, "teleop", vx=0.3, vy=0, vyaw=0, hold_ms=500))
        assert sdk.moving(), "재전송 중 멈췄다"
    assert all(not acceptance(fc, i) for i in ids), "같은 값인데 Acceptance 를 냈다"
    assert not [w for i in [c1] + ids for w, _ in fc.uplink(i) if w in ("status", "result")]
    ok("200ms 재전송 5건 — 계속 이동, 같은 값은 Acceptance 생략, Status·Result 없음")

    c2 = send(n, "teleop", vx=0.3, vy=0, vyaw=0.6, hold_ms=500)
    assert acceptance(fc, c2) and acceptance(fc, c2)[0].accepted
    time.sleep(0.1)
    assert abs(sdk.last_frame[2] - 0.6) < 1e-6
    ok("조합 변경(vyaw=+0.6, 반시계) → Acceptance 1건, 브리지 wz=+0.60")

    resend(n, c2, "teleop", vx=0.3, vy=0, vyaw=0.6, hold_ms=500)
    assert len(acceptance(fc, c2)) == 2
    ok("같은 command_id 재수신 → 재실행 없이 이전 Acceptance 재송신(ALREADY_EXISTS 아님)")

    c3 = send(n, "teleop", vx=0, vy=0, vyaw=0, hold_ms=500)
    dt = wait_until(lambda: not sdk.moving(), 0.3)
    assert dt is not None and dt < 0.25, dt
    assert wait_until(lambda: sdk.mode == 0, 0.2) is not None
    ok(f"뗌(0,0,0) → 즉시 정지 ({dt*1000:.0f}ms, 정지 프레임 + MODE 0)")

    # 유지 신호 끊김: 마지막 teleop 뒤 hold_ms 안에 선다
    for hold in (500, None):
        p = dict(vx=0.2, vy=0.1, vyaw=0)
        if hold:
            p["hold_ms"] = hold
        send(n, "teleop", **p)
        wait_until(sdk.moving, 0.3)
        t_last = time.monotonic()
        dt = wait_until(lambda: not sdk.moving(), 2.0)
        assert dt is not None
        total = time.monotonic() - t_last
        assert 0.45 <= total <= 0.75, total
        ok(f"화면 꺼짐(hold_ms={hold}) → 마지막 수신 뒤 {total*1000:.0f}ms 에 정지 "
           f"(노드 hold 500 + 브리지 0.15s)")

    send(n, "teleop", vx=0.2, vy=0, vyaw=0, hold_ms=5000)
    wait_until(sdk.moving, 0.3)
    t_last = time.monotonic()
    wait_until(lambda: not sdk.moving(), 3.0)
    total = time.monotonic() - t_last
    assert total <= 1.25, total
    ok(f"hold_ms=5000 → 상한 1000ms 로 잘림 ({total*1000:.0f}ms 에 정지)")

    send(n, "teleop", vx=2.0, vy=-1.0, vyaw=3.0, hold_ms=500)
    time.sleep(0.15)
    assert sdk.last_frame[:3] == (0.4, -0.3, 1.0), sdk.last_frame
    ok(f"범위 밖 속도 → 잘림 (2.0,-1.0,3.0 → {sdk.last_frame[:3]})")
    send(n, "teleop", vx=0, vy=0, vyaw=0)
    time.sleep(0.1)

    c = send(n, "teleop", vx=float("nan"), vy=0, vyaw=0)
    a = acceptance(fc, c)[0]
    assert not a.accepted and a.rejection.code == "INVALID_ARGUMENT", a
    ok(f"NaN → 거절 {a.rejection.code}/{a.rejection.message}")

    # 브로커 단절 → 즉시
    send(n, "teleop", vx=0.3, vy=0, vyaw=0, hold_ms=1000)
    wait_until(sdk.moving, 0.3)
    n._on_disconnect(None, None, None, 0, None)
    dt = wait_until(lambda: not sdk.moving(), 1.0)
    assert dt is not None and dt < 0.25, dt
    n.connected = True
    ok(f"브로커 단절 → hold 를 기다리지 않고 {dt*1000:.0f}ms 에 정지")

    # abort
    send(n, "teleop", vx=0.3, vy=0, vyaw=0, hold_ms=1000)
    wait_until(sdk.moving, 0.3)
    ca = send(n, "abort")
    dt = wait_until(lambda: not sdk.moving(), 1.0)
    r = wait_result(fc, ca)
    assert dt is not None and dt < 0.25 and r.result["had_teleop"] == 1.0
    ok(f"abort → {dt*1000:.0f}ms 에 정지, Result had_teleop=1")
    c = send(n, "teleop", vx=0.3, vy=0, vyaw=0, hold_ms=500)   # 키를 누른 채 계속
    a = acceptance(fc, c)[0]
    assert not a.accepted and a.rejection.message == "teleop_halted_by_abort", a
    time.sleep(0.2)
    assert not sdk.moving()
    ok(f"abort 직후 계속 오는 teleop → 거절 {a.rejection.code}/{a.rejection.message}")
    send(n, "teleop", vx=0, vy=0, vyaw=0)
    c = send(n, "teleop", vx=0.3, vy=0, vyaw=0, hold_ms=500)
    assert acceptance(fc, c)[0].accepted
    ok("0 을 받은 뒤(키를 뗌) 다시 누르면 수락")

    # ---------------- 겹침: teleop → 임무 ----------------
    print("[겹침]")
    wait_until(sdk.moving, 0.3)
    cm = send(n, "move_relative", dx_m=-1.0, dy_m=0)
    assert acceptance(fc, cm)[0].accepted
    dt = wait_until(lambda: not sdk.moving(), 0.3)
    assert dt is not None
    ok(f"teleop 중 move_relative → teleop 접고({dt*1000:.0f}ms) 임무 수락")
    for p in (dict(vx=0.3, vy=0, vyaw=0), dict(vx=0, vy=0, vyaw=0)):
        c = send(n, "teleop", hold_ms=500, **p)
        a = acceptance(fc, c)[0]
        assert not a.accepted and a.rejection.code == "FAILED_PRECONDITION" \
            and a.rejection.message == "mission_in_progress", a
    ok("임무 중 teleop(0 포함) → 거절 FAILED_PRECONDITION/mission_in_progress — 0 이 임무를 세우지 않음")
    c = send(n, "move_forward", distance_m=1.0)
    a = acceptance(fc, c)[0]
    assert not a.accepted and a.rejection.message == "mission_in_progress"
    c = send(n, "move_relative", dx_m=0, dy_m=1.0)
    a = acceptance(fc, c)[0]
    assert not a.accepted and a.rejection.message == "mission_in_progress"
    ok("이동 중 move_forward·move_relative → 거절 FAILED_PRECONDITION/mission_in_progress")

    # ---------------- move_relative ----------------
    print("[move_relative]")
    r = wait_result(fc, cm)
    assert r.status == pb.SUCCEEDED, r
    st = [b.detail for w, b in fc.uplink(cm) if w == "status"]
    assert r.result["moved_dx_m"] == -1.0 and r.result["moved_dy_m"] == 0.0
    assert r.result["src_odo"] == 1.0 and r.result["reached"] == 1.0
    ok(f"뒤로 1m: Acceptance → Status {len(st)}건 → Result SUCCEEDED {dict(r.result)}")

    for dx, dy, name in ((0, 1.0, "왼쪽 1m"), (0, -1.0, "오른쪽 1m"), (0.5, 0.5, "대각선")):
        c = send(n, "move_relative", dx_m=dx, dy_m=dy, v_mps=0.2)
        r = wait_result(fc, c)
        assert r.status == pb.SUCCEEDED and r.result["moved_dy_m"] == dy, r
        ok(f"{name}: Result moved_dx_m={r.result['moved_dx_m']:+.2f} "
           f"moved_dy_m={r.result['moved_dy_m']:+.2f}")

    c = send(n, "move_relative", dx_m=-1.0, dy_m=0)
    time.sleep(0.2)
    send(n, "abort")
    r = wait_result(fc, c)
    assert r.status == pb.ABORTED and r.failure.code == "ABORTED", r
    ok(f"이동 중 abort → Result ABORTED ({r.failure.code}/{r.failure.message})")
    send(n, "teleop", vx=0, vy=0, vyaw=0)   # abort 걸쇠 풀기

    bad = [(dict(dx_m=0, dy_m=0), "distance_required"),
           (dict(dx_m=0.01, dy_m=0), "dx_m_out_of_range"),
           (dict(dx_m=0, dy_m=-11), "dy_m_out_of_range"),
           (dict(dx_m=1, dy_m=0, v_mps=0.5), "v_mps_out_of_range"),
           (dict(dx_m=1, dy_m=0, v_mps=0.01), "v_mps_out_of_range")]
    for p, why in bad:
        a = acceptance(fc, send(n, "move_relative", **p))[0]
        assert not a.accepted and a.rejection.message == why, (p, a)
    ok("범위 밖 move_relative → INVALID_ARGUMENT 거절 (" +
       ", ".join(w for _, w in bad) + ")")

    # ---------------- 브리지 없음 + sdk_auto 꺼짐 ----------------
    print("[브리지 없음]")
    go1_mission.MissionClient.__init__.__defaults__ = ("127.0.0.1", DEAD_PORT, ACK_PORT)
    n.sdk_autostart = False
    for action, p in (("teleop", dict(vx=0.3, vy=0, vyaw=0)),
                      ("move_relative", dict(dx_m=1.0, dy_m=0))):
        a = acceptance(fc, send(n, action, **p))[0]
        assert not a.accepted and a.rejection.message == "go1_sdk_not_running", a
    ok("sdk_auto 꺼짐 + 브리지 없음 → teleop·move_relative 모두 "
       "FAILED_PRECONDITION/go1_sdk_not_running")

    print("전부 통과", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
