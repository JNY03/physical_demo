"""
피지컬팀 mk2 — pi1 가시화 중계 (viz-nav-feed)
===============================================
유니티가 Go1 을 모는 「문 앞까지 자율주행」에서, pi1 이 받은 것을 관제 웹(가시화)이
볼 수 있게 로컬 브로커(localhost:1883, 브라우저는 9001 websockets)로 옮긴다.

**보기만 한다.** 로봇·브리지·유니티 포트로 아무것도 보내지 않고, 어떤 포트도 bind 하지 않는다.
이 프로세스가 죽어도 유니티 ↔ go1_sdk_pc 경로는 그대로다.

## 원천

| 값 | 어디서 | 왜 |
|---|---|---|
| 경로 사건 (`path_received`·`path_cancel`·`path_done`) | `journalctl -u go1-sdk` 의 브리지 로그 줄 | 브리지가 **실제로 한 일**이다. 패킷을 세면 브리지가 버린 경로(한 루프의 마지막만 씀·파싱 실패)까지 사건이 되고, 500 Hz 흐름 속 mode 99 한 개를 놓치면 경로가 영영 안 끝난다 |
| (위의 대안) | 15110·15101 수동 캡처 | journal 을 못 읽을 때만. **한 프로세스는 한 원천만 쓴다** — 섞으면 같은 경로가 두 번 나간다 |
| `cancel_ack` | 15101 mode == 98 | 현 go1_sdk_pc 는 98 을 보내지 않는다 — 로직만 있고 지금은 나오지 않는다 |
| `estop` | 15100 텔레옵 4번째 필드 0→1 | 15101 의 estop 필드는 `cmd.mode==1`(서 있음)이라 쓰지 않는다 |
| `yaw_deg` | 15101 yaw(rad) → 도, 없으면 Go1 IMU | 유니티가 경로를 계산하는 좌표계의 값이 우선 |
| `yaw_odometry_deg` | Go1 내부 MQTT `robot/state` | 브리지 yaw 는 경로마다 offset 이 다시 맞춰져 IMU 와 나란히 보인다 |
| `battery_pct` | Go1 내부 MQTT `bms/state` | robot-node 는 엣지 브로커로 발행해 로컬에 없다 |
| `moving` | 5009 JSON 중 `state_change`·`motion_active` 가 둘 다 있는 것 | 같은 포트로 탐지 JSON 도 나간다 |

## 수동 캡처

`AF_PACKET` + `SOCK_DGRAM`(링크 헤더를 커널이 떼어 준다 → wlan0·lo·tun 모두 IP 부터) 에
커널 BPF 필터(IPv4·UDP·목적지 포트)를 붙인다. 소켓을 포트에 bind 하지 않으므로 브리지 수신을
빼앗지 않는다. `CAP_NET_RAW` 필요.

Go1 내부 MQTT 는 **구독만** 한다. `Go1Link` 인스턴스는 만들지 않는다 — client_id 가 초 단위라
robot-node 와 같은 초에 뜨면 서로의 접속을 끊는다. 해석 상수·규칙만 가져다 쓴다.
"""
import ctypes
import json
import os
import socket
import struct
import subprocess
import threading
import time

import paho.mqtt.client as mqtt

from common import config
from common.schema import iso_now
from robot import go1_link
from robot.go1_link import Go1Link


def _env(name, default):
    return os.environ.get(name, default)


ZONE = _env("HW_ZONE_ID", "zoneA")
ENTITY = _env("HW_ENTITY_ID", "go1-001")
NODE = _env("HW_NODE_ID", "pi1")

BROKER_HOST = _env("VIZ_BROKER_HOST", "localhost")
BROKER_PORT = int(_env("VIZ_BROKER_PORT", "1883"))
GO1_HOST = _env("VIZ_GO1_MQTT_HOST", config.GO1_MQTT_HOST)
GO1_CLIENT_ID = _env("VIZ_GO1_CLIENT_ID", "viz-nav-feed-pi1")

# 캡처 포트 — §4 시험에서 비어 있는 포트로 바꾼다
PATH_PORT = int(_env("VIZ_PATH_PORT", "15110"))
STATE_PORT = int(_env("VIZ_STATE_PORT", "15101"))
TELEOP_PORT = int(_env("VIZ_TELEOP_PORT", "15100"))
MOTION_PORT = int(_env("VIZ_MOTION_PORT", "5009"))

# auto: journal 을 읽을 수 있으면 journal, 아니면 capture
PATH_EVENT_SOURCE = _env("VIZ_PATH_EVENT_SOURCE", "auto")
JOURNAL_MATCH = _env("VIZ_JOURNAL_MATCH", "-u go1-sdk").split()
RCVBUF = int(_env("VIZ_RCVBUF", str(4 * 1024 * 1024)))

STATE_PERIOD_S = 0.5
BRIDGE_STALE_S = 1.0      # 15101 은 500 Hz — 1초 끊기면 모르는 값
MOTION_STALE_S = 2.0      # 5009 하트비트 0.3 s
SILENT_WARN_S = float(_env("VIZ_SILENT_WARN_S", "10"))

SCHEMA = "viz-nav/1"
BASE = f"{ZONE}/robot/{ENTITY}"
T_STATE = f"{BASE}/nav_state"
T_EVENT = f"{BASE}/nav_event"

LOG_ACTIVATED = "[PATH] activated id="
LOG_CANCEL = "[PATH] PATH_CANCEL received"
LOG_DONE = "[PATH] done notify sent (mode=99)"


def log(*a):
    print("[viz-nav]", *a, flush=True)


def now_ms():
    return int(time.time() * 1000)


def wrap180(deg):
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


class Feed:
    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        # 경로 상태 (사건으로만 바뀐다)
        self.path_active = False
        self.path_id = None          # 지금 따라가는 경로
        self.last_path_id = None     # 가장 최근 path_received 의 id
        # 캡처 값 + 받은 시각(monotonic)
        self.bridge_yaw_deg, self.bridge_at = None, 0.0
        self.last_mode = None
        self.moving, self.moving_at = None, 0.0
        self.teleop_estop = None
        self.chunk_pid, self.chunk_total, self.chunk_parts = -1, 0, {}
        # journal 문구가 바뀌면 경로 사건이 오류 없이 끊긴다 — 그 낌새를 로그로만 남긴다
        self.journal_mode = False
        self.journal_events = 0
        self.motion_since = None
        self.silent_warned = False
        # Go1 내부 MQTT
        self.go1_state, self.go1_state_at = None, 0.0
        self.go1_bms, self.go1_bms_at = None, 0.0

        self.pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                               client_id=f"viz-nav-feed-pub-{NODE}",
                               protocol=mqtt.MQTTv5)
        self.pub.max_queued_messages_set(200)
        self.pub.reconnect_delay_set(1, 10)
        self.pub.on_connect = lambda c, u, f, rc, p=None: log(f"로컬 브로커 접속 rc={rc}")
        self.pub.on_disconnect = lambda c, u, f=None, rc=None, p=None: log(f"로컬 브로커 단절 rc={rc}")
        self.pub.connect_async(BROKER_HOST, BROKER_PORT, keepalive=30)
        self.pub.loop_start()

        self.go1 = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=GO1_CLIENT_ID)
        self.go1.reconnect_delay_set(1, 10)
        self.go1.on_connect = self._go1_connect
        self.go1.on_message = self._go1_message
        self.go1.on_disconnect = lambda c, u, f=None, rc=None, p=None: log(f"Go1 MQTT 단절 rc={rc}")
        self.go1.connect_async(GO1_HOST, 1883, keepalive=20)
        self.go1.loop_start()

    # ---------------- 발행 ----------------
    def event(self, name, source, path_id=None, point_count=None, note="", ts_ms=None):
        with self.lock:
            self.seq += 1
            msg = {"schema": SCHEMA, "node_id": NODE, "entity_id": ENTITY,
                   "seq": self.seq, "ts_ms": ts_ms if ts_ms is not None else now_ms(),
                   "event": name, "path_id": path_id, "point_count": point_count,
                   "source": source, "note": note}
        payload = json.dumps(msg, ensure_ascii=False)
        rc = self.pub.publish(T_EVENT, payload, qos=1, retain=False).rc
        log(f"event {payload} rc={rc}")

    def state(self):
        t = time.monotonic()
        with self.lock:
            bridge = self.bridge_yaw_deg if t - self.bridge_at <= BRIDGE_STALE_S else None
            moving = self.moving if t - self.moving_at <= MOTION_STALE_S else None
            st = self.go1_state if t - self.go1_state_at <= config.GO1_STALE_S else None
            bms = self.go1_bms if t - self.go1_bms_at <= config.GO1_BMS_STALE_S else None
            path_active, path_id = self.path_active, self.path_id

        odo = None
        rpy = None
        if st is not None and any(st):     # 전부 0 = 데이터 부재 (go1_link 규칙)
            rpy = struct.unpack_from("<3h", st, go1_link.BODY_RPY)
            odo = round(wrap180(float(rpy[2])), 2)
        bat = Go1Link._bms_detail(bms)
        soc = bat["soc_pct"] if bat else None
        if bridge is not None:
            yaw, src = bridge, "bridge_state"
        elif odo is not None:
            yaw, src = odo, "go1_odometry"
        else:
            yaw, src = None, None
        # 기존 키는 그대로 두고 상세를 덧붙인다. 같은 바이트에서 이미 읽고 있는
        # 값들이라 비용은 없고, 가시화가 배터리 경보·기울기를 스스로 판정할 수 있다.
        out = {"schema": SCHEMA, "node_id": NODE, "entity_id": ENTITY, "ts_ms": now_ms(),
               "battery_pct": int(soc) if soc is not None else None,
               "yaw_deg": yaw, "yaw_source": src,
               "yaw_odometry_deg": odo,
               "moving": moving, "path_active": path_active, "path_id": path_id,
               "battery": bat,
               "ts_iso": iso_now()}
        if rpy is not None:
            out["pose"] = {"roll_deg": float(rpy[0]), "pitch_deg": float(rpy[1]),
                           "yaw_deg": float(rpy[2]),
                           "body_height_m": round(struct.unpack_from(
                               "<f", st, go1_link.F_HEIGHT)[0], 4)}
        return out

    def state_loop(self):
        nxt = time.monotonic()
        while True:
            try:
                if self.pub.is_connected():
                    self.pub.publish(T_STATE, json.dumps(self.state()), qos=0, retain=False)
            except Exception as e:
                log("nav_state 오류:", repr(e))
            nxt += STATE_PERIOD_S
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 2.0:
                nxt = time.monotonic()

    # ---------------- 경로 사건 (원천 공통) ----------------
    def on_path_received(self, pid, count, source, ts_ms=None):
        with self.lock:
            self.path_active, self.path_id, self.last_path_id = True, pid, pid
        self.event("path_received", source, pid, count, ts_ms=ts_ms)

    def on_path_cancel(self, source, ts_ms=None):
        with self.lock:
            was, pid = self.path_active, self.last_path_id
            self.path_active, self.path_id = False, None
        self.event("path_cancel", source, pid, note=f"path_active_before={str(was).lower()}", ts_ms=ts_ms)

    def on_path_done(self, source, ts_ms=None):
        with self.lock:
            pid = self.last_path_id
            self.path_active, self.path_id = False, None
        self.event("path_done", source, pid, ts_ms=ts_ms)

    # ---------------- Go1 내부 MQTT (구독만) ----------------
    def _go1_connect(self, client, userdata, flags, rc, properties=None):
        log(f"Go1 MQTT 접속 {GO1_HOST} rc={rc}")
        if rc == 0:
            client.subscribe([("robot/state", 0), ("bms/state", 0)])

    def _go1_message(self, client, userdata, msg):
        t = time.monotonic()
        with self.lock:
            if msg.topic == "robot/state" and len(msg.payload) >= go1_link.STATE_LEN:
                self.go1_state, self.go1_state_at = bytes(msg.payload), t
            elif msg.topic == "bms/state" and len(msg.payload) >= go1_link.BMS_LEN:
                self.go1_bms, self.go1_bms_at = bytes(msg.payload), t

    # ---------------- journal ----------------
    def journal_loop(self):
        cursor = None
        while True:
            args = ["journalctl", *JOURNAL_MATCH, "-f", "-o", "json", "--no-pager"]
            # 처음엔 지난 줄을 사건으로 내지 않는다(-n 0). journalctl 이 죽어 다시 띄울 때는
            # 마지막으로 처리한 줄 뒤부터 이어 받는다.
            args += ["--after-cursor", cursor] if cursor else ["-n", "0"]
            log("journal 구독:", " ".join(args))
            try:
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    try:
                        e = json.loads(line)
                        cursor = e.get("__CURSOR", cursor)
                        m = e.get("MESSAGE")
                        if not isinstance(m, str) or "[PATH]" not in m:
                            continue
                        ts = int(e["__REALTIME_TIMESTAMP"]) // 1000
                        self.handle_log_line(m, ts)
                    except Exception as ex:
                        log("journal 줄 파싱 오류:", repr(ex), line[:200])
                proc.wait()
                log(f"journalctl 종료 rc={proc.returncode} — 2초 뒤 다시")
            except Exception as ex:
                log("journal 오류:", repr(ex))
            time.sleep(2)

    def handle_log_line(self, m, ts_ms):
        if any(k in m for k in (LOG_ACTIVATED, LOG_CANCEL, LOG_DONE)):
            with self.lock:
                self.journal_events += 1
        if LOG_ACTIVATED in m:
            rest = m.split(LOG_ACTIVATED, 1)[1].split()
            pid = int(rest[0])
            count = next((int(x.split("=", 1)[1]) for x in rest if x.startswith("waypoints=")), None)
            self.on_path_received(pid, count, "journal", ts_ms)
        elif LOG_CANCEL in m:
            self.on_path_cancel("journal", ts_ms)
        elif LOG_DONE in m:
            self.on_path_done("journal", ts_ms)

    # ---------------- 수동 캡처 ----------------
    def open_capture(self, ports):
        ETH_P_ALL, ETH_P_IP, SO_ATTACH_FILTER = 0x0003, 0x0800, 26
        # ETH_P_ALL 이어야 한다 — 커널은 **나가는** 패킷(브리지 → 유니티 15101·5009)을
        # ETH_P_ALL 소켓에만 복사한다. ETH_P_IP 로 열면 들어오는 것만 보인다.
        s = socket.socket(socket.AF_PACKET, socket.SOCK_DGRAM, socket.htons(ETH_P_ALL))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF)
        # classic BPF: IPv4 · UDP · 첫 조각 · 목적지 포트 ∈ ports
        n = len(ports)
        prog = [
            (0x20, 0, 0, 0xFFFFF000),         # ld skb->protocol (SKF_AD_PROTOCOL)
            (0x15, 0, 9 + n, ETH_P_IP),       # jeq IPv4 else drop
            (0x30, 0, 0, 0),                  # ldb [0]
            (0x54, 0, 0, 0xF0),               # and #0xf0
            (0x15, 0, 6 + n, 0x40),           # jeq #0x40 else drop
            (0x30, 0, 0, 9),                  # ldb [9]
            (0x15, 0, 4 + n, 17),             # jeq #17 (UDP) else drop
            (0x20 | 0x08, 0, 0, 6),           # ldh [6]
            (0x45, 2 + n, 0, 0x1FFF),         # jset frag offset → drop
            (0xB1, 0, 0, 0),                  # ldxb 4*([0]&0xf)
            (0x48, 0, 0, 2),                  # ldh [x+2]  (dst port)
        ]
        for i, p in enumerate(ports):
            prog.append((0x15, n - i, 0, p))  # jeq port → accept
        prog += [(0x06, 0, 0, 0), (0x06, 0, 0, 0xFFFF)]   # drop, accept
        # 점프는 "다음 명령부터 몇 칸"이다. drop = 11+n, accept = 12+n.
        raw = b"".join(struct.pack("HBBI", *ins) for ins in prog)
        buf = ctypes.create_string_buffer(raw)
        fprog = struct.pack("HL", len(prog), ctypes.addressof(buf))
        s.setsockopt(socket.SOL_SOCKET, SO_ATTACH_FILTER, fprog)
        self._bpf_buf = buf
        eff = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        log(f"캡처 열림 ports={ports} SO_RCVBUF={eff}")
        return s

    def capture_loop(self, use_capture_for_path):
        ports = [STATE_PORT, TELEOP_PORT, MOTION_PORT]
        if use_capture_for_path:
            ports.insert(0, PATH_PORT)
        while True:
            try:
                s = self.open_capture(ports)
                while True:
                    pkt, addr = s.recvfrom(65535)
                    # lo 에서는 한 데이터그램이 나감(4)·들어옴(0) 두 번 보인다
                    if addr[3] == 772 and addr[2] == socket.PACKET_OUTGOING:
                        continue
                    try:
                        self.handle_ip(pkt, use_capture_for_path)
                    except Exception as ex:
                        log("패킷 처리 오류:", repr(ex), pkt[:80])
            except Exception as ex:
                log("캡처 오류:", repr(ex), "— 3초 뒤 다시")
                time.sleep(3)

    def handle_ip(self, ip, use_capture_for_path):
        ihl = (ip[0] & 0x0F) * 4
        if struct.unpack_from("!H", ip, 6)[0] & 0x2000:
            log("조각난 UDP 데이터그램은 해석하지 않는다", struct.unpack_from("!H", ip, ihl + 2)[0])
            return
        dport = struct.unpack_from("!H", ip, ihl + 2)[0]
        data = ip[ihl + 8:]
        if dport == STATE_PORT:
            self.handle_state(data, use_capture_for_path)
        elif dport == TELEOP_PORT:
            self.handle_teleop(data)
        elif dport == MOTION_PORT:
            self.handle_motion(data)
        elif dport == PATH_PORT and use_capture_for_path:
            self.handle_path(data.decode("utf-8", "replace"))

    def handle_state(self, data, use_capture_for_path):
        # seq tms x z yaw vx vy wz estop mode
        f = data.split()
        if len(f) < 10:
            return
        yaw = round(wrap180(float(f[4]) * 180.0 / 3.141592653589793), 2)
        mode = int(f[9])
        with self.lock:
            self.bridge_yaw_deg, self.bridge_at = yaw, time.monotonic()
            prev, self.last_mode = self.last_mode, mode
        if mode == prev:
            return          # 같은 값이 연달아 오면 처음 한 번만
        if mode == 98:
            with self.lock:
                pid = self.last_path_id
            self.event("cancel_ack", "udp15101_mode98", pid)
        elif mode == 99 and use_capture_for_path:
            self.on_path_done("udp15101_mode99")

    def handle_teleop(self, data):
        # vx vy wz estop — "MISSION PING" 같은 텍스트는 건너뛴다
        f = data.split()
        if len(f) != 4:
            return
        try:
            es = int(f[3])
            float(f[0])
        except ValueError:
            return
        with self.lock:
            prev, self.teleop_estop = self.teleop_estop, es
        if prev == 0 and es == 1:
            with self.lock:
                pid = self.last_path_id
                self.path_active, self.path_id = False, None
            self.event("estop", "udp15100_estop", pid)

    def handle_motion(self, data):
        if b"state_change" not in data or b"motion_active" not in data:
            return
        j = json.loads(data)
        if "state_change" in j and "motion_active" in j:
            t = time.monotonic()
            active = bool(j["motion_active"])
            with self.lock:
                gap = t - self.moving_at
                self.moving, self.moving_at = active, t
                if not active or gap > MOTION_STALE_S:
                    self.motion_since = t if active else None
                elif self.motion_since is None:
                    self.motion_since = t
                warn = (self.journal_mode and not self.silent_warned
                        and self.journal_events == 0 and self.motion_since is not None
                        and t - self.motion_since > SILENT_WARN_S)
                if warn:
                    self.silent_warned = True
            if warn:
                # 사건을 지어내지 않는다. nav_event 로도 보내지 않는다 — 로그에만 한 번.
                log(f"경고: 5009 motion_active=true 가 {SILENT_WARN_S:.0f}초 넘게 이어지는데 "
                    f"이 프로세스가 뜬 뒤 journal 경로 사건이 한 번도 없다. 텔레옵 주행일 수도 있지만, "
                    f"go1_sdk_pc 로그 문구({LOG_ACTIVATED!r} · {LOG_CANCEL!r} · {LOG_DONE!r})가 "
                    f"바뀌었는지 확인할 것 (match: {' '.join(JOURNAL_MATCH)})")

    def handle_path(self, s):
        if s.startswith("PATH_CANCEL"):
            self.chunk_pid, self.chunk_total, self.chunk_parts = -1, 0, {}
            self.on_path_cancel("udp15110")
            return
        if s.startswith("CHUNK "):
            meta, _, part = s[6:].partition(" ")
            pid, total, idx = (int(x) for x in meta.split("/"))
            if total <= 0 or not 0 <= idx < total:
                return
            if pid != self.chunk_pid or total != self.chunk_total:
                self.chunk_pid, self.chunk_total, self.chunk_parts = pid, total, {}
            self.chunk_parts[idx] = part
            if len(self.chunk_parts) < total:
                return
            s = "".join(self.chunk_parts[i] for i in range(total))
            self.chunk_pid, self.chunk_total, self.chunk_parts = -1, 0, {}
        j = json.loads(s)
        if j.get("type") != "go1_path":
            log("go1_path 아님 — 무시", s[:120])
            return
        pts = j.get("points")
        self.on_path_received(j.get("path_id"), len(pts) if isinstance(pts, list) else None, "udp15110")


def journal_readable():
    try:
        r = subprocess.run(["journalctl", *JOURNAL_MATCH, "-n", "1", "-q", "--no-pager"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "insufficient permissions" not in r.stderr \
            and "No journal files" not in r.stderr
    except Exception as ex:
        log("journal 확인 실패:", repr(ex))
        return False


def main():
    feed = Feed()
    if PATH_EVENT_SOURCE == "journal":
        use_journal = True
    elif PATH_EVENT_SOURCE == "capture":
        use_journal = False
    else:
        use_journal = journal_readable()
    feed.journal_mode = use_journal
    log(f"시작 topic={BASE} 경로 사건 원천={'journal' if use_journal else 'capture'} "
        f"ports path={PATH_PORT} state={STATE_PORT} teleop={TELEOP_PORT} motion={MOTION_PORT}")

    # feed_started 는 브로커에 붙은 뒤 낸다(최대 10초 기다리고, 못 붙으면 paho 큐에 둔다)
    for _ in range(100):
        if feed.pub.is_connected():
            break
        time.sleep(0.1)
    feed.event("feed_started", "startup",
               note=f"path_events={'journal' if use_journal else 'capture'}")

    threading.Thread(target=feed.state_loop, daemon=True, name="state").start()
    threading.Thread(target=feed.capture_loop, args=(not use_journal,), daemon=True,
                     name="capture").start()
    if use_journal:
        threading.Thread(target=feed.journal_loop, daemon=True, name="journal").start()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
