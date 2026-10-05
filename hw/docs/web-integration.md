# 관제 웹 연동 가이드 (가시화 ↔ pi7)

작성 2026-09-10 · 대상 가시화 팀 · 확인 환경 pi7 (192.168.50.172)

관제 웹이 로봇·센서의 상태를 읽고 명령을 내리기 위해 알아야 할 것을 모았다.
상태는 JSON, 명령은 protobuf 다 — 두 경로의 인코딩이 다르다는 점이 가장 자주 걸린다.

---

## 1. 브로커 접속

브로커는 로봇 옆 파이(pi7)에서 mosquitto 로 돈다. 브라우저는 생 TCP 를 못 열어서
1883 이 아니라 WebSocket 리스너로 붙는다. 같은 브로커·같은 토픽이고 전송 계층만 다르다.

```
ws://pi7.local:9001/mqtt
```

경로는 `/` 와 `/mqtt` 둘 다 받는다. MQTT 3.1.1 과 5 둘 다 되고, 서브프로토콜은 `mqtt` 다.
인증은 아직 없다(`allow_anonymous true`) — 현장망이 폐쇄망이라는 전제이므로 공개망에 두지 말 것.

### 붙지 않을 때 확인 순서

1. **페이지가 https 인가.** https 페이지에서는 `ws://` 가 mixed content 로 차단된다.
   웹을 http 로 띄우거나 브로커에 TLS(wss)를 붙여야 한다.
2. **`pi7.local` 이 풀리는가.** mDNS 는 OS 마다 다르다. 윈도우는 Bonjour 가 없으면 실패하고
   콘솔에 `ERR_NAME_NOT_RESOLVED` 가 뜬다. 임시로 IP(`ws://192.168.50.172:9001`)를 써서
   이름 문제인지 가른다. 다만 현장에서 핫스팟으로 망이 바뀌면 IP 도 바뀌므로 상시 하드코딩은 금물.
3. **같은 망인가.** pi7 은 지금 `SysAILAB_5GHz` 에 있다.

---

## 2. 상태 구독 (JSON)

토픽은 `zoneA/{종류}/{장치id}/{채널}` 꼴이다.
로봇은 `zoneA/robot/go1-001`, 센서는 `zoneA/sensor/wl-001`.

| 채널 | retain | 주기 | 담긴 것 |
|---|---|---|---|
| `status` | **예** | 10초 | 장치 목록, 배터리, `link`, `device_status`, `media` |
| `state` | 아니오 | 5초 (임무 중 20Hz) | 배터리, 위치, 속도, `robot_mode` |
| `heartbeat` | 아니오 | 5초 | 살아있음만 |

`status` 가 retain 이라는 점이 중요하다. **붙는 즉시 등록된 장치 전부의 마지막 상태가 한 번에 온다.**
첫 화면은 `status` 로 그리고 이후 갱신을 `state` 로 받으면 빈 화면 없이 시작한다.
`state` 만 구독하면 최대 5초간 아무것도 못 그린다.

```js
// npm i mqtt
import mqtt from "mqtt"

const client = mqtt.connect("ws://pi7.local:9001/mqtt", {
  clientId: "web-" + Math.random().toString(16).slice(2, 8),
  protocolVersion: 5,
  reconnectPeriod: 2000,
})

const devices = new Map()

client.on("connect", () => {
  client.subscribe(["zoneA/+/+/status", "zoneA/+/+/state", "zoneA/+/+/heartbeat"])
})

client.on("message", (topic, buf) => {
  const [zone, etype, eid, channel] = topic.split("/")
  let msg
  try { msg = JSON.parse(buf.toString()) } catch { return }

  const d = devices.get(eid) ?? { eid, etype }
  d.lastSeen = Date.now()

  if (channel === "status") {
    d.online = msg.status === "online"   // 끊기면 LWT 가 offline 을 대신 넣는다
    d.health = msg.device_status         // ok | degraded | fault
    d.link   = msg.link                  // 로봇 <-> 파이 내부 링크
    d.mode   = msg.robot_mode
    if (msg.battery_pct !== undefined) d.battery = msg.battery_pct
  } else if (channel === "state") {
    d.battery  = msg.battery_pct
    d.position = msg.position
    d.health   = msg.device_status
    d.mode     = msg.robot_mode
  }
  devices.set(eid, d)
})
```

### 2-1. 확장 필드 (2026-09-30 추가)

로봇이 이미 보내고 있는데 파이가 버리던 값들을 전부 실었다. **기존 최상위 키
(`battery_pct`·`position`·`speed_mps`·`robot_mode`)는 이름도 의미도 그대로다** —
위 코드는 고치지 않아도 계속 돈다. 새 정보는 하위 객체로만 붙었다.

`state` 한 건의 크기가 약 330B → 약 1.4KB 로 늘었다. 임무 중 20Hz 면 약 27KB/s 다
(카메라 한 대의 영상이 0.58Mbps=72KB/s 이므로 그보다 작다).

| 키 | 담긴 것 | 비고 |
|---|---|---|
| `pose` | `roll_deg`, `pitch_deg`, `yaw_deg`, `body_height_m` | 예전엔 yaw 만 있었다. 기울기·엎드림을 이제 볼 수 있다 |
| `velocity` | `vx`, `vy`, `speed_mps`, `yaw_rate` | `speed_mps` 는 크기뿐이라 **방향이 없었다**. 전후진 구별이 가능해졌다 |
| `battery` | `pct`, `current_ma`, `cycles`, `voltage_v`, `temp_c[4]`, `cells_mv[6]`, `status`, `age_s` | `current_ma` 음수 = 방전. `temp_c` 는 BQ 2점 + MCU 2점 |
| `joints_deg` | 12관절 각도(도), 4다리 × (hip, thigh, calf) | 기립 자세는 `(±1, 46, -92) × 4` 근처 |
| `freshness` | `state_age_s`, `bms_age_s`, `sampled_at`, `sample_hz`, `state_hz` | **§6 의 "멈춘 값" 문제를 웹이 혼자 판정할 수 있다** |
| `origin` | `device_type`, `fw_version`, `controller_link`, `hostname`, `ip`, `mac`, `pid`, `boot_id` | 메시지 한 건만 보고도 출처가 완결된다 |
| `link`, `sdk` | 예전엔 `status` 에만 있었다 | 이제 `state` 에도 같이 온다 |
| `raw_unknown` | `{hex, confidence:"unknown", layout}` | **해독하지 못한 바이트다. 이름이 없는 이유가 그것이다 — 쓰지 말 것** |

`status`(retain·10초)에는 추가로 이런 것이 온다.

| 키 | 담긴 것 |
|---|---|
| `battery` | 위와 같은 상세. retain 이라 **붙는 즉시 한 번에** 받는다 |
| `robot_fw` | 로봇이 스스로 보고한 펌웨어 — `sport_mode`, `app`, 노드별(`raspi`/`nano1~3`) 구성 |
| `origin` | 위와 같음 |

`battery.cells_mv` 는 **실제로 물린 셀만** 들어간다(Go1 은 6S). 로봇은 10칸을 보내는데
빈 네 칸에 32mV 같은 값이 들어와서, 그대로 내보내면 "셀 하나가 죽었다"로 보인다.
실측 대조: SOC 18% → 셀 3424mV·합 20.5V / SOC 82% → 셀 3872mV·합 23.2V.

---

## 3. 화면에 그릴 때 지켜야 할 것

### 3-1. 연결상태는 세 겹이다

한 덩어리로 뭉치면 "로봇이 꺼진 것"과 "웹이 끊긴 것"을 구별할 수 없다.

| 구간 | 어디서 아나 |
|---|---|
| 브라우저 ↔ 브로커 | mqtt 클라이언트의 `connect` / `close` 이벤트 |
| 노드 ↔ 브로커 | `status.status` (`online` / `offline`). 노드가 죽으면 브로커가 LWT 로 대신 발행 |
| 로봇 ↔ 파이 | `status.link` (`ok` / `degraded` / `fault`) |

`degraded` 는 자세는 오는데 배터리가 안 오는 상태다.

### 3-2. 자체 타임아웃도 둔다

LWT 는 약 15초 뒤에 오는데, 브라우저가 그 사이에 붙으면 놓친다.
마지막 수신 시각으로 직접 판정하는 편이 확실하다. 노드 쪽 판정 기준과 같은 20초를 쓰면
화면과 서버가 같은 소리를 한다.

```js
setInterval(() => {
  const now = Date.now()
  for (const d of devices.values()) {
    if (now - d.lastSeen > 20000) d.online = false   // 하트비트 5초 x 4회
  }
}, 1000)
```

### 3-3. 배터리의 `null` 과 정지값을 구별한다

`null` 은 "모른다"는 뜻이다. **0% 로 그리면 안 된다** — 방전 직전과 구별되지 않는다.

그리고 지금 알려진 문제 하나. 로봇 링크가 끊겨도 마지막 배터리·위치 값이 계속 흘러나온다.
`link` 가 `fault` 면 값에 시각을 붙이거나 흐리게 처리해야 사용자가 속지 않는다.

```js
function battery(d) {
  if (d.battery == null) return "—"
  return d.link === "fault" ? `${d.battery}% (마지막 수신)` : `${d.battery}%`
}
```

### 3-4. 카메라 메타데이터는 영상 채널로 같이 온다

`status.media` 는 카메라가 아니라 **노드 자신의 영상 송출기** 상태다.
Go1 의 실제 카메라 5대는 Nano 3대(192.168.123.13/14/15)에 물려 있고,
별도 뷰어(`http://pi7.local:8090`)가 WebSocket 으로 브라우저에 직접 물린다. MQTT 로는 안 나간다.

**2026-09-30 부터 그 영상 채널이 메타데이터를 같이 보낸다.** 별도 엔드포인트는 없다 —
지금 붙는 방식 그대로 받으면 된다.

`ws://pi7.local:8090/ws/<1~5>` (또는 `wss://pi7.local:8443/...`)

| 텍스트 프레임 | 언제 | 담긴 것 |
|---|---|---|
| `{"t":"hello", ...}` | 붙고 첫 키프레임 뒤 1회 | 아래 표 전부 |
| `{"t":"stat", ...}` | 2초마다 | 같은 필드 + 갱신된 계수 |

| 키 | 담긴 것 | 실측(정면 카메라) |
|---|---|---|
| `cam`, `position`, `label` | 카메라 번호 / 기계용 위치 / 사람용 이름 | `1`, `front`, `정면` |
| `node_id`, `zone_id` | **어느 파이가 보냈는가** | `pi7`, `zoneA` |
| `source` | 로봇 쪽 원천 주소 | `ws://192.168.123.13:9101` |
| `codec`, `profile`, `level` | SPS 에서 매번 읽는다 | `avc1.424028`, `66/64`, `40` |
| `width`, `height` | SPS 에서 매번 읽는다 | `464`, `400` |
| `seq`/`frames`, `bytes`, `keyframes` | 누적. 재연결하면 0 부터 | `650`, `1465917`, `44` |
| `fps`, `gap_ms`, `key_interval_s` | 실측 프레임률 / 도착 간격 / 키프레임 간격 | `30.0`, `33.3`, `0.5` |
| `first_frame_at`, `last_frame_at`, `last_frame_at_iso`, `age_s` | 프레임 시각과 나이 | — |
| `connected`, `error` | 상류 연결 상태 | `true`, `null` |
| `server_time`, `server_time_iso`, `uptime_s` | 파이 시계. 시계 차이 보정의 기준 | — |

**바이너리 프레임의 9바이트 헤더는 바뀌지 않았다** — `[1B 키프레임][8B 파이 수신시각 ms]`
그대로다. 이미 붙어 있는 클라이언트는 고칠 것이 없다. 프레임마다 새로 생기는 정보는
사실상 순번뿐이라 `stat` 이 실어 준다(2초에 한 번이면 유실 여부를 알기에 충분하다).

MJPEG 대비 경로(`/stream/<1~5>`)는 multipart 파트마다 헤더로 같은 것을 붙인다.

```
X-Frame-Seq, X-Timestamp, X-Timestamp-Iso, X-Node-Id, X-Zone-Id,
X-Camera, X-Camera-Position, X-Camera-Source, X-Image-Bytes,
X-Image-Width, X-Image-Height, X-Sha1, X-Upstream-Seq, X-Upstream-Fps, X-Path
```

`X-Sha1` 은 **카메라 정지를 알아내는 유일한 방법**이다. Go1 은 그림이 멈춰도 30fps 를
계속 내보내므로 프레임률로는 알 수 없다(`robot/go1_camera.py` 참조). 직전 파트와
`X-Sha1` 이 같으면 같은 그림이다.

여전히 남은 것: 카메라별 도달 여부가 **MQTT 로는** 안 나간다. 위 경로로 카메라마다
한 번 붙어 보면 알 수 있지만, 상태 요약 한 건으로 5대를 보려면 노드에 필드 추가가 필요하다.

---

## 4. 제어 명령 (protobuf)

상태와 달리 **명령 경로는 protobuf 다.** JSON 이 아니다.

```
terminal/go1-001/downlink   상위 → 로봇 (command / cancel_request)
terminal/go1-001/uplink     로봇 → 상위 (acceptance / status / result / capability)
```

QoS 는 양방향 1. 봉투(`PhysicalCommandEnvelope`)의 `oneof body` 가 메시지 종류를 정하고,
토픽에는 방향만 담긴다.

클라이언트는 [`mqtt-command.js`](./mqtt-command.js) 에 있다. 그대로 복사해 쓰면 된다.

```bash
npm i mqtt protobufjs
```

```js
import { createRobotControl, CommandRejected, CommandFailed } from "./mqtt-command.js"

const robot = createRobotControl({
  device: "go1-001",                      // 시뮬레이터는 "go1-sim"
  onProgress: ({ action, detail }) => console.log("진행", action, detail),
  onCapability: (actions) => console.log("쓸 수 있는 명령", actions),
  onLink: (up) => setBadge(up ? "연결" : "끊김"),
})
```

### 4-1. 기본 제어 명령

각 함수는 **종료 보고까지 기다리는 Promise** 를 돌려준다.
거부·중단·시한초과는 전부 reject 로 온다.

```js
await robot.ping()                        // 로봇 안 움직임
await robot.diag()                        // 노드 진단 한 벌

await robot.turn(45)                      // 오른쪽 45도. 그 방향에 그대로 선다
await robot.turn(-90)                     // 왼쪽 90도
await robot.moveForward(0.5)              // 0.5m 직진
await robot.moveForward(2.0, 0.2)         // 속도 지정
await robot.scan({ steps: 8, step_deg: 45, forward_m: 1.0 })
await robot.scan({ steps: 8, step_deg: 45, forward_m: 0, hold_after_capture: 1 })
await robot.scanContinue(45)              // 45도 결과를 다 띄웠다 — 다음 회전

await robot.abortMission()                // 진행 중인 임무만 접는다
await robot.abort()                       // 무엇이 돌고 있든 멈춘다
```

성공하면 결과 맵이 온다. 회전은 이런 모양이다.

```js
const { result } = await robot.turn(45)
// result: { deg: 45, duration_s: 6.4, yaw_deg: 315.0 }
```

실패는 종류를 갈라 받는다. 이 구분이 중요하다 —
**거부는 로봇이 움직이지 않았다는 뜻이라 그대로 재시도해도 안전하고, 중단은 아니다.**

```js
try {
  await robot.turn(45)
} catch (e) {
  if (e instanceof CommandRejected) {
    // 아무것도 실행되지 않았다. e.code 예: go1_sdk_not_running, robot_state_dead,
    //   mission_in_progress, deg_out_of_range, battery_too_low
  } else if (e instanceof CommandFailed) {
    // 실행에 들어갔다가 중단·취소됐다. 로봇이 움직였을 수 있다.
  } else {
    // CommandTimeout — 아직 돌고 있을 수 있다. abort() 를 고려할 것.
  }
}
```

### 4-2. 명령 어휘

| action | 웹에서 호출 | 로봇이 움직이나 | 파라미터 |
|---|---|---|---|
| `ping` | 가능 | 아니오 | 없음 |
| `diag` | 가능 | 아니오 | 없음 |
| `turn` | 가능 | **예** | `deg` (오른쪽 +, 5~360) |
| `move_forward` | 가능 | **예** | `distance_m` (0.05~10), `vx` (0.05~0.30) |
| `move_relative` | 가능 | **예** | `dx_m` 앞+/뒤− · `dy_m` 왼쪽+/오른쪽− (각각 0 또는 0.05~10), `v_mps` (0.05~0.30, 기본 0.15) |
| `teleop` | 가능 | **예** | `vx` (±0.40), `vy` (±0.30), `vyaw` 반시계+ (±1.0 rad/s), `hold_ms` (기본 500, 250~1000) — **스트림 어휘**, 아래 참조 |
| `scan_mission` | 가능 | **예** | `steps` (1~36), `step_deg` (5~180), `forward_m` (0~10), `vx`, `hold_after_capture` (1/0), `hold_timeout_s` (3~60, 기본 20) |
| `scan_continue` | 가능 | 아니오 | `rotation_deg` (방금 화면에 띄운 각도 0·45·…·315) |
| `abort_mission` | 가능 | 예 | 없음 |
| `abort` | 가능 | 예 (정지) | 없음 |
| `sdk_start` | 가능 | **예 (기립)** | 없음 |
| `sdk_stop` | 가능 | 아니오 | 없음 |
| `sdk_auto` | 가능 | 아니오 | `on` (1/0) |
| `assign_mission` | **불가** | — | 문자열이 필요하다 |
| `stream` | **불가** | — | 문자열이 필요하다 |

`scan_mission` 의 `hold_after_capture=1` 은 **촬영마다 로봇을 세워 두고 웹 신호를 기다린다**
(5-3). 0 이거나 없으면 예전과 똑같이 돈다. `hold_timeout_s` 는 3~60 으로 잘린다.

`move_relative`·`teleop` 의 축은 **ROS REP-103**(x 앞 +, y 왼쪽 +, yaw 반시계 +)이다.
`turn` 의 `deg` 는 오른쪽이 + 라서 회전 부호가 반대다.

**`move_relative`** 는 `move_forward` 와 생명주기·거부 규칙이 같다(Acceptance → Status → Result,
다른 이동 중이면 `mission_in_progress` 거부, abort 면 Result ABORTED). dx·dy 를 둘 다 주면
대각선으로 한 번에 간다. 방향은 출발 방위를 IMU 로 붙들고 회전하지 않는다. Result:
`dx_m, dy_m, v_mps, duration_s, reached`(1/0) + 잰 값 `moved_dx_m, moved_dy_m`(출발 자세 몸 기준),
`odo_m, cmd_m, src_odo`(1=odometry, 0=명령 적분). 모르는 값은 키가 없다.

**`teleop`** 은 스트림 어휘다. 건마다 Result 가 없고, 수락 Acceptance 는 세션 첫 건과 값이
바뀐 건만 나온다. **거절은 항상 Acceptance 로 나온다.** 마지막 teleop 뒤 `hold_ms` 안에 다음
건이 없으면 노드가 스스로 세운다(브리지 0.15s 포함 최대 약 hold_ms+0.15s). 0·0·0 은 즉시
정지, 브로커 단절도 즉시 정지. 범위 밖 속도는 잘라서 쓰고 NaN/inf 는 `INVALID_ARGUMENT`.

| 상황 | teleop 응답 |
|---|---|
| 임무(scan_mission·move_forward·turn·move_relative) 진행 중 — 0 포함 | 거부 `FAILED_PRECONDITION / mission_in_progress` (임무는 그대로) |
| teleop 중 임무가 옴 | 임무 수락, teleop 은 즉시 접힌다 |
| abort 직후 계속 오는 teleop | 거부 `teleop_halted_by_abort` — 0 을 받거나 1초 끊긴 뒤 다시 받는다 |
| 브리지 없음 + `sdk_auto` 꺼짐 | 거부 `go1_sdk_not_running` (상태 미수신이면 `robot_state_dead`) |
| 브리지 없음 + `sdk_auto` 켜짐 | 수락, 브리지를 띄운다(**로봇이 일어선다**), 뜨면 그때부터 따른다 |

`scan_continue` 는 이동 명령이 아니라서 **스캔이 도는 중에 보내는 것이 정상이다**
(`mission_in_progress` 로 거부되지 않고 브리지도 띄우지 않는다).

| 상황 | 응답 |
|---|---|
| 지금 대기 중인 촬영의 각도 | 수락 → 대기 해제. 결과 `{rotation_deg, step, waited_s}` |
| 아직 대기에 들어가지 않은 (안 찍은) 각도 | 수락 → 기억해 뒀다가 그 촬영 뒤 곧바로 해제. 결과 `{rotation_deg, step, latched: 1}` |
| 이미 지나간 각도 (같은 신호 두 번째 포함) | 거부 `stale_rotation`. 대기는 그대로 |
| 스캔이 없거나 대기를 켜지 않은 판 | 거부 `no_scan_in_progress` |

거부의 `rejection.code` 는 `FAILED_PRECONDITION`, 사유(`stale_rotation` 등)는 `rejection.message`
— `mqtt-command.js` 에서는 `e.detail` 이다.

마지막 둘(`assign_mission`, `stream`)은 `Capability` 목록에는 나오지만 **이 경로로는 호출할 수 없다.**
규약의 `parameters` 가 `map<string, double>` 이라 `"start"` 같은 문자열을 못 싣는다.
보내면 `INVALID_ARGUMENT` 로 거부된다. 규약이 확장되기 전까지는 다른 경로가 필요하다.

### 4-3. 구동 브리지는 평시에 내려가 있다

로봇이 브로커에 붙어 있다고 움직일 수 있는 것은 아니다. 실제 구동은 별도 브리지
(`go1-sdk`)가 맡는데, **이 브리지는 기동하는 순간 로봇을 일으켜 세운다.** 그래서
부팅 자동시작이 꺼져 있고 평시에는 "연결만 된 상태"로 둔다.

이동 명령은 필요하면 스스로 브리지를 띄운다. 그 사이 진행 보고가 두 건 더 온다.

```
수락 → sdk_starting → sdk_ready → executing → (임무 ACK…) → 종료
```

`sdk_starting` 이 보이면 **로봇이 지금 일어서는 중**이다. 화면에 그대로 드러내야 한다.
기립에 몇 초 걸리므로 이동 명령의 시한이 그만큼 길다.

```js
const robot = createRobotControl({
  onProgress: ({ detail }) => {
    if (detail?.event === "sdk_starting") setBanner("로봇이 일어서는 중…")
    if (detail?.event === "sdk_ready")    setBanner(null)
  },
})

await robot.sdkStart()        // 임무 전에 미리 세워 둔다 (선택)
await robot.turn(45)          // 브리지가 없으면 알아서 띄운다
await robot.sdkStop()         // 내린다. 로봇은 선 채로 남는다
await robot.sdkAuto(false)    // 자동 기동을 끈다 — 이후 이동 명령은 거부된다
```

브리지 상태는 상태 요약의 `sdk` 로도 읽는다.

```json
"sdk": { "ready": true, "autostart": true }
```

`ready` 가 `null` 이면 **"아직 모른다"이지 "꺼짐"이 아니다.** 노드가 방금 떠서 아직
확인하지 못한 상태다. 회색으로 두고 꺼짐으로 그리지 말 것.

### 4-4. 함정

- **파라미터는 숫자만 들어간다.** 위 제약과 같은 이유다. 그래서 임무 종류를
  파라미터가 아니라 action 이름으로 가른다(`turn`, `move_forward`, `scan_mission`).
- **`command_id` 는 멱등키다.** 같은 id 를 다시 보내면 재실행하지 않고 이전 응답만
  되돌려준다. 페이지 새로고침 때 카운터가 1부터 다시 시작하면 계속 충돌한다
  (실제로 겪은 문제다). 클라이언트가 시각을 섞어 만드는 이유가 그것이다.
- **응답은 수락 1건 → 진행 n건 → 종료 1건이다.** 거부되면 종료가 **아예 오지 않는다.**
- **명령 어휘(`Capability`)는 노드가 브로커에 붙는 순간 한 번만 발행된다.** retain 이
  아니라 그때 듣고 있지 않으면 못 받는다. 목록이 필요하면 구독한 상태에서
  `sudo systemctl restart robot-node` 를 하거나 `diag` 로 확인한다.
- **자동 기동이 꺼져 있으면** 브리지가 없을 때 이동 명령이 `go1_sdk_not_running`
  으로 거부된다. 켜져 있으면 대신 `sdk_starting` 진행 보고가 온다(4-3).
- **브리지를 띄우려면 노드에 권한이 있어야 한다.** polkit 규칙
  (`pi/deploy/50-go1-sdk.rules`)이 설치돼 있지 않으면 `systemctl_start_denied` 로
  거부된다. 조용히 실패하지 않고 사유가 그대로 올라온다.
- **`abort` 는 비상정지가 아니다.** 통신 경로의 정지다. E-stop 은 통신과 독립인
  장치 자체 안전장치다(규약 §7).


## 5. 임무 진행 ACK

`scan_mission` 의 진행 보고(`CommandStatus.detail`)는 JSON 문자열이다.

```json
{"ack":3,"of":9,"ack_seq":24,"event":"scan_turn","step":3,"steps":8,
 "yaw_deg":225.0,"note":"ok"}
```

| 키 | 뜻 |
|---|---|
| `ack` / `of` | **이번 임무의** ACK 순번과 총 건수. 진행률은 이걸 쓴다 |
| `ack_seq` | 로봇 원본 카운터. 브리지가 사는 동안 누적된다 — 진행률에 쓰지 말 것 |
| `event` | `scan_turn` / `forward` / `aborted` (대기가 켜진 판은 `scan_hold` / `scan_release` 도 — 5-3) |
| `step` / `steps` | 그 단계에서 몇 번째인가 |
| `yaw_deg` | 그 시점 방위(도). 모르면 `null` |

### 5-1. `of` 는 상수가 아니다

```python
expected = steps + (1 if forward_m > 0 else 0)
```

스캔 8회에 전진이 있으면 9, 전진 없이 스캔만 하면 8이다. 스캔 횟수는 명령 파라미터라
웹에서 바뀐다. **상수로 박지 말 것.** 첫 진행 보고에 `expected_acks` 가 실려 온다.

### 5-2. `yaw_deg` 는 대본과 회전 방향이 반대다

로봇은 오른쪽(시계)으로 돈다. 대본이 반시계를 가정하면 어긋난다.

| ACK step | index | 로봇 yaw_deg | 대본 각도 |
|---|---|---|---|
| 1 | 0 | 315 | 0 |
| 2 | 1 | 270 | 45 |
| 3 | 2 | 225 | 90 |
| 4 | 3 | 180 | 135 |
| 5 | 4 | 135 | 180 |
| 6 | 5 | 90 | 225 |
| 7 | 6 | 45 | 270 |
| 8 | 7 | 0 | 315 |

두 값을 더하면 항상 315 다. `대본각 = (315 - 로봇yaw) mod 360`.

**하지만 절대 yaw 로 맞추지 말 것.** 기준점이 움직인다.

- 시뮬레이터는 yaw 를 임무 사이에 초기화하지 않는다. 이전 임무가 끝난 방위에서
  다음 임무가 시작한다.
- 실물은 Unity 에서 경로를 내려보내면 yaw 오프셋이 출발 자세에 맞춰 재보정된다.

노드 선택도 탐지 정합도 **`step` 에서 유도한다.**

```
출발기준_방위 = -step_deg × step      // 로봇 기준, 시계 +
대본각        = step_deg × (step - 1)
```

### 5-3. 촬영 뒤 대기 (`hold_after_capture=1`)

한 판의 순서 (steps=8, step_deg=45):

```
0도 촬영 → /frame → [대기] → 회전1 → 정지 → 0.6초 → 45도 촬영 → /frame → [대기] → 회전2 → …
→ 315도 촬영 → /frame → [대기] → 회전8(출발 방향 복귀, 촬영 없음) → (forward_m>0 이면 직진) → 종료
```

대기는 **그 촬영의 `/frame` 이 나간 직후** 시작해 8번 모두 들어간다. 대기 중 로봇은 제자리에 서 있다.
대기에 들어가고 풀릴 때 진행 보고가 하나씩 온다. 이 둘은 `sdk_starting` 같은 **단계 보고**라
`ack` / `of` 가 없고 늘리지도 않는다(`of` 는 그대로 스캔 걸음 수 8).

```json
{"event":"scan_hold","step":3,"steps":8,"rotation_deg":135.0,"seq":3,
 "timeout_s":20.0,"yaw_deg":180.0,"note":"ok"}
{"event":"scan_release","step":3,"steps":8,"rotation_deg":135.0,
 "by":"web","waited_s":2.4,"note":"ok"}
```

| 키 | 뜻 |
|---|---|
| `step` | 촬영 순번 0~7 (0 = 출발 방향). `scan_turn` 의 `step`(1~8)과 다르다 |
| `rotation_deg` | 그 촬영의 각도. `/frame` 의 `rotation_deg` 와 같은 값 |
| `seq` | 그 `/frame` 의 `seq`. 사진이 안 나갔으면 `null` |
| `timeout_s` | 이만큼 신호가 없으면 로봇이 스스로 넘어간다 |
| `yaw_deg` | 지금 방위. 0도(출발) 촬영에서는 `null` |
| `note` (hold) | `ok` / `no_frame`(촬영 다리가 사진을 못 집음) / `frame_not_confirmed`(사진 알림이 안 옴 — 다리가 꺼졌을 수 있다) |
| `by` (release) | `web`(scan_continue) / `timeout` / `abort` |
| `waited_s` | 대기에 서 있던 시간(초) |

- **웹은 `scan_hold` 를 받은 판에서만 `scan_continue` 를 보낸다.** `scan_hold` 가 한 번도
  오지 않으면 대기가 없는 로봇이다.
- 신호가 대기보다 조금 먼저 와도 된다 — 로봇이 기억했다가 그 촬영 뒤 곧바로 푼다.
- 시간 초과(`by:"timeout"`)는 실패가 아니다. 다음 회전으로 가고 판은 계속된다.
- `abort` / `abort_mission` 은 대기 중에도 곧바로 먹는다. `scan_release`(`by:"abort"`) 뒤
  종료 결과는 예전과 같다(ABORTED / CANCELED). `/scan` 의 `scan_end` 도 그대로 온다.
- 한 바퀴 뒤 추가 회전(`door_turn`)은 없다. 로봇은 출발 방향에 서서 끝난다.

### 5-4. `note` 의 `odo` 와 `cmd`

전진 ACK 의 `note` 는 `ok odo=1.00m cmd=1.86m` 꼴이다.

- `odo` — 로봇 오도메트리로 잰 실제 이동거리. **정지 조건이 `odo >= forward_m` 이라
  목표값 언저리에서 늘 멈춘다.** 그래서 매번 1.00 으로 보이는 것이지 지령값을
  베껴 적은 것이 아니다. 오도메트리 자체의 누적 오차는 별개로 존재한다.
- `cmd` — 명령 속도를 시간으로 적분한 값(dead-reckoning). 같은 거리라도 걸린 시간이
  다르면 달라진다. 실이동이 명령보다 느려서 보통 `odo` 보다 크다.
  오도메트리가 죽었을 때의 대체 기준이며, 두 값의 차이가 곧 미끄러짐의 크기다.

## 6. 지금 알려진 문제

| 문제 | 영향 | 상태 |
|---|---|---|
| 로봇 링크가 끊겨도 마지막 배터리·위치가 계속 발행된다 | 멈춘 값을 살아있는 값으로 보여준다 | **`freshness.state_age_s` 로 웹이 직접 판정할 수 있게 됐다**(2-1). 값 자체를 막는 근본 수정은 아직 |
| 카메라 연결상태가 MQTT 로 안 나간다 | 상태 요약 한 건으로 5대를 못 본다 | 영상 채널의 `hello`/`stat` 으로는 카메라별로 다 나온다(3-4). MQTT 통합은 미정 |
| 동작 모드가 초당 두 번꼴로 떤다 | 상태 배지가 깜빡인다 | 속도 문턱으로 모드를 추정해서다. 웹에서 잠깐 눌러 줄 것 |

### 고쳐진 것 (2026-09-30)

| 문제 | 고친 내용 |
|---|---|
| 등록 정보의 `ip` 가 `127.0.0.1` 로 나갔다 | 브로커가 파이 자신이면 루프백이 잡히던 것을 고쳤다. 기본 경로로 다시 물어본다. **부작용: 보고되는 `mac` 이 실제 트래픽이 지나는 인터페이스(wlan0)의 것으로 바뀐다** |
| 로봇이 보내는 값의 3/4 를 버리고 있었다 | 자세·관절·속도벡터·배터리 상세를 전부 싣는다(2-1) |
| 영상에 "언제·어느 기기·어느 카메라"가 없었다 | `hello`/`stat`/MJPEG 헤더로 싣는다(3-4) |

### 고쳐진 것 (2026-09-10)

| 문제 | 고친 내용 |
|---|---|
| `forward_m: 0` 을 무시하고 1m 전진했다 | 0 을 "안 준 값"으로 보던 파싱을 고쳤다. 이제 0 이면 스캔만 돈다 |
| `ack` 가 명령을 넘어 누적됐다 | `ack` 는 임무별 순번이 되고, 로봇 원본은 `ack_seq` 로 따로 간다 |
| 연결 후 첫 명령이 응답을 놓쳤다 | `mqtt-command.js` 가 SUBACK 을 기다린 뒤 발행한다 |


## 7. 참고 — 시험 환경

로봇 없이도 전 구간을 돌려볼 수 있다. 시뮬레이터가 실물과 같은 필드·같은 방향으로 ACK 를 낸다.

```bash
# pi7 에서
cd /home/physical/hw/pi
python3 -m bench.fake_go1_sdk --interval 0.8
```

시뮬레이터로 시험할 때는 `DEVICE` 를 `go1-sim` 으로 둔다.
단, 5-3 의 회차 이월 문제는 시뮬레이터에만 있다.

CLI 로 같은 명령을 보내 비교할 수도 있다.

```bash
python3 -m bench.send_physical_command --ws --device go1-001 --action diag
```
