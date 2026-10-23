# 프로토콜

기기와 모델을 갈아끼워도 바뀌지 않는 부분. 이 문서에 적힌 것만 지키면 다른
기기·다른 모델을 붙일 수 있다.

두 계층이 있고 **전송 방식이 다르다.**

```
관제/가시화 ──MQTT──▶ 말단 노드 (로봇·센서·구동기·드론 엣지)
            ◀────────

드론 엣지  ──HTTP──▶ 드론 말단
           ◀────────
```

MQTT는 **제어와 상태**, HTTP는 **프레임과 판정**이다. 섞지 않는다 — 영상은
주기가 다르고 끊겨도 되는 반면, 명령은 한 번만 실행돼야 한다.

---

## 1. 제어 — MQTT + protobuf

### 토픽

```
terminal/<device_id>/downlink    관제 → 기기   명령
terminal/<device_id>/uplink      기기 → 관제   수락·진행·결과·상태
terminal/<device_id>/status      기기 → 관제   등록·생존 (retain, LWT)
```

`device_id`는 기기마다 **달라야 한다.** 같은 값으로 둘이 붙으면 MQTT client_id가
겹쳐 브로커가 번갈아 끊는다. k3s에서는 노드 이름(`spec.nodeName`)에서 끌어온다.

### 봉투

**protobuf가 정본이다**(`hw/pi/common/physical_command_pb2.py`). JSON 경로는
폐기됐다 — 노드는 `ParseFromString`으로만 읽으므로, JSON을 보내면 토픽에
도착은 하지만 파싱에 실패해 버려진다. 발행 쪽에는 성공으로 보이므로 특히
주의한다.

```protobuf
PhysicalCommandEnvelope {
  oneof body {
    Command           command;          // 관제 → 기기
    CancelCommandRequest cancel_request;
    CommandAcceptance acceptance;       // 기기 → 관제
    CommandStatus     status;
    CommandResult     result;
    Capability        capability;
  }
}

Command {
  string command_id;              // 멱등 키. 같은 값이면 다시 실행하지 않는다
  string target;                  // device_id
  string action;                  // 아래 어휘
  map<string,double> parameters;  // **숫자만** 담긴다
  int64  deadline_unix_ms;
  string traceparent;
}
```

`parameters`가 `double`뿐인 것은 제약이다. 문자열 인자가 필요한 명령은 이
규약으로 보낼 수 없다 — 담지 못하는 값을 조용히 버리지 말고 거절한다.

### 명령 4단계

기기는 명령 하나에 대해 순서대로 올린다.

```
acceptance   받았다 / 거절했다 (사유 포함)
status       executing · state_changed … (여러 번)
result       completed / failed (한 번, 마지막)
```

**관제는 result를 기다리지 않는다.** 발행하고 바로 돌아오고, 진행은 uplink를
구독해서 본다. 기다리면 느린 명령 하나가 화면 전체를 붙잡는다.

### 어휘

**공통** — 모든 노드가 가진다.

| action | 뜻 |
|---|---|
| `ping` | 살아 있나. uptime을 돌려준다 |
| `diag` | 지금 무엇을 하고 있나. 노드별 진단 모음 |

**이동체(robot)**

| action | 뜻 |
|---|---|
| `drive_start` / `drive_stop` | 구동을 쓸 수 있게 / 내린다. 기종에 따라 로봇이 일어선다 |
| `drive_auto` | 이동 명령이 왔을 때 구동을 자동으로 올릴지 (`on`=1\|0) |
| `move_forward` / `turn` / `move_relative` | 이동 |
| `assign_mission` / `abort_mission` / `scan_mission` | 임무 |
| `abort` | 진행 중인 모든 동작을 즉시 멈춘다 |
| `stream_start` / `stream_stop` | 스트림 |

**인지 엣지(edge)**

| action | 뜻 |
|---|---|
| `perceive_start` / `perceive_stop` | 추론 게이트 |
| `ingest_start` / `ingest_stop` | 말단에서 프레임 당기기 |
| `publish_start` / `publish_stop` | 판정을 말단에 반영 |
| `snapshot` | 한 장만 당긴다 |

**기종 이름을 어휘에 쓰지 않는다.** Go1이든 RoboMaster EP든 `drive_start`
하나로 말한다. 무엇을 실제로 올릴지는 링크 구현이 안다
([EXTENDING.md](EXTENDING.md)). 과거 이름 `sdk_start`/`sdk_stop`/`sdk_auto`는
같은 핸들러를 가리키는 별칭으로 남아 있다.

**어휘는 받는 쪽이 정한다.** 관제가 모르는 `action`을 보내면 노드가 거절하고
사유가 uplink로 올라온다. 관제에 어휘 목록을 두고 검사하면 노드가 어휘를 늘릴
때 그 목록이 낡고, 그 순간 "보냈는데 아무 일도 안 일어난다"가 된다.

### 물리 동작 표시

각 노드는 `PHYSICAL_ACTIONS`로 **물리 세계를 바꾸는 명령**을 선언한다. 승인
절차나 확인 대화를 붙일 자리를 규약이 알려 주는 것이다. 넓게 잡으면 의미를
잃으므로 실제로 물리 상태를 바꾸는 것만 넣는다 — 드론 엣지는 `publish_start`
하나뿐이다(추론·당기기는 엣지 안에서만 일어난다).

---

## 2. 프레임 — HTTP

드론 엣지와 드론 말단 사이. **엣지가 당긴다**(pull). 말단은 엣지 주소를 모르고
먼저 연결하지 않으므로, 말단 쪽에 포트포워딩이 필요 없다.

| 경로 | 뜻 |
|---|---|
| `GET /api/frame?cam=N&since=S&wait=W` | 최신 프레임. `since`보다 새 것이 없으면 `wait`초까지 기다리고 204 |
| `GET /api/spool` | 단절 구간에 쌓인 것 한 장 |
| `POST /api/spool/ack` | 받은 것을 지운다 |
| `POST /api/verdict` | 엣지 → 말단. 판정 반영 |
| `GET /api/health` | 상태 |

### 프레임 응답

`multipart/form-data; boundary=drone`으로 **메타와 픽셀을 나눠** 싣는다.
나중에 메타를 MQTT로, 픽셀을 별도 미디어 경로로 옮길 때 고칠 곳이 한 군데다.

```
--drone
Content-Disposition: form-data; name="meta"
Content-Type: application/json
{ "frame_id", "frame_seq", "camera_id", "captured_at",
  "frame_wh", "input_crop",
  "camera":   { "hfov_deg", "focal_px" },     // 거리 계산의 기하 근거
  "flow":     { "flow_ms", "flow_points", "dx", "dy", "scale" },
  "obstacles": [ ... ] }                       // 현재 좌표로 옮긴 판정
--drone
Content-Disposition: form-data; name="frame"; filename="frame.jpg"
Content-Type: image/jpeg
<JPEG>
--drone--
```

`since` + 롱폴이 핵심이다. 없으면 같은 프레임을 반복해서 받는다 — 실측으로
요청 137회 중 새 프레임이 14개뿐이었고 90%가 중복이었다.

### 단절

말단은 엣지가 `spool_after_idle_s` 동안 오지 않으면 스풀에 쌓는다. 링크가
돌아오면 엣지가 `/api/spool`로 훑어 가고 `ack`로 지운다. **전송 중에 끊겨도
다시 받을 수 있도록 ack에서 지운다** — 중복이 유실보다 낫다.

스풀에는 상한이 있다(개수·나이). 한계에 닿으면 오래된 것의 **픽셀부터** 버리고
메타는 남긴다. 버린 수는 헬스에 드러난다.

---

## 3. 스트리밍 — MJPEG

영상은 `multipart/x-mixed-replace; boundary=frame`으로 나간다. 브라우저
`<img>`가 그대로 받는다. WebSocket이나 별도 런타임을 들이지 않은 이유는, 이
경로에 필요한 것이 "최신 프레임을 계속 밀어 넣기"뿐이고 MJPEG가 표준 HTTP만으로
그걸 하기 때문이다.

파트마다 `Content-Type: image/jpeg`와 `Content-Length`가 붙는다.

---

## 4. 지키는 쪽이 알아야 할 것

**모르는 값은 0이 아니라 `null`이다.** 배터리 0%와 "아직 모른다"는 다르다.
규약의 모든 선택 필드가 이 규칙을 따른다.

**조용히 실패하지 않는다.** 버린 것은 센다(`dropped`, `async_dropped`,
`inference_dropped`), 실패한 것은 사유를 남긴다(`last_error`), 낡은 것은 나이를
드러낸다(`age_s`). 화면이 초록인데 실제로는 멈춰 있는 상태를 만들지 않는다.

**단계가 빠지면 그 단계만 빠진다.** 검출기가 없으면 검출만, 깊이가 없으면 거리만
빠지고 나머지는 그대로 돈다. 하나가 없다고 전체가 서지 않는다.
