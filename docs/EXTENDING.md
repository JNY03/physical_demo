# 확장하기

내 기기, 내 모델로 바꿔 끼우는 방법. [PROTOCOL.md](PROTOCOL.md)가 "무엇을 지켜야
하나"이고, 이 문서는 "어디를 고치면 되나"다.

원칙은 하나다. **기종과 모델 이름은 한 파일 안에만 있다.** 그 파일 밖으로 새면
갈아끼울 때 전부 고쳐야 하므로, 아래 이음새는 모두 "여기만 고치면 밖은 그대로"가
되도록 잡혀 있다.

| 바꾸려는 것 | 고칠 곳 | 밖에서 할 일 |
|---|---|---|
| 로봇·구동기 기종 | `hw/pi/robot/controller_link.py` 의 `create()` + 링크 구현 1개 | `HW_CONTROLLER_LINK=<kind>` |
| 1차 검출기 | `edge/models.py` 에 클래스 1개 | `ovd.provider` 한 줄 |
| 깊이/거리 | `edge/models.py` 에 클래스 1개 | `depth.provider` 한 줄 |
| 카메라 종류 | `drone_rpi/agent.py` 의 `_argv()` | `cameras[].provider` |
| 노드 종류를 새로 | `hw/pi/<역할>/<역할>_node.py` (BaseNode 상속) | k3s DaemonSet 1개 |
| 주소·식별자 | 아무 코드도 아님 | 환경변수 / ConfigMap |

---

## 1. 다른 로봇을 붙인다

`ControllerLink` 추상 클래스의 **메서드 셋만** 채운다
(`hw/pi/robot/controller_link.py`).

```python
class MyLink(ControllerLink):
    def read_state(self) -> RobotState: ...   # 내부 수집. 전송 주기보다 빠르게
    def send_command(self, action, params): ...  # 중립 명령 → 기종 명령
    def link_health(self) -> str: ...         # ok | degraded | fault
```

그리고 팩토리에 한 줄:

```python
def create(kind="sim"):
    if kind == "sim": return SimLink()
    if kind == "mine":
        from robot.my_link import MyLink   # 의존을 여기서만 진다
        return MyLink()
```

`HW_CONTROLLER_LINK=mine` 으로 고른다. **import를 분기 안에 둔다** — 그래야 내
기종의 SDK가 설치되지 않은 기기에서도 나머지가 돌아간다. 지금 `go1`은 paho,
`ep`는 robomaster SDK를 각자 분기 안에서만 진다.

### 구동을 따로 올려야 하는 기종

로봇이 "일어서야" 움직이는 기종이 있다. 그때만 셋을 덮어쓴다.

```python
def drive_up(self)    -> dict: ...   # {"already", "waited_s", "supported"}
def drive_down(self)  -> dict: ...
def drive_probe(self) -> tuple[bool, bool]   # (올라와 있나, 명령 받을 준비됐나)
```

기본 구현은 **아무것도 하지 않고 `supported: False`를 말한다.** 구동 분리가 없는
기종(`sim`, `ep`)은 손대지 않아도 그대로 돈다 — 모르는 것을 0이나 True로
속이지 않는다.

`drive_probe`가 둘을 나누는 이유: 브리지가 떴다고 바로 쓸 수 있는 게 아니다.
기종에 따라 상태 보고가 올라오기까지 시간이 걸리고, 그 전에 명령을 걸면 열린
루프가 된다.

**여기가 기종 이름을 막는 자리다.** 예전에는 `robot_node`가 Go1의 systemd 유닛을
직접 불러서 명령 어휘에 `sdk_start`라는 Unitree SDK 수명주기가 그대로 드러나
있었다. EP로 갈아타면 그 이름은 거짓이 된다. 지금은 관제가 `drive_start`만
말하고, 무엇을 올릴지는 링크가 안다.

---

## 2. 다른 검출기를 붙인다

`detect(bgr) -> list[dict]` 하나다.

```python
class MyOvd:
    def __init__(self, cfg: dict): ...        # edge/config.json 전체를 받는다
    def detect(self, bgr) -> list[dict]:
        return [{"box": [x1, y1, x2, y2],     # 원본 해상도 좌표, float
                 "label": "tunnel",            # 사람이 읽는 이름
                 "conf": 0.87}]                # 0~1
```

`edge/server.py`의 provider 분기에 한 줄 더하고, `ovd.provider`로 고른다. 지금 두
구현이 같은 규약을 지킨다.

- `yoloe` — 내장 어휘 4585종. 프롬프트 없이 흔한 물체를 넓게 잡는다
- `color` — HSV 색 분할. 어휘가 둘(`label`, `hole_label`)뿐이고 `conf`는 항상 1.0

**둘의 차이가 규약 밖으로 새지 않는다.** 색 분할로 바꿔도 추적·거리·반영·화면이
전부 그대로 돌고, 되돌리는 것은 `ovd.provider` 한 줄이다. 결정적 검출기의
`conf`는 늘 1.00이라 정보가 없으므로 화면에 찍지 않도록 `stream.show_conf: false`를
같이 둔다.

선택 필드를 더 실을 수 있다. `area_px`, `source_kind`는 화면과 기록이 읽고, 없으면
그 칸만 빈다.

### 거리

`depth.provider`도 같은 구조다. 단 **단위를 속이지 않는 것**이 조건이다.

- metric provider(`moge2_aerial`)만 미터를 낸다
- 상대값만 내는 provider(`zipdepth`)는 깊이 맵만 내고 화면은 m를 붙이지 않는다

상대 역depth에 m를 붙이면 숫자는 그럴듯한데 전부 틀린 값이 된다. 새 provider를
붙일 때 metric인지 아닌지를 먼저 정하고, 아니면 미터를 내지 않는다.

### 단계를 빼기

`enabled: false`로 어느 단계든 끈다. 검출기가 없으면 검출만, 깊이가 없으면 거리만
빠지고 나머지는 돈다. `ovd2`(교차 확인), `clip`(의미 점수), `llm`/`vlm`(장면 해석)은
전부 꺼진 상태로도 파이프라인이 완결된다.

---

## 3. 다른 카메라를 붙인다

`drone_rpi/agent.py`의 `_argv()` 한 곳. 카메라마다 `provider`를 고른다.

| provider | 쓰는 곳 | 필요한 것 |
|---|---|---|
| `rpicam` | Raspberry Pi 카메라(libcamera) | `index` |
| `v4l2` | 일반 USB/CSI | `device`(`/dev/video0`) 또는 `index`, ffmpeg |
| `rtsp` | IP 카메라·NVR | `device`에 `rtsp://` URL, ffmpeg |

공통 전제는 하나다 — **MJPEG를 stdout으로 흘릴 것.** 그래서 provider를 늘려도
프레임 파싱은 그대로다. RTSP는 TCP로 고정한다. UDP는 프레임이 깨져도 조용히
넘어가서 카메라 문제인지 망 문제인지 못 가린다.

한 기기에 Pi 카메라와 USB 카메라를 섞어 달 수 있다. `cameras` 배열에 따로 적으면
워커가 그만큼 뜬다.

### 카메라 기하

`hfov_deg`와 `focal_px`를 말단 설정에 적는다. 엣지 depth가 metric 스케일을 잡는 데
쓴다 — 아는 값을 안 넘기면 모델이 추정하고, 항공 영상 실측에서 그 차이가 δ₁
5~34%였다. 저장소에 들어 있는 값은 특정 렌즈의 역산값이고 **캘리브레이션 값이
아니다.** 체커보드로 재서 덮어쓴다.

---

## 4. 노드를 새로 만든다

`BaseNode`를 상속하고 훅만 채운다(`hw/pi/common/node.py`). 등록·하트비트·명령
4단계·스풀·OTLP는 이미 들어 있다.

```python
class MyNode(BaseNode):
    ENTITY_TYPE = "mine"                    # 토픽 2번째 칸
    ACTIONS = dict(BASE_ACTIONS, **{
        "do_thing": _act_do_thing,          # (stage, detail) 제너레이터
    })
    PHYSICAL_ACTIONS = frozenset({"do_thing"})

    def on_sample(self, now): ...           # 계측 수집과 보고 판정
```

핸들러는 제너레이터다. `yield "executing", None` → 일하고 →
`yield "completed", {...}`. 거절은 `CommandError(code, detail)`을 올린다.

**`on_sample`에서 블로킹하지 않는다.** 메인 루프에서 불리므로 거기서 HTTP를 직접
때리면 그 시간만큼 루프가 선다. 샘플 간격 1초에 타임아웃 5초면 느린 응답 하나로
하트비트가 밀리고 브로커가 keepalive 초과로 끊는다. 실측으로 rc=141과 미발신
100여 건, 그리고 downlink 구독까지 잃었다. 느린 수집은 스레드로 떼고
`on_sample`은 마지막 표본과 그 나이만 올린다 — `edge/mqtt_node.py`의 poller가
그 꼴이다.

`PHYSICAL_ACTIONS`는 **실제로 물리 상태를 바꾸는 것만** 넣는다. 넓게 잡으면
승인 절차가 의미를 잃는다.

---

## 5. 주소와 식별자

**저장소에 기기 주소가 없다.** 전부 환경변수이고, k3s에서는 ConfigMap 하나
(`deploy/k3s/10-config.yaml`)에서 온다.

| 변수 | 뜻 |
|---|---|
| `HW_BROKER_HOST` / `HW_BROKER_PORT` | MQTT 브로커 |
| `HW_ENTITY_ID` | `device_id`. k3s에서는 `spec.nodeName` |
| `HW_NODE_ID` / `HW_ZONE_ID` | 소속 |
| `HW_CONTROLLER_LINK` | 링크 구현 (기본 `sim`) |
| `HW_MQTT_V5` | 1=MQTT5(기본), 0=3.1.1 |
| `HW_SPOOL_PATH` | 스풀 파일 |
| `EDGE_API` | 엣지 서버 HTTP 창구 |

엣지는 `edge/config.json`의 `terminals[].host`에 말단 주소를 적는다. 받은 그대로는
`CHANGE-ME`다. 가시화는 `--robot1`/`--drone` 같은 인자로 주소를 받고
`--device-id 소스=id`로 MQTT 대상을 맞춘다.

`device_id`가 겹치면 MQTT client_id가 겹쳐 브로커가 둘을 번갈아 끊는다. 기기마다
다른 값을 준다.

**MQTT 버전을 확인한다.** 브로커가 MQTT 5를 지원하지 않으면 기본값으로는 즉시
끊긴다(mosquitto는 되고, amqtt는 안 된다). `HW_MQTT_V5=0`으로 내린다.

---

## 6. 가시화 화면에 붙인다

`data_stream/server_stream_multi_source.py`가 소스 4개를 한 화면에 모으고 같은
어휘로 명령을 보낸다. 새 종류를 붙일 때 고칠 곳은 둘이다.

```python
SOURCE_KIND = {"drone": "edge", ...}        # 소스 이름 → 역할
UI_ACTIONS  = {"edge": [...], "robot": [...]}  # 역할 → 버튼
```

**관제에 어휘 목록을 두는 것 자체가 위험하다.** 노드가 어휘를 늘리면 이 목록이
낡고, 그 순간 "보냈는데 아무 일도 안 일어난다"가 된다. 그래서 모르는 action은
노드가 거절하고 사유를 uplink로 올린다 — 화면은 그 사유를 그대로 보여 준다.

명령은 **protobuf로 만들어야 한다**(`_build_command()`). JSON을 발행하면 토픽에
도착은 하지만 노드가 파싱에 실패해 버린다. 발행 쪽에는 성공으로 보이므로 특히
주의한다.

---

## 7. 배포

### 로컬에서 먼저

기기도 모델도 없이 돌려 볼 수 있다. `HW_CONTROLLER_LINK=sim`에
`ovd.enabled: false`면 명령 4단계·보고 주기·버퍼링·화면이 전부 검증된다.

`edge/Dockerfile.lean`은 torch와 CUDA 없이 `opencv-python-headless`,
`numpy`, `onnxruntime`, `requests`만 담는다. 색 분할 provider는 이것으로 충분하다.

### k3s

`deploy/k3s/`에 네 개. 노드 역할은 **라벨로** 고른다.

```bash
kubectl label node <노드> physical.io/role=robot
kubectl apply -f deploy/k3s/
```

말단 노드는 DaemonSet이다. 역할 라벨이 붙은 기기마다 하나씩 뜨고 `HW_ENTITY_ID`를
노드 이름에서 끌어온다 — 기기를 추가할 때 YAML을 고치지 않는다.

**SDK는 이미지에 굽지 않고 마운트한다.** 기종별 SDK는 라이선스도 크기도 제각각이고,
굽는 순간 이미지가 기종에 못 박힌다.

### 주의 — 말단 코어는 k3s에 넣지 않는다

`hw/pi/common/node.py`는 네트워크에 의존하지 않도록 만들어져 있다. 브로커·엣지·
오케스트레이터가 전부 죽어도 계측과 판정과 버퍼링은 계속된다. 그래서 그 층은
systemd로 상주시킨다 — k3s agent는 제어평면에 연결되지 못하면 기동조차 못한다.
k3s로 올리는 것은 **그 위의 교체 가능한 부분**이다.

---

## 8. 확장할 때 지킬 것

**모르면 `null`이다.** 0은 측정값이다. 배터리 0%와 "아직 모른다"를 같은 값으로
내보내면 화면이 거짓말을 한다.

**버린 것을 센다.** 큐가 넘쳐서 버렸으면 그 수를 올린다
(`dropped`, `async_dropped`, `inference_dropped`). 조용히 버리면 "느린가?"와
"멈췄나?"를 구분할 수 없다.

**낡은 것은 나이를 드러낸다.** 캐시된 상태에는 `age_s`를 붙인다. 화면이 초록인데
실제로는 15분 전 값인 상태를 만들지 않는다 — 실측으로 겪었다.

**죽을 때는 죽는다.** 카메라가 조용해지면 예외를 내고 프로세스를 끝내 systemd가
다시 띄우게 한다. 프로세스가 살아 있으면 `Restart=always`가 걸리지 않아서,
`/api/health`는 200을 주는데 `frame_seq`는 15분간 고정이었다.

**기종 이름을 어휘에 쓰지 않는다.** 한 번 새면 그 뒤로 모든 기종이 그 이름을
따라 해야 한다.
