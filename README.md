# physical_demo

로봇·드론의 **인지와 원격 제어를 한 벌로 묶은 시스템**. 기기와 모델은 설정으로
갈아끼운다 — 이 저장소에 특정 기기 주소나 특정 모델이 전제로 박혀 있지 않다.

> **정리 중입니다.** 네 갈래로 따로 자라던 것을 하나로 합치는 중이고, 아래 구성과
> 문서는 그 과정에서 계속 바뀝니다. 통합 전 각 브랜치는 `origin/drone`,
> `origin/main`, `origin/data_stream`, `origin/go1_raspi`에 그대로 남아 있습니다.

## 구성

| 경로 | 무엇인가 | 문서 |
|---|---|---|
| `edge/` | 엣지 노드 — 검출·거리·스트리밍. 말단에서 프레임을 당겨 판정을 되돌려 준다 | [docs/drone-edge.md](docs/drone-edge.md) |
| `drone_rpi/` | 드론 말단 — 카메라와 KLT 플로우 정렬만. 신경망이 없다 | [docs/drone-edge.md](docs/drone-edge.md) |
| `hw/` | 4족 로봇(Go1) 온디바이스 노드 — 구동·센서·MQTT 브로커 | [docs/go1-raspi.md](docs/go1-raspi.md) |
| `data_stream/` | 다중 소스 스트림 수집과 GPS·자세 기반 객체 위치 추정 | `data_stream/` 내 문서 |

## 설계

**기기는 설정이다.** 주소·카메라·포트는 전부 설정 파일이나 환경변수로 들어간다.
저장소에 특정 기기 주소를 두지 않는다 — 다른 기기에서 받아 그대로 쓰면 조용히
엉뚱한 곳으로 가기 때문이다.

**모델도 설정이다.** 검출기는 `detect(bgr) → [{box, label, conf}]` 한 규약만
지키면 된다. 지금 세 구현이 같은 슬롯에 꽂힌다 — YOLOE prompt-free(4585종),
YOLO-World(텍스트 어휘), 그리고 모델을 전혀 쓰지 않는 색 분할. 깊이도 마찬가지로
provider로 고른다. 단계가 빠지면 그 단계만 빠지고 나머지는 그대로 돈다.

**계층 사이는 문서화된 규약으로 이어진다.** 드론 쪽은 HTTP 2계층(엣지가 당기고
판정을 되민다), 로봇 쪽은 MQTT다. 통합 제어면은 이 둘을 한 버스 위에 올린다.

## 지금 할 수 있는 것

```bash
cd edge && ./run.sh        # 엣지 → http://127.0.0.1:8891/
```

Windows에서는 `.\run.ps1`. 자세한 것은 [docs/drone-edge.md](docs/drone-edge.md)에
있다.

## 라이선스

**아직 정하지 않았습니다.** 그때까지 이 저장소의 코드에는 아무 사용 허가도 붙어
있지 않습니다.

의존성 중 ultralytics(YOLOE·YOLO-World)가 **AGPL-3.0**이라는 점이 선택에 영향을
줍니다. 다만 검출기를 색 분할 provider로 두면 ultralytics 없이도 전체 파이프라인이
돌기 때문에, 그 의존성을 선택 사항으로 두는 구성이 가능합니다.
