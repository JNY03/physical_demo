# drone_perception

드론 말단(Raspberry Pi 5)과 엣지 노드로 나뉜 실시간 장애물 인지 파이프라인.

```
drone_rpi/   말단 — 카메라 · KLT 플로우 정렬 · 프레임 제공 · 판정 반영
edge/        엣지 — YOLOE + YOLO-World 합의 · MoGe2-Aerial 거리 · 미검출 분석 · 스트리밍
```

## 설계

**말단에는 신경망이 없다.** 검출·거리·의미 해석은 전부 엣지 몫이다. 말단이 하는
일은 셋뿐 — 캡처, 프레임 간 움직임 추정(KLT), 엣지가 준 장애물을 현재 좌표로
옮겨 내보내기.

```
  엣지                                  말단(Pi)
   │  GET /api/frame?cam=0&since=N&wait=2  ──▶  최신 프레임 (없으면 204, 롱폴)
   │  ◀──────────  같은 연결의 응답 본문(meta + jpeg)
   │
   │  YOLOE + YOLO-World 합의 → 확정 장애물
   │  POST /api/verdict  ─────────────────▶  KLT로 현재 좌표에 맞춰 반영
   │
   │  (비동기) MoGe2-Aerial 거리 · 미검출 분석
```

### 왜 이렇게 나뉘는가

**시점 정렬에 필요한 것은 움직임이지 정체성이다.** 예전에는 말단이 RPN으로
후보를 만들고 ByteTrack으로 track_id를 붙여 거기 판정을 매달았다. Pi 5 실측
프레임당 350ms였고, KLT 플로우가 같은 일을 6ms에 한다(58배).

| | 이전(RPN+ByteTrack) | 지금(KLT 정렬) |
|---|---|---|
| 프레임률 | 1.40 fps | **8.19 fps** (카메라 상한) |
| CPU | 277% | **25.6%** |
| 온도 | 85.6°C (스로틀링) | **63.1°C** |
| 클럭 | 1.5 GHz | **2.4 GHz** (정격) |

**엣지가 당긴다(pull).** 당기는 속도가 곧 backpressure다. 요청한 IP를 되찾아
되쏘지 않고 같은 연결의 응답에 싣는다 — NAT·LTE에서 안 깨지고 왕복도 절반이다.
`?since=<seq>`로 같은 프레임을 다시 받지 않는다(실측 중복 90% → 0%).

**정체성은 엣지가 소유한다.** 말단에 추적기가 없으므로 정체성 공간이 하나뿐이고
어긋날 곳이 없다.

### 거리는 왜 엣지에만 있나

Pi 5에서 못 돈다. 실측 GEMM 75.1 GFLOPS인데 MoGe2-Aerial ViT-L은 700×392에서
1.04 TFLOPs다(≈35초/프레임). 그래서 거리는 늦게 오고, 말단이 **트랙 박스 크기
변화**로 현재 시점에 당긴다(`Z_now ≈ Z_obs / 누적배율`). 추가 센서도 통신도 없다.

이건 배율 보정이지 pose 보정이 아니다 — `ego_compensated`는 여전히 `False`다.
MAVLink로 고도·자세가 들어오기 전까지 그렇게 부르지 않는다.

## 빠른 시작

### 1. 말단 (Raspberry Pi 5 + Camera Module 3)

```bash
cd drone_rpi
DRONE_HOST=<pi주소> DRONE_USER=<계정> ./deploy.sh
ssh <계정>@<pi주소> 'cd ~/drone_rpi && ./install_service.sh'   # 부팅 시 자동 시작
curl http://<pi주소>:8890/api/health
```

**모델을 보내지 않는다** — 말단에 신경망이 없다. 의존성은 numpy와
opencv-headless 둘뿐이다.

자동 시작을 원하지 않으면 `./venv/bin/python -u agent.py`로 직접 띄운다.

### 2. 엣지 (CUDA GPU 권장)

```bash
cd edge
# config.json 의 terminals[0].host 를 말단 주소로 바꾼다
./fetch_models.sh        # 가중치 ~1.6GB (저장소에 없다)
./run.sh                 # http://127.0.0.1:8891/
```

**`./run.sh`를 쓴다.** torch가 있는 파이썬을 찾아 주고, 못 찾으면 뜨지 않고
이유를 말한다. 가중치 존재도 미리 점검한다.

## 화면

카메라마다 두 줄. `http://127.0.0.1:8891/`

**위 — 무엇을 잡았나**: `original · depth · obstacle information(박스·클래스·거리 m)`

**아래 — 무엇을 놓쳤나**: 미검출 후보의 crop + 의심 클래스 + 거리/실제 크기

```
s2  seen 8x  obj 1.00
1.1m | 56x59cm | +34cm
의심:
  flipchart 0.85 (iou 0.63)
  laptop 0.37 (iou 0.38)
```

의심 클래스는 **확정 문턱을 못 넘은 OVD 검출**이다 — 추가 연산 0이고, 겹치는
검출이 아예 없으면 `OVD 반응 없음`으로 뜬다(가장 강한 미검출 신호).

## 제어

버튼 두 개(`이미지 가져오기/정지`, `추론 시작/종료`)이거나 CLI:

```bash
python server.py ctl {fetch|infer-on|infer-off|push-on|push-off|pull-on|pull-off|status}
```

기동 직후 추론·보내기는 꺼져 있다 — 모델 적재 중에 말단 레코드가 바뀌면
무엇이 언제부터 반영됐는지 경계가 흐려진다.

## 측정된 값 (RTX 3060 / Pi 5)

| 단계 | 시간 | 비고 |
|---|---|---|
| 말단 KLT 정렬 | 6~9 ms | 8 fps 유지 |
| YOLOE | 15 ms | 어휘 4585종 |
| YOLO-World | 16 ms | 현장 어휘 프롬프트 |
| MoGe2-Aerial | 280~500 ms | **비동기** (손익분기 209ms 초과) |
| 미검출 분석 | 138 ms | RPN 54ms + 기하 규칙 |

## 알려진 제약

- **거리에 ego-motion 보정이 없다.** 말단에 MAVLink 텔레메트리가 없다. 고도·자세가
  들어오면 지면 평면 교차로 훨씬 정확해진다(자세 0.1°면 고도 4.4m·거리 50m에서 2%).
- **장거리를 막는 것은 모델이 아니라 렌즈다.** Camera Module 3 Wide(2.75mm)는
  1280×720에서 focal_px ≈ 590이라 100m의 사람이 10px이다. 표준(4.74mm)이면 1.9배.
- **미검출 슬롯은 IoU 매칭이라 시야가 빠르게 움직이면 끊긴다.** 끊기면 패널이
  비는 쪽으로 틀린다 — 엉뚱한 것을 오래 본 것처럼 보이는 것보다 낫다.
- **depth 지연은 이 GPU 기준이다.** 다른 기기에서는 다시 재고 `depth.mode`를
  정해야 한다(손익분기 209ms).

## 라이선스 주의

- MoGe-2 / MoGe2-Aerial — MIT
- YOLOE / YOLO-World(ultralytics) — AGPL-3.0
- UniDepthV2(CC BY-NC), Metric3Dv2(BSD-2 비상업)는 **쓰지 않는다** — 상업 이용 제한
