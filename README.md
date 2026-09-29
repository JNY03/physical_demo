# drone_perception

드론 말단(Raspberry Pi 5)과 엣지 노드로 나뉜 실시간 장애물 인지 파이프라인.
말단은 카메라와 프레임 간 움직임 추정(KLT)만 맡고, **검출·거리·의미 해석은 전부
엣지에서 돈다** — 말단에는 신경망이 없다.

```
drone_rpi/   말단 — 카메라 · KLT 플로우 정렬 · 프레임 제공 · 판정 반영
edge/        엣지 — YOLOE + YOLO-World 합의 · MoGe2-Aerial 거리 · 미검출 분석 · 스트리밍
```

엣지가 말단에서 프레임을 **당기고**(`GET /api/frame`), 판정이 끝나면 되밀어 준다
(`POST /api/verdict`). 말단은 받은 장애물을 KLT 누적 변환으로 현재 프레임 좌표에
맞춰 내보낸다. 연결은 전부 엣지 → 말단 한 방향이라 말단에 포트포워딩이 필요 없다.

## 실행

### 말단 (Raspberry Pi 5 + Camera Module 3)

```bash
cd drone_rpi
DRONE_HOST=<pi주소> DRONE_USER=<계정> ./deploy.sh
ssh <계정>@<pi주소> 'cd ~/drone_rpi && ./install_service.sh'   # 부팅 시 자동 시작
curl http://<pi주소>:8890/api/health
```

모델을 보내지 않는다. 의존성은 numpy와 opencv-headless 둘뿐이다. 자동 시작이
싫으면 `./venv/bin/python -u agent.py`로 직접 띄운다.

### 엣지 — WSL · Linux (CUDA GPU 권장)

```bash
cd edge
# config.json 의 terminals[0].host 를 말단 주소로 바꾼다
./fetch_models.sh        # 가중치 ~1.6GB (저장소에 없다)
./run.sh                 # http://127.0.0.1:8891/
./run.sh --port 9000
./run.sh stop            # 띄우지 않고 남은 엣지를 정리만 한다
```

`./run.sh`를 쓴다 — torch가 있는 파이썬을 찾아 주고, 못 찾으면 뜨지 않고 이유를
말한다. 가중치 존재도 미리 점검한다.

### 엣지 — PowerShell · cmd (WSL에 들어가지 않고)

엣지가 WSL(Ubuntu) 안에서 돌더라도 Windows 터미널에서 바로 띄운다.

```powershell
cd C:\...\physical_demo\edge
.\run.cmd                # 기동 → http://127.0.0.1:8891/
.\run.cmd --port 9000
.\run.cmd stop
```

`Ctrl-C` 한 번으로 전부 내려간다. `run.cmd`가 Git Bash를 찾아 `run.sh`에 넘기고,
`run.sh`는 자기가 Windows 위에 있는 것을 알아채면 WSL로 넘긴 뒤 **감시자로 남아**
Ctrl-C를 받으면 pid와 포트를 직접 보고 죽인다. wsl.exe가 콘솔 Ctrl-C를 삼키고
리눅스 쪽에 넘기지 않는 경우가 있어서, 신호 전달만 믿으면 창은 닫혔는데 서버가
포트를 쥔 채 남는다.

- **PowerShell에서 `bash run.sh`는 쓰지 않는다.** 거기서 `bash`는 System32의 WSL
  기본 배포판 실행기로 잡히고, 기본 배포판이 Ubuntu가 아니면 `/bin/bash`조차 없어
  죽는다. Git Bash 안에서라면 `./run.sh`가 그대로 된다.
- cmd는 Ctrl-C 뒤에 `Terminate batch job (Y/N)?`을 한 번 더 묻는다. 그 질문이 뜰
  때는 이미 정리가 끝난 뒤다.

### 화면과 제어

`http://127.0.0.1:8891/` — 카메라마다 두 줄. 위는 `original · depth · 장애물(박스·
클래스·거리 m)`, 아래는 놓친 후보의 crop과 의심 클래스다.

기동 직후 추론·보내기는 꺼져 있다. 화면의 버튼 두 개로 켜거나 CLI로:

```bash
python3 server.py ctl {fetch|infer-on|infer-off|push-on|push-off|pull-on|pull-off|status}
```

PowerShell에서는 같은 것을 HTTP로 부른다. `wsl -- bash -c "..."`로 감싸지 않는다 —
wsl.exe가 `--` 뒤를 한 줄로 이어 붙여 바깥 셸에 다시 먹이므로 따옴표 안의 변수가
거기서 먼저 펼쳐져 빈 값이 된다.

```powershell
$api = "http://127.0.0.1:8891/api/command"
irm -Method Post -Uri $api -ContentType application/json -Body '{"command":"terminal_status"}'
irm -Method Post -Uri $api -ContentType application/json -Body '{"command":"start_inference"}'
# start_pull / stop_pull / start_push / stop_push / stop_inference / fetch_image
```

## 다른 기기에서 구독하기

엣지에서 `.\run.cmd`(또는 `./run.sh`)로 띄우면 그걸로 끝이다. 같은 tailnet의 다른
기기에서 아래 주소로 바로 붙는다 — 서로 다른 네트워크여도 된다.

```
http://100.114.96.78:8891/          # 현재 엣지 노드
http://desktop-tcj72rt-1:8891/      # MagicDNS 이름도 같다
```

브라우저로 열면 화면이 그대로 나온다. 스트림 하나만 가져가려면 경로를 붙인다.

```bash
curl http://100.114.96.78:8891/api/obstacles?camera=0            # 추론 결과(JSON)
curl -o out.mjpg http://100.114.96.78:8891/original?camera=0     # 영상
```

### 스트림

MJPEG는 `multipart/x-mixed-replace; boundary=frame`이고 파트마다
`Content-Type: image/jpeg`와 `Content-Length`가 붙는다. 최대 12 fps, JPEG q70.
`camera=0`은 카메라 인덱스다.

| 경로 | 내용 | 대역 |
|---|---|---|
| `/panels` | `original`+`depth`+`장애물` 세 칸을 옆으로 붙인 것 (3840×720) | 780 KB/s |
| `/original` | 말단이 보낸 원본 | 131 KB/s |
| `/depth` | MoGe2-Aerial 거리 컬러맵 | |
| `/stream` | 원본 위에 박스·클래스·거리를 얹은 것 | 118 KB/s |
| `/missed` | 검출이 놓친 후보들의 crop 격자 | |

| 경로 | 내용 |
|---|---|
| `/api/obstacles` | 장애물마다 `box·label·conf·range·age_s·quality·moved_px`, 프레임 `flow` |
| `/api/live` | 카메라별 `frame_id·seq·n_obstacles·age_s` |
| `/api/health` | 단계 가동 상태 |

푸시 구독(WebSocket·SSE)은 없다 — MJPEG와 폴링뿐이다. 인증도 없다. tailnet ACL이
유일한 접근 통제다.

## 라이선스 주의

- MoGe-2 / MoGe2-Aerial — MIT
- YOLOE / YOLO-World(ultralytics) — AGPL-3.0
- UniDepthV2(CC BY-NC), Metric3Dv2(BSD-2 비상업)는 **쓰지 않는다** — 상업 이용 제한
