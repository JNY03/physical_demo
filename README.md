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

## 다른 기기에서 구독하기 (Tailscale)

**엣지·말단·구독자가 서로 다른 네트워크여도 된다.** 주소가 Tailscale IP 하나뿐이라
물리 네트워크가 바뀌어도 설정을 고치지 않고, 연결 방향이 전부 한쪽이라(엣지→말단,
구독자→엣지) 말단에도 구독자에도 포트포워딩이 필요 없다.

NAT 홀펀칭으로 직접 경로가 뚫리면 인터넷 RTT만 더해지고, 안 뚫려도 DERP 중계로
붙는다 — 끊기지 않는다. 다른 망의 노드로 실측했을 때 직접 경로는 4~10ms, 중계로
빠진 노드는 432ms였다(DERP 지역 지연 tok 37ms · hkg 87ms · sea 144ms).

### 엣지를 tailnet에 내보내기

엣지는 `0.0.0.0:8891`로 열리므로 리눅스에서 직접 돌린다면 그 기기의 Tailscale IP로
바로 붙으면 된다.

**WSL에서 돌린다면 WSL의 tailnet IP를 그대로 주면 안 된다** — 이미지가 나가지
않는다. WSL의 `eth0`과 `tailscale0` MTU가 둘 다 1280이라 WireGuard로 감싼 뒤
eth0을 넘겨 큰 패킷이 통째로 버려진다(JSON은 통과하고 MJPEG만 0바이트로 온다).
Windows 쪽 Tailscale 노드로 중계한다:

```powershell
python .\edge\windows_pi_proxy.py --listen-host <Windows의 Tailscale IP> --listen-port 8891 --target-host 127.0.0.1 --target-port 8891
tailscale serve --bg --http=8891 http://127.0.0.1:8891     # 재부팅 후에도 유지
```

WSL MTU를 고쳐 직접 여는 쪽도 된다 — WSL 안에서 `sudo ip link set dev tailscale0 mtu 1180`.

### 구독할 수 있는 것

`base = http://<엣지 Tailscale IP>:8891`. MJPEG는
`multipart/x-mixed-replace; boundary=frame`이고 파트마다 `Content-Type: image/jpeg`와
`Content-Length`가 붙는다. 최대 12 fps, JPEG q70.

| 경로 | 내용 | 실측 대역 |
|---|---|---|
| `/panels?camera=0` | original + depth + 장애물 3칸 (3840×720) | 780 KB/s |
| `/original?camera=0` | 원본만 | 131 KB/s |
| `/stream?camera=0` | 장애물 오버레이만 | 118 KB/s |
| `/depth?camera=0` · `/missed?camera=0` | depth 컬러맵 · 놓친 후보 격자 | |

추론 결과 원본은 `/api/obstacles?camera=0`이다 — 프레임별 `flow`와 장애물마다
`box·label·conf·range·age_s·frames_late·quality·moved_px`. 그 밖에 `/api/live`,
`/api/health`. 푸시 구독(WebSocket·SSE)은 없다.

`/panels`는 6 Mbps쯤 든다 — DERP 중계로는 버겁다. 중계 경로가 예상되면
`/original`이나 `/stream`(1 Mbps)에 `/api/obstacles` 폴링을 붙이는 편이 낫다.

**인증이 없다.** tailnet ACL이 유일한 접근 통제다.

```python
import requests
BASE = "http://100.x.y.z:8891"

d = requests.get(f"{BASE}/api/obstacles?camera=0", timeout=5).json()
for o in d["obstacles"]:
    print(o["label"], o["conf"], o["box"], o.get("range"))

r = requests.get(f"{BASE}/original?camera=0", stream=True, timeout=(3, 30))
buf = b""
for chunk in r.iter_content(8192):
    buf += chunk
    while True:
        s, e = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
        if s < 0 or e < 0 or e < s:
            break
        jpeg, buf = buf[s:e + 2], buf[e + 2:]
```

## 라이선스 주의

- MoGe-2 / MoGe2-Aerial — MIT
- YOLOE / YOLO-World(ultralytics) — AGPL-3.0
- UniDepthV2(CC BY-NC), Metric3Dv2(BSD-2 비상업)는 **쓰지 않는다** — 상업 이용 제한
