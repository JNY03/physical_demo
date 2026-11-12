# 모듈 카메라 파이 → data_stream

CSI 로 붙은 **Raspberry Pi Camera Module 3(imx708)** 를 data_stream 의 소스로 들인다.
현재 두 대가 같은 파일을 쓴다 — **pi11**(포트 10008) · **pi12**(포트 10009).

## 왜 이 모양인가

- **MJPEG 으로 낸다.** `server_stream_multi_source.py` 의 `robot_video_loop` 가 로봇에서
  읽는 모양(`GET /stream/<번호>` multipart MJPEG)과 같게 내면 **데이터 스트림 쪽에 새
  풀러를 만들 필요가 없다**. pi6 의 `realsense_work/` 와 같은 선택이고, 실제로 더한 것은
  소스 한 줄과 주소 한 줄이다.
- **cv2.VideoCapture 를 쓰지 않는다.** 여기가 pi6 와 갈리는 자리다. pi6 의 D455 는 UVC 라
  `/dev/video*` 가 바로 컬러를 낸다. **모듈 카메라는 CSI 라 그 자리가 베이어 RAW 다** —
  같은 코드를 쓰면 장치는 열리고 쓸 수 없는 그림이 나온다. 그래서 libcamera 쪽
  (`rpicam-vid --codec mjpeg`)에서 받는다.
- **표준 라이브러리만 쓴다.** 이 파이(Pi 3 Model B, RAM 905MB)에는 `picamera2` 도 `cv2` 도
  없다. 설치를 요구하지 않으려고 `rpicam-vid` 의 stdout 에서 JPEG 경계(FFD8…FFD9)를 직접
  잘라낸다.

## 설치

파이에서:

```bash
scp picam_mjpeg.py physical@<pi11>:/home/physical/
ssh physical@<pi11>
sudo cp picam-mjpeg.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now picam-mjpeg
systemctl status picam-mjpeg --no-pager
```

손으로 띄울 때:

```bash
python3 /home/physical/picam_mjpeg.py 8090 0 1280 720 10
```

## 확인

```bash
curl -s http://<pi11>:8090/health          # opened:true · frames 가 늘어야 한다
```

브라우저로 `http://<pi11>:8090/` 를 열면 그대로 보인다.

받는 쪽:

```bash
python server_stream_multi_source.py --on pi11
```

## 확인된 값 (2026-11-12)

두 대 모두 Raspberry Pi 3 Model B Rev 1.2 · Debian 13 (trixie) · Python 3.13.5 이고,
카메라는 `imx708` (`/base/soc/i2c0mux/i2c@1/imx708@1a`, module ID `0x0301`) 이다.
설정은 1280x720 @10fps, JPEG quality 80.

| | pi11 | pi12 |
|---|---|---|
| 주소 | `hosts.local.json` 의 `pi11` | `hosts.local.json` 의 `pi12` |
<!-- 테일넷 주소는 git 에 넣지 않는다 — hosts.local.json(gitignore)·PD_HOST_PI11/PI12 에 둔다 -->
| data_stream 소스·포트 | `pi11` 10008 | `pi12` 10009 |
| 실측 | 83초 821장 ≈9.9fps · 30초 16.0MB | 40장 10.3fps |
| `read_mjpeg` 로 읽기 | 40장 연속 · seq 누락 0 | 40장 연속 · seq 누락 0 |
| systemd 서비스 | enabled · active | **아직 등록 안 됨** (손으로 띄운 상태) |

## 남은 것

- **두 대 모두 상이 180° 뒤집혀 나온다.** 카메라가 거꾸로 달린 것으로 보인다. 보정하려면
  `picam_mjpeg.py` 의 `_cmd()` 에 `--rotation 180` 을 더한다. 2026-11-12 현재는 **보정하지
  않은 상태**다 — 장착을 바로잡을지 소프트웨어로 돌릴지 정해지지 않았다.
- **pi12 는 렌즈 바로 앞을 연두색 물체가 화면의 약 40% 가린다.** 카메라 고장이 아니라
  앞에 놓인 물건이다. 치우기 전에는 그 영역에서 아무것도 보이지 않는다.
- **pi12 의 systemd 서비스가 아직 등록되지 않았다.** 지금은 손으로 띄운 상태라 재부팅하면
  사라진다. 위 "설치" 의 세 줄을 pi12 에서 실행하면 된다.
- 상태(MQTT)는 없다. pi11 에 브로커가 없어 `STATE_SOURCES` 에 넣지 않았다. 영상만 들어온다.
- `hfov_deg_spec: 66.0` 은 Camera Module 3 표준형의 **제원값**이다. 관측이 아니므로
  보정 전에는 근거로 쓰지 않는다.
