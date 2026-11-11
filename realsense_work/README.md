# pi6 RealSense → data_stream

pi6 에 USB 로 붙은 **RealSense D455 의 컬러 스트림**을 data_stream 의 소스
`realsense`(포트 10005)로 들인다.

## 왜 이 모양인가

- **MJPEG 으로 낸다.** `server_stream_multi_source.py` 의 `robot_video_loop` 가 로봇에서
  읽는 모양(`GET /stream/<번호>` multipart MJPEG)과 같게 내면 **데이터 스트림 쪽에 새
  풀러를 만들 필요가 없다**. 실제로 더한 것은 소스 한 줄과 주소 한 줄뿐이다.
- **pyrealsense2 를 쓰지 않는다.** 이 파이(aarch64)에 librealsense 가 없고 바퀴도 없다.
  D455 는 UVC 라 컬러가 `/dev/video*` 로 그냥 보인다.
  **한계가 분명하다 — 깊이는 이 경로로 나오지 않는다.** 깊이가 필요해지면 그때 SDK 를 올린다.
- **상태(MQTT)는 없다.** pi6 에 브로커가 없다. `STATE_SOURCES` 에 넣지 않았으므로
  영상만 받고 상태는 비운다(없는 브로커에 붙으러 가면 로그가 재접속 경고로 덮인다).

## 띄우기

```bash
# pi6 에서
scp realsense_mjpeg.py pi6:~/
ssh pi6 '~/pf_runtime/venv/bin/python ~/realsense_mjpeg.py 8090 4 1280 720 15'
curl http://<pi6 주소>:8090/health               # 다른 기기에서 확인
```

재부팅에도 살리려면 `realsense-mjpeg.service` 를 설치한다(파일 머리에 명령이 있다).

## 받기

```bash
# hosts.local.json 에 "realsense": "<pi6 tailscale IP>" 를 적고
python server_stream_multi_source.py --on realsense
```

## 프레임 머리

MJPEG 파트마다 `Frame-Meta` 한 줄을 싣는다(DS-03). 받는 쪽이 그것을 `sender` 로 저장하고
비전이 `frame_ref{camera_id, sequence_id, timestamp}` 로 되짚는다.

`camera.hfov_deg 90` 은 **제원값**이고 관측이 아니다(`"source": "datasheet"`).
보정 전에는 거리·좌표의 확정 근거로 쓰지 않는다.
