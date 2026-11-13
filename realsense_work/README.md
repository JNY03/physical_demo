# ds5 RealSense → data_stream

ds5 에 USB 로 붙은 **RealSense D455 의 컬러 스트림**을 data_stream 의 소스
`realsense`(포트 10005)로 들인다.

> 2026-11-13 카메라를 **pi6 에서 ds5 로** 옮겼다. ds5 에도 같은 경로(`~/pf_runtime/venv`,
> `~/realsense_mjpeg.py`)와 같은 서비스를 깔았고 컬러 노드도 같은 `/dev/video4` 다.
> pi6 의 `realsense-mjpeg` 서비스는 disable 했다. `camera_id` 는 기기 이름에서 따므로
> 지금은 `ds5-realsense-d455-color` 다(예전 `pi6-realsense-d455-color`).

## 왜 이 모양인가

- **MJPEG 으로 낸다.** `server_stream_multi_source.py` 의 `robot_video_loop` 가 로봇에서
  읽는 모양(`GET /stream/<번호>` multipart MJPEG)과 같게 내면 **데이터 스트림 쪽에 새
  풀러를 만들 필요가 없다**. 실제로 더한 것은 소스 한 줄과 주소 한 줄뿐이다.
- **pyrealsense2 를 쓰지 않는다.** 이 파이(aarch64)에 librealsense 가 없고 바퀴도 없다.
  D455 는 UVC 라 컬러가 `/dev/video*` 로 그냥 보인다.

### 깊이에 대해 — 앞서 적었던 것을 고친다

처음에 "UVC 라 깊이는 이 경로로 나오지 않는다" 고 적었다. **틀렸다.** 깊이는 나온다:

```
$ v4l2-ctl -d /dev/video0 --list-formats
    [0]: 'Z16 ' (16-bit Depth)      # 256x144 … 848x480 … 1280x720
$ v4l2-ctl -d /dev/video0 --set-fmt-video=width=848,height=480,pixelformat='Z16 ' \
           --stream-mmap --stream-count=1 --stream-to=/tmp/depth.raw
    814,080 바이트 (848*480*2) · 0 아닌 화소 13.2% · 최소 505 · 중앙 3014 · 최대 65535
```

깊이는 D455 의 ASIC 이 **카메라 안에서** 계산해 Z16 노드로 내보낸다. 받는 것 자체는
SDK 없이 된다. 그런데 **그 숫자가 무엇인지를 UVC 가 말해 주지 않는다.** 없는 것은 이것들이다.

1. **단위(depth scale).** 위의 중앙값 3014 를 3.014 m 로 읽은 것은 **가정**이다.
   D4xx 의 기본이 1 mm/단위일 뿐, 실제 값은 장치 설정이고 librealsense 가 읽어 준다.
   스트림에는 실려 오지 않는다.
2. **내부 파라미터(intrinsics).** 깊이 화소를 3D 점으로 펴려면 깊이 이미저의
   fx·fy·cx·cy 가 필요하다. 카메라 EEPROM 의 보정값이고 SDK 가 꺼내 준다.
3. **깊이↔컬러 외부 파라미터(extrinsics).** 깊이와 컬러는 **다른 이미저**이고 화각도
   해상도도 다르다. 컬러에서 잡은 상자에 거리를 붙이려면 둘을 정렬해야 하는데
   그 변환도 SDK 쪽이다.
4. **두 스트림의 짝맞춤.** `/dev/video0`(깊이)과 `/dev/video4`(컬러)는 **각각 따로 도는**
   V4L2 스트림이다. 같은 순간의 두 장을 짝지으려면 SDK 의 프레임 동기나
   메타데이터 노드(`/dev/video1`·`3`·`5`) 파싱이 필요하다.
5. **후처리.** 위 한 장은 화소의 **86.8% 가 0**(깊이 없음)이고 최대가 65535(무효 표시)다.
   구멍 메움·시공간 필터 없이는 그대로 쓰기 어렵다.
6. **전송.** 이 경로는 JPEG 을 나른다. 깊이는 16비트 1채널이라 JPEG 으로 누르면
   정밀도가 깨진다 — PNG-16 이나 raw 가 필요하고, 그건 다른 전송 결정이다.

그래서 **"깊이가 안 나온다" 가 아니라 "깊이를 쓰려면 librealsense 가 필요하다"** 가 맞다.
지금 이 소스는 컬러만 받는다. 깊이가 필요해지면 그 파이에 SDK 를 올린다.
- **상태(MQTT)는 없다.** 리얼센스 스크립트는 상태를 내지 않는다(ds5 의 1883 브로커는 이 스트림과 상관없다). `STATE_SOURCES` 에 넣지 않았으므로
  영상만 받고 상태는 비운다(없는 브로커에 붙으러 가면 로그가 재접속 경고로 덮인다).

## 띄우기

```bash
# 이 PC 에서 ds5 로 (WSL 에서는 ds5.local 이 안 풀린다 — tailscale 이름 ds5 를 쓴다)
scp realsense_mjpeg.py ds5:~/
ssh ds5 '~/pf_runtime/venv/bin/python ~/realsense_mjpeg.py 8090 4 1280 720 15'
curl http://<ds5 주소>:8090/health               # 다른 기기에서 확인
```

재부팅에도 살리려면 `realsense-mjpeg.service` 를 설치한다(파일 머리에 명령이 있다).

## 받기

```bash
# hosts.local.json 에 "realsense": "<ds5 tailscale IP>" 를 적고
python server_stream_multi_source.py --on realsense
```

## 프레임 머리

MJPEG 파트마다 `Frame-Meta` 한 줄을 싣는다(DS-03). 받는 쪽이 그것을 `sender` 로 저장하고
비전이 `frame_ref{camera_id, sequence_id, timestamp}` 로 되짚는다.

`camera.hfov_deg 90` 은 **제원값**이고 관측이 아니다(`"source": "datasheet"`).
보정 전에는 거리·좌표의 확정 근거로 쓰지 않는다.
