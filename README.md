# go1_raspi — Go1 온디바이스 라즈베리파이 홈 디렉터리

Go1 로봇 등에 올라가는 라즈베리파이(현재 `pi7`)의 **홈 디렉터리(`/home/physical`) 구성**을 담은 브랜치입니다.
두 번째 Go1용 라즈베리파이를 같은 구성으로 맞추고, 이후 코드가 바뀌었을 때 두 대에 똑같이 반영하려고 만들었습니다.

> **이 저장소의 루트 = `/home/physical`** 입니다.
> systemd 유닛(`hw/pi/deploy/*.service`)이 `/home/physical/hw/pi`, `/home/physical/venv`, `/home/physical/go1sdk`
> 를 절대경로로 쓰기 때문에, 사용자 이름은 **`physical`**, 위치는 **홈 디렉터리 그대로**여야 합니다.
> 경로를 바꾸려면 유닛 파일을 고쳐야 하므로 이름을 맞추는 쪽을 권장합니다.

## 구성

| 경로 | 저장소에 포함 | 얻는 방법 |
|---|---|---|
| `README.md`, `.gitignore`, `requirements.txt` | ✅ | 이 브랜치 |
| `fix_broker_host.sh`, 작업 메모 `*.md` 3개 | ✅ | 이 브랜치 |
| `hw/` (노드 코드 본체·유닛 파일) | ❌ | **HW 브랜치를 클론** (2절) |
| `unitree_legged_sdk/` | ❌ | upstream 클론 (3절) |
| `go1sdk/` (`go1_sdk_pc` 등 실행 파일) | ❌ | `hw/pi/robot/*.cpp` 를 빌드 (4절) |
| `venv/` | ❌ | `requirements.txt` 로 생성 (3절) |
| `captures/`, 백업 폴더, 촬영 샘플 | ❌ | 장치별 데이터 — 공유하지 않음 |

`.gitignore` 는 "전부 무시 + 허용 목록" 방식입니다. 홈 디렉터리에 있는 `.ssh/` 같은 파일이 실수로 올라가지 않게 하려는 것이므로,
새 파일을 올릴 때는 `.gitignore` 의 허용 목록에 `!/<경로>` 를 추가하세요.

---

## 1. 이 브랜치 받기 (새 라즈베리파이)

전제: Raspberry Pi OS(64-bit), 사용자 이름 `physical`.

홈 디렉터리는 이미 비어 있지 않아서 `git clone` 을 바로 쓸 수 없습니다. 홈에서 저장소를 초기화한 뒤 브랜치를 받습니다.

```bash
cd ~
git init
git remote add origin https://github.com/JNY03/physical_demo.git
git fetch origin go1_raspi
git checkout -b go1_raspi --track origin/go1_raspi
```

> 홈 전체가 git 작업 트리가 되므로, 홈 아래 다른 폴더에서 `git status` 를 치면 이 저장소가 잡힙니다.
> `hw/`, `unitree_legged_sdk/` 는 각자 `.git` 을 가진 독립 저장소라 영향이 없습니다.

## 2. HW 브랜치 클론 → `~/hw`

`hw/` 는 [Physical-Project-mk2](https://github.com/khw18033/Physical-Project-mk2) 의 `HW` 브랜치에서 관리하므로 이 브랜치에는 없습니다.
HW 브랜치의 루트(`pi/`, `docs/`, ...)가 그대로 `~/hw` 가 됩니다.

```bash
git clone -b HW https://github.com/khw18033/Physical-Project-mk2.git ~/hw
ls ~/hw/pi/robot/robot_node.py    # 이 파일이 보이면 정상
```

## 3. 패키지 설치

```bash
# 시스템 패키지
sudo apt update
sudo apt install -y git python3-venv python3-pip g++ libboost-dev ffmpeg mosquitto mosquitto-clients chrony

# Unitree SDK (upstream 그대로, 검증 커밋 4539a6c)
git clone -b go1 https://github.com/unitreerobotics/unitree_legged_sdk.git ~/unitree_legged_sdk
git -C ~/unitree_legged_sdk checkout 4539a6c

# 파이썬 venv (pi7 과 같은 버전으로)
python3 -m venv ~/venv
~/venv/bin/pip install --upgrade pip
~/venv/bin/pip install -r ~/requirements.txt
```

`go1-camview`, `go1-front-upload`, `go1-watchdog` 은 시스템 `/usr/bin/python3` 표준 라이브러리 + `ffmpeg` 로 돕니다.

## 4. Go1 SDK 브리지 빌드 → `~/go1sdk`

`go1sdk/` 의 소스는 `hw/pi/robot/` 의 것과 같으므로, 장치마다 HW 브랜치 소스로 빌드합니다.

```bash
mkdir -p ~/go1sdk
SDK=~/unitree_legged_sdk
cd ~/hw/pi/robot
for f in go1_sdk_pc hw_highcmd_daemon hw_highcmd; do
  g++ -O2 -std=c++14 -I $SDK/include $f.cpp \
      -L $SDK/lib/cpp/arm64 -lunitree_legged_sdk -lpthread -o ~/go1sdk/$f
done
ls -l ~/go1sdk
```

코드가 바뀌어 `git pull` 한 뒤에도 이 단계를 다시 실행해야 `go1-sdk` 서비스에 반영됩니다.

## 5. 장치 이름 바꾸기 (두 번째 로봇용)

**1호기와 2호기는 서로 독립된 시스템입니다.** 코드와 기능만 같고, MQTT 브로커는 각자 자기 Pi 안에서 따로 돌립니다.
**2호기는 1호기(`pi7`) 브로커를 참조하지 않습니다.**

브로커가 분리돼 있더라도 Unity·관제·로그에서 두 로봇을 구분할 수 있도록 식별자는 다르게 둡니다.
아래 표의 오른쪽 열은 예시입니다. 팀의 채번 규칙에 맞게 정하세요.

| 항목 | 1호기 (현재) | 2호기 (예시) | 바꾸는 곳 |
|---|---|---|---|
| 호스트명 / 노드 ID | `pi7` | `pi8` | `hostnamectl`, `/etc/node_id` |
| MQTT 브로커 | `pi7` 자체 (`127.0.0.1`) | **`pi8` 자체 (`127.0.0.1`)** | 5-1 절 |
| MQTT 개체 ID (`HW_ENTITY_ID`) | `go1-001` | `go1-002` | `/etc/hw-robot.env` |
| Unity/릴레이 로봇 ID (`--robot_id`) | `go1-1` | `go1-2` | `go1-sdk`, `robot-relay` 유닛 |
| Unity 호스트 PC (`--unity_ip`) | `192.168.50.244` | 2호기 쪽 Unity PC 주소 | `go1-sdk`, `robot-relay` 유닛 |
| Go1 내부망에서 Pi 주소 | `192.168.123.x` | 같아도 됨 (로봇마다 별도 망) | `pi_base_setup.sh --go1` |

> Go1 쪽 주소(`192.168.123.161`, `.13`)는 로봇마다 같습니다. 각 Pi 가 자기 로봇에 유선으로 직결되므로 그대로 둡니다.

### 5-1. 호스트명·기본 셋업

```bash
sudo hostnamectl set-hostname pi8
# /etc/hosts 의 127.0.1.1 줄도 pi8 로 바꿀 것

# 식별자 파일·chrony·venv·Go1 내부망(eth0)·/etc/hw-node.env·robot-node 유닛을 한 번에 구성
cd ~/hw/pi
./pi_base_setup.sh --entity go1-002 --node pi8 --zone zoneA --role robot \
                   --go1 192.168.123.162 --broker 127.0.0.1

# 2호기 자체 MQTT 브로커 (1883: 말단·백엔드, 9001: 관제 웹 WebSocket)
sudo cp ~/hw/pi/deploy/mosquitto-hw.conf /etc/mosquitto/conf.d/hw.conf
sudo systemctl enable --now mosquitto
sudo systemctl restart mosquitto
systemctl is-active mosquitto
```

`--broker 127.0.0.1`: 2호기 노드는 **자기 Pi 안의 mosquitto** 에만 붙습니다.
2호기를 보는 관제 웹·백엔드는 `pi8.local:1883` (웹은 `ws://pi8.local:9001`) 로 접속합니다.

> `/etc/hw-node.env` 의 `HW_BROKER_HOST` 가 `127.0.0.1` 인지 꼭 확인하세요.
> `pi7` 이나 `192.168.50.172` 같은 1호기 주소가 들어 있으면 두 시스템이 섞입니다.
> `fix_broker_host.sh` 는 이 값을 `127.0.0.1` 로 되돌리는 스크립트라서 2호기에서도 그대로 쓸 수 있습니다.

### 5-2. 환경 파일

`/etc/hw-robot.env` 는 저장소에 없습니다(장치 고유값·토큰 포함). 1호기에서 복사한 뒤 ID 만 바꿉니다.

```bash
scp physical@pi7.local:/etc/hw-robot.env /tmp/ && sudo mv /tmp/hw-robot.env /etc/
sudo sed -i 's/^HW_ENTITY_ID=.*/HW_ENTITY_ID=go1-002/' /etc/hw-robot.env
sudo grep -E '^HW_(ENTITY_ID|NODE_ID|BROKER_HOST|DETECT_URL)' /etc/hw-robot.env /etc/hw-node.env   # 1호기 값(pi7, go1-001, 1호기 IP)이 남아 있지 않은지 확인
```

### 5-3. systemd 유닛 설치 (ID 는 설치본에서만 변경)

저장소의 유닛 파일은 1호기 값(`go1-1`, `pi7`)을 담고 있습니다. **저장소 파일은 건드리지 않고**
`/etc/systemd/system/` 에 복사한 설치본에서만 바꿉니다.

```bash
cd ~/hw/pi/deploy
sudo cp robot-node.service robot-relay.service detect-bridge.service go1-sdk.service \
        go1-camview.service go1-front-upload.service go1-watchdog.service go1-watchdog.timer \
        /etc/systemd/system/

# 로봇 ID / 노드 ID 바꾸기
sudo sed -i 's/--robot_id go1-1/--robot_id go1-2/' /etc/systemd/system/go1-sdk.service
sudo sed -i -e 's/--node_id pi7/--node_id pi8/' -e 's/--default_robot_id go1-1/--default_robot_id go1-2/' \
        /etc/systemd/system/robot-relay.service

# 2호기 Unity 호스트 PC 가 1호기와 다르면 주소도 바꾼다
UNITY2=<2호기 Unity PC IP>
sudo sed -i "s/--unity_ip 192.168.50.244/--unity_ip $UNITY2/" \
        /etc/systemd/system/go1-sdk.service /etc/systemd/system/robot-relay.service
```

`detect-bridge` 는 장치 ID 기본값이 코드 인자(`--device go1-001 --node-id pi7`)로 되어 있어 drop-in 으로 덮어씁니다.

```bash
sudo systemctl edit detect-bridge
```
```ini
[Service]
ExecStart=
ExecStart=/home/physical/venv/bin/python3 -u -m robot.detect_bridge --device go1-002 --node-id pi8
```

활성화:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now robot-node robot-relay detect-bridge go1-camview go1-front-upload go1-watchdog.timer
```

> ⚠ **`go1-sdk` 는 `enable` 하지 마세요.** 기동하는 순간 로봇을 기립(force-stand)시킵니다.
> 쓸 때만 `sudo systemctl start go1-sdk` 로 수동 기동합니다.
>
> `sensor-node`(수위센서, `wl-001`)는 1호기 전용입니다. 2호기에 센서가 없으면 설치하지 않습니다.

### 5-4. 선택 사항

- 원격 접속(Tailscale): `sudo tailscale up --hostname=pi8`
- WiFi/핫스팟 프로파일은 비밀번호가 들어 있어 저장소에 없습니다. `nmcli` 로 직접 등록하세요.

## 6. 확인

```bash
systemctl status mosquitto robot-node robot-relay detect-bridge --no-pager
journalctl -u robot-node -n 30 --no-pager
mosquitto_sub -h localhost -t 'zoneA/robot/#' -v -W 10 | cut -c1-120   # 2호기 자체 브로커에 go1-002 만 보이면 정상 (go1-001 이 보이면 안 됨)
ss -tnp | grep ':1883'                                                    # 접속 대상이 127.0.0.1 뿐인지 (1호기 IP 가 없어야 함)
```

브라우저에서 `http://pi8.local:8090/` 으로 2호기 카메라 뷰가 열리면 정상입니다.

---

## 코드 수정이 생겼을 때 (두 대에 반영)

| 바뀐 것 | 수정·커밋할 곳 | 각 Pi 에서 |
|---|---|---|
| `hw/` 의 코드·유닛 | HW 브랜치 | `git -C ~/hw pull` → 서비스 재시작 (C++ 이면 4절 재빌드) |
| 이 브랜치의 파일 | `go1_raspi` 브랜치 | `cd ~ && git pull` |
| 유닛 파일 | HW 브랜치 | 5-3 절 복사·ID 변경을 다시 수행 |

```bash
sudo systemctl restart robot-node robot-relay detect-bridge
```
