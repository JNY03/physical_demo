# 말단 노드 컨테이너 — robot / actuator / sensor 를 **한 이미지**로 낸다
# =====================================================================
# 노드 종류는 이미지가 아니라 env 로 갈린다(HW_ENTITY_TYPE). 기종도 마찬가지다
# (HW_CONTROLLER_LINK = sim | go1 | ep | can | eth). 이미지를 종류마다 만들면
# 빌드가 N배가 되고, 같은 공통 코어가 N번 복사돼 버전이 어긋난다.
#
#   docker buildx build --platform linux/arm64 -t <registry>/physical-node:<태그> \
#          -f pi/Dockerfile pi
#
# 말단은 보통 aarch64 다. **반드시 대상 아키텍처로 빌드한다** — amd64 이미지를
# 밀어 넣으면 ImagePullBackOff 가 아니라 exec format error 로 죽어서 원인을
# 찾는 데 시간이 걸린다.
#
# 배포는 태그가 아니라 **digest 로 고정**한다. 태그는 같은 이름으로 내용이 바뀔
# 수 있어 이벤트 시점에 예상과 다른 것이 뜬다(SDD §7.3).
FROM python:3.13-slim-bookworm

# 말단에 서버형 미들웨어를 두지 않는다(AI-B-10). 전송 클라이언트와 규약 직렬화만.
RUN pip install --no-cache-dir paho-mqtt==2.1.0 protobuf

WORKDIR /app
COPY common/ /app/common/
COPY robot/ /app/robot/
COPY actuator/ /app/actuator/
COPY sensor/ /app/sensor/

# 기종 SDK 는 **이미지에 넣지 않는다.** Unitree SDK 와 RoboMaster SDK 는 각각
# 라이선스와 빌드 전제가 다르고, 둘을 넣으면 쓰지 않는 쪽까지 끌고 다닌다.
# 필요한 기종은 호스트에서 빌드해 볼륨으로 마운트한다(docs/go1-raspi.md 4절).
# 기본값 CONTROLLER_LINK=sim 은 SDK 없이 그대로 뜬다 — 배포 경로를 먼저
# 검증하고 기종을 나중에 붙일 수 있다.

RUN useradd -r -u 10001 node && mkdir -p /var/lib/hw-node && chown node /var/lib/hw-node
USER node

ENV PYTHONUNBUFFERED=1 \
    HW_CONTROLLER_LINK=sim \
    HW_SPOOL_PATH=/var/lib/hw-node/spool.jsonl

# 어느 노드를 띄울지는 실행 시점에 정한다.
#   docker run -e HW_ENTITY_TYPE=robot ... <image> python3 -m robot.robot_node
# k3s 에서는 매니페스트의 command 가 이 자리를 채운다.
ENTRYPOINT ["python3"]
CMD ["-m", "robot.robot_node"]
