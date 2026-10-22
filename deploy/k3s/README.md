# k3s 배포

말단(라즈베리파이 등)과 엣지에 제어 코드와 환경을 한 번에 올린다.

**이 디렉터리에는 특정 기기의 주소가 한 줄도 없다.** 어느 기기가 무엇인지는
쓰는 사람이 **노드 라벨**로 선언하고, 주소는 `10-config.yaml` 한 곳에만 적는다.

## 1. 라벨로 역할을 정한다

```bash
kubectl label node <말단-노드>  physical.io/role=terminal
kubectl label node <엣지-노드>  physical.io/role=edge
```

기기를 바꾸면 라벨만 옮긴다. 매니페스트는 손대지 않는다.

## 2. 설정을 자기 환경에 맞게 고친다

`10-config.yaml` 하나만 본다.

| 키 | 뜻 |
|---|---|
| `HW_BROKER_HOST` | MQTT 브로커. 클러스터에 띄우면 기본값 그대로 |
| `HW_ZONE_ID` | 구역 식별자 |
| `physical-images.*` | 레지스트리 경로. **digest 고정을 권한다** |

`HW_ENTITY_ID`는 적지 않는다 — 매니페스트가 노드 이름에서 끌어다 쓴다. 같은
id로 둘이 붙으면 client_id 플래핑으로 서로를 끊는다.

## 3. 이미지를 만든다

대상 아키텍처로 빌드한다. 말단은 보통 `arm64`다 — `amd64`를 밀어 넣으면
`ImagePullBackOff`가 아니라 `exec format error`로 죽어 원인을 찾기 어렵다.

```bash
# 말단 노드 (robot/sensor/actuator 공용 — 종류는 env 로 갈린다)
docker buildx build --platform linux/arm64 \
  -t <registry>/physical-node:<태그> -f hw/pi/Dockerfile hw/pi

# 드론 엣지
docker buildx build -t <registry>/drone-edge:<태그> edge
```

## 4. 적용한다

```bash
kubectl apply -f 10-config.yaml
kubectl apply -f 20-broker.yaml      # 브로커가 이미 있으면 건너뛴다
kubectl apply -f 30-terminal-nodes.yaml
kubectl apply -f 40-edge.yaml

kubectl -n physical get pods -o wide
```

엣지 설정은 ConfigMap으로 따로 준다 — 말단 주소가 여기 들어가므로 이미지를
다시 굽지 않고 기기를 바꿀 수 있어야 한다.

```bash
kubectl -n physical create configmap edge-config --from-file=config.json=edge/config.json
```

## 5. 모델 없이 먼저 띄워 본다

`edge/config.json`의 `ovd.provider`를 `color`로 두면 **모델도 GPU도 없이** 전체
파이프라인이 돈다(1280×720에서 CPU 7.7ms 실측). 배포 경로를 먼저 검증하고
모델은 나중에 붙이는 편이 원인을 가르기 쉽다.

말단 노드도 같다. `HW_CONTROLLER_LINK` 기본값이 `sim`이라 기종 SDK 없이 뜬다.
구동부를 붙일 때 `go1`·`ep` 등으로 바꾼다.

## 알아 둘 것

**`dnsPolicy: Default`를 쓴다.** 말단은 클러스터 DNS를 거치지 않고 브로커를
직접 잡는다. 제어평면과 말단이 다른 네트워크에 있으면 파드 간 오버레이(flannel
VXLAN)가 통하지 않아 클러스터 DNS가 죽는 경우가 있다. 브로커를 서비스 이름으로
쓰려면 그 줄을 지우고 `ClusterFirst`로 둔다.

**말단은 DaemonSet이다.** 그 기기에서 돌아야 의미가 있다 — Deployment로 두면
스케줄러가 다른 기기에 올려 구동부에 닿지 못한다.

**기종 SDK는 이미지에 없다.** Unitree와 RoboMaster SDK는 라이선스와 빌드 전제가
달라, 호스트에서 빌드해 `/opt/sdk`로 마운트한다.

**프로브는 `/api/live`를 쓴다.** `/api/health`에는 저장소 사용량 집계가 섞여
있어 파일이 쌓이면 느려진다 — 2026-10-22에 records 19,989개에서 8.9초가 걸려
프로브가 통째로 실패했다. 지금은 캐시를 넣어 고쳤지만, 프로브는 가장 싼 것을
쓰는 편이 맞다.

## 검증 상태

매니페스트는 **스키마 수준까지만 확인했다**(리소스 10개, 필수 키·이미지 지정).
실제 클러스터에 적용해 보지는 못했다 — 검증 환경에 k3s가 없었다. 적용 전에
`kubectl apply --dry-run=server -f .`로 한 번 더 확인하기를 권한다.
