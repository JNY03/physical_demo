# vision/runners — 모델 실행기 프로토콜 (M2)

모델 하나 = 이 폴더의 파일 하나(`vision/runners/<이름>.py`) + 레지스트리 항목 하나(`vision/model_registry.yaml` 의 `models.<키>`).
등록하면 능력(`provides`)이 맞는 역할 슬롯마다 선택지로 뜨고(`ds_resources/slots.py`), 고르면 비전이
`vision.runners.load_runner(키, 항목, 장치, cfg)` 로 만든다. 적재기는 고를 때만 모듈을 import 한다.

## 레지스트리 항목

```yaml
models:
  my_model:
    runner: my_model:MyRunner          # vision/runners/my_model.py 의 MyRunner (점이 있으면 절대 모듈 이름)
    kind: detect                       # detect | segment | depth | track | vlm | embed (없으면 provides 로 짐작)
    provides: [detect, open_vocab]     # 능력: detect segment open_vocab ovd depth_metric depth_relative track vlm embed
                                       # 지정 물체 탐지 근거 모델 자리: detect_text(YOLOE 텍스트 가중치) ·
                                       # detect_free(YOLOE -pf) · embed_image_text(CLIP ViT-B/32 ONNX 셋, files_for) —
                                       # 이 셋은 runner 없이도 targeted_detect 의 detector · free_arm · embed 슬롯에 뜬다(component)
    runs_on: [gpu]                     # gpu | cpu_x86 | cpu_arm64 (기기 단계 슬롯은 cpu_arm64 만 본다)
    size: small                        # 기기(파이) 슬롯 중 size: small 인 곳에 뜨려면
    gpu_mb: 800                        # 배치 어림 (MB) — 같은 모델은 프로세스에서 한 번만 센다
    p90_ms: {s: 40, m: 60, l: 90, xl: 150}   # 해상도 구간 s ≤640 · m ≤1280 · l ≤1920 · xl
    license: Apache-2.0                # 안내만 — 모델은 바꿔 끼우는 부품이라 라이선스로 막지 않는다
    files: [models/my_model.pt]        # 있어야 쓸 수 있는 파일 (가중치 유무 판정)
    device: gpu
    packages: [ultralytics]
    source:                            # tools/fetch_models.py 가 받는 곳 (기존 규칙)
    - {path: models/my_model.pt, url: https://…, sha256: …}
```

매개변수는 `vision/vision_config.yaml` 의 `models.<키>` 에 둔다 — 그 사전이 실행기의 `cfg` 로 간다
(없으면 레지스트리 `models.<키>.config`, 그것도 없으면 빈 사전). 경로(`models/…`)는 DS_HOME 기준으로 풀려서 온다.

## 실행기 클래스

```python
class MyRunner:
    name = "my_model"     # 레지스트리 키와 같게
    kind = "detect"       # detect | segment | depth | track | vlm | embed

    def __init__(self, cfg: dict, device: str):
        # device: "cuda:0" | "cpu". 무거운 import(torch · ultralytics …)는 여기서 한다
        ...

    def close(self):      # 선택 — GPU 메모리를 놓는다
        ...
```

### detect · segment

```python
def __call__(self, bgr: np.ndarray) -> tuple[dict, list[dict]]:
```

- `bgr`: H×W×3 uint8 (OpenCV BGR)
- `info`: `{"infer_ms": float, ...}` — 결과 기록에 그대로 붙는다 (JSON 으로 쓸 수 있는 값만)
- `dets`: 검출 하나마다
  - `cls` int · `name` str · `conf` float(0~1) · `xyxy` `[x1, y1, x2, y2]` (원본 px)
  - 선택: `_poly` np.ndarray (N,2) 분할 외곽 — 밑줄로 시작하는 칸은 저장 전에 빠진다(거리 결합에만 쓴다)
  - 선택: `polygon` `[[x, y], …]` (저장할 외곽) · `mask` · `track_id` int
- 구역(ROI)을 쓰면 비전이 잘라서 넘기고 좌표를 되돌린다 — 실행기는 받은 그림만 본다

### depth (UniDepthRunner · MogeAerialRunner 와 같은 약속)

```python
def __call__(self, bgr: np.ndarray, fov_x: float | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
```

- `fov_x`: 가로 화각(도). 주면 그 화각으로, 없으면 모델이 추정
- `info`: `{"infer_ms": float, "checkpoint": str, ...}`
- `depth`: H×W float32 **미터**, 원본 크기로 되돌린 것. 0 또는 비유한값 = 무효 픽셀
- `K`: 3×3 float 내부 행렬 (원본 px 기준 — 줄여 돌렸으면 되돌린 것)
- **상대 깊이(provides: depth_relative)** 실행기도 같은 꼴로 돌려주되 미터가 아니다 — `info["relative"] = True`,
  `info["units"] = "relative"`. depth 는 「클수록 멀다」 · 중앙값 1.0 으로 맞춘 값. 순서 · 근거리 판단에만 쓴다
  (`depth_anything_cpu.py`)

### CPU 실행기 (기기 · 서버 CPU 여유분, 2026-11-27)

`runs_on: [cpu_arm64, cpu_x86]` · `size: small` 인 실행기는 기기(파이) 컨테이너에 **파일 하나만 복사해도** 돌게
numpy · cv2 · onnxruntime(· ncnn)만 쓰고 저장소 모듈을 import 하지 않는다. 상대 경로(`models/…`)는 스스로
`DS_HOME`(없으면 저장소 루트) 기준으로 푼다. 매개변수 `threads` (기본 4 — 파이 5 코어 수).

| 파일 | 레지스트리 키 | 능력 |
|---|---|---|
| `yolo_seg_cpu.py` | `yolo26n_seg_cpu` · `yoloe26n_vocab_cpu` | detect · segment (· open_vocab — 어휘를 굳힌 ONNX) |
| `depth_anything_cpu.py` | `da3_small_cpu` · `dav2_small_cpu` | depth_relative |
| `clip_embed_cpu.py` | `mobileclip_s0_cpu` | embed |

ONNX 는 `tools/export_cpu_models.py` 로 굳힌다(서버에서 — ultralytics 가 있는 venv).

### track · vlm · embed

지금 비전 루프는 부르지 않는다 — 고른 것은 오케스트레이터 계획(plan.slots)에만 실린다. 약속만 정해 둔다.

- track: `update(dets: list[dict], ts: float, bgr: np.ndarray | None = None) -> list[dict]` — `track_id` 를 붙여 돌려준다
- vlm: `__call__(bgr, prompt: str) -> (info, text: str)`
- embed: `__call__(bgr, boxes: list[list[float]]) -> (info, np.ndarray (N, D), L2 정규화)`

## 확인

```bash
venv/bin/python -m ds_resources.slots              # 기능마다 슬롯과 선택지 — 새 모델이 떴는지
venv/bin/python vision/vision_infer.py --models my_model --runs latest --limit 3   # 실제로 돌려 보기
```
