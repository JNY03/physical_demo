# 위험물 프로토타입 (`prototypes.npz`, 39 KB)

`hazard_verify.py` 의 `match` 단서가 읽는 **유일한 런타임 파일**이다. 없으면 match 만
빠지고 나머지 단서(class·size·detail·color·structure)는 그대로 선다.

## 무엇이 들어 있나

`{인코더}/{클래스}` → (K, 512) 정규화 임베딩. 사진이 아니라 **임베딩만** 들어 있다.

| 묶음 | 장수 | 무엇 |
|---|---:|---|
| `clip_b32/wooden_box_mine` | 2 | 현장 사진의 나무 토막 |
| `clip_b32/butane_canister` | 12 | 현장 2 + 제품컷 10 (썬·맥스 두 브랜드) |
| `clip_b32/_negative` | 5 | **음료캔 1** + 현장 배경 4 (콘크리트·이끼) |

음성 묶음이 핵심이다. 점수가 `pos − neg` 라 음성이 없으면 "무엇과도 비슷한" 흐릿한
crop 이 높게 나온다.

## 다시 만들려면

원본 crop 은 **이 저장소에 없다**(33 MB · 개발 기계의 `local/datasets/butane_ref/`).
폴더 구조는 `<클래스>/*.png` 이고 `_negative/` 는 헷갈리는 물건이다.

```python
from hazard_verify import build_bank
build_bank("<crop 폴더>", "assets/butane_ref/prototypes.npz", [{
    "name": "clip_b32",
    "path": "<.../clip-vit-base-patch32/onnx/vision_model_quantized.onnx>",
    "size": 224,
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std":  [0.26862954, 0.26130258, 0.27577711]}])
```

**운영 영상은 넣지 않는다** — 그것은 시험 자료다. 기준 사진에서만 만든다.

## 한계

나무 상자 프로토타입이 **2 개**뿐이고 같은 현장의 같은 상자다. 실내에서 절반이
통과한 것은 "그 상자와 닮았다" 는 뜻이지 "나무 상자를 잘 가린다" 가 아니다.
현장 사진이 늘면 다시 만들어 갈아 끼운다 — 코드는 안 바뀐다.
