"""엣지 모델 계층 — OVD 두 종, CLIP, 그리고 metric depth.

implements: AI-B-08, AI-B-09, AI-C-05, AI-E-04, AI-S-03

**모델을 여기 모은 이유**: 전부 "무거운 것을 올리고 배열을 돌려준다"는 같은 계약을
가진다. 어느 것이 없어도 나머지는 돈다(AI-C-05) — 없는 단계는 실패가 아니라 축소다.

각 클래스는 `available()`을 갖고, 로드에 실패하면 조용히 False가 된다. 예외를
띄우지 않는 이유는 실측 경험이다: 2026-09-21에 transformers 버전 차이로 stage2가
매 프레임 죽었는데 비동기 워커 안이라 stage1은 멀쩡해 보였고, 며칠을 그대로 돌았다.
그래서 지금은 **가용 여부를 상태로 들고** 헬스에 드러낸다(AI-O-02).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def resolve(cfg: dict, rel: str) -> Path:
    """설정 안의 상대경로를 config.json 위치 기준으로 푼다."""
    p = Path(rel)
    return p if p.is_absolute() else (Path(cfg["_root"]) / p).resolve()



def embed_tensor(out):
    """transformers `get_*_features` 반환에서 임베딩 텐서를 꺼낸다.

    4.x는 텐서를, 5.x는 `ModelOutput`을 돌려준다. 버전 차이를 여기 한 곳에서
    흡수한다 — 2026-09-21에 이 차이로 stage2가 매 프레임 실패했는데, 예외가
    비동기 워커 안에서만 나서 stage1은 멀쩡해 보였다. `pooler_output`은
    projection까지 적용된 512차원이라 그대로 쓸 수 있다.

    클래스 메서드가 아니라 모듈 함수인 이유: 같은 이름의 메서드를 두 scorer
    클래스에 각각 두면 한쪽만 고치는 실수가 나기 쉽다(실제로 한 번 냈다).
    """
    import torch

    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"임베딩을 못 찾았다: {type(out).__name__}")



# --------------------------------------------------------------------------
# stage1 — OVD. YOLOE prompt-free 한 종만 쓴다.
# --------------------------------------------------------------------------
class YoloeOvd:
    """YOLOE prompt-free — 텍스트 프롬프트 없이 **내장 어휘 4585종**으로 검출한다.

    2026-09-21 확인(`yoloe-11s-seg-pf.pt`, ultralytics 8.4.146): `-pf` 가중치는
    prompt-free(PE-Free)라 `set_classes` 없이 그대로 predict된다. 어휘가 넓을수록
    '흔한 물체'가 OVD 쪽으로 흡수되어 미지 집합에 진짜 미지만 남는다.
    실측 비교(같은 10장): YOLOE 17.1 ms / 4585종 vs YOLO-World 12.0 ms / 87종.
    5 ms 손해를 어휘 53배와 바꾼 것이고, 어휘 설계를 사람이 하지 않아도 된다.
    """

    def __init__(self, cfg: dict):
        c = cfg["ovd"]
        self._cfg = cfg
        self.conf = float(c.get("conf", 0.10))
        self.iou = float(c.get("iou", 0.5))
        self.imgsz = int(c.get("imgsz", 640))
        self.mode = "prompt_free"
        self.model = None
        self._names: dict = {}
        self.vocab: list[str] = []
        # 시작부터 도메인 어휘로 연다 — prompt-free로 한 번 열었다 다시 여는 것은
        # 가중치를 두 번 읽는 낭비다.
        # **이 모델은 언제나 prompt-free다.** 도메인 어휘는 두 번째 모델(WorldOvd)이
        # 맡고, 확정은 둘의 합의로 정한다(consensus.py). 한쪽이라도 도메인에
        # 묶이면 "어휘 설계 없이 새 현장에 투입한다"는 요구가 깨진다(AI-C-15).
        self.set_vocabulary(None)

    def set_vocabulary(self, words) -> None:
        """어휘를 갈아 끼운다. None이면 prompt-free, 목록이면 텍스트 프롬프트.

        **왜 필요한가**(2026-09-21 실측): prompt-free 4585종은 실내 장면에서 모니터를
        `computer chair`, 벽면을 `press room`, 빈 곳을 `speaker`/`hassock`으로 불렀다.
        어휘가 넓으면 미지 집합이 깨끗해지는 대신, 화면에 올라오는 이름 자체를 믿을 수
        없게 된다 — 4585개 중 하나를 고르는 문제라 아무것이나 그럴듯하게 맞는다.

        텍스트 프롬프트 모드는 후보를 사람이 정한 몇십 개로 좁힌다. 어느 쪽이 나은지는
        배치마다 다르므로 **코드가 아니라 설정에서** 고른다(AI-C-15: 도메인 차이는
        분기문이 아니라 배포 프로파일). 어휘가 바뀌면 재측정 대상이다(AI-B-01).
        """
        from ultralytics import YOLOE  # 지연 import (AI-C-11)

        c = self._cfg["ovd"]
        if words:
            path = resolve(self._cfg, c.get("text_weights", c["weights"]))
            self.model = YOLOE(str(path))
            self.model.set_classes(list(words), self.model.get_text_pe(list(words)))
            self._names = {i: w for i, w in enumerate(words)}
            self.vocab = list(words)
            self.mode = "text_prompt"
        else:
            self.model = YOLOE(str(resolve(self._cfg, c["weights"])))
            names = getattr(self.model.model, "names", {}) or {}
            self._names = names
            self.vocab = [names[i] for i in sorted(names)] if names else []
            self.mode = "prompt_free"

    def detect(self, bgr) -> list[dict]:
        res = self.model.predict(bgr, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append({"box": [x1, y1, x2, y2],
                        "label": self._names.get(int(b.cls[0]), str(int(b.cls[0]))),
                        "conf": float(b.conf[0])})
        return out


class WorldOvd:
    """YOLO-World 텍스트 프롬프트 — 현장 어휘로 좁혀 보는 두 번째 눈.

    **동기 경로에 둘 수 있다**(2026-09-21 실측, 같은 프레임·RTX 3060):
        YOLOE prompt-free 4585종  15 ms
        YOLO-World 텍스트 20종    16 ms
        Grounding DINO tiny       453 ms (autocast fp16 299 ms)
    GDINO는 라벨 품질과 무관하게 이 예산에 들어오지 못한다 — 말단 왕복이 이미
    ~0.5 s인데 여기에 0.3~0.45 s를 더하면 시점 반영의 의미가 없어진다.

    어휘는 설정이다(`ovd2.vocabularies[<domain>]`). 코드에 클래스 이름을 박으면
    도메인이 늘 때마다 핵심 코드가 바뀐다(AI-C-15).
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        c = cfg.get("ovd2", {})
        self.conf = float(c.get("conf", 0.05))
        self.iou = float(c.get("iou", 0.5))
        self.imgsz = int(c.get("imgsz", 640))
        self.model = None
        self.vocab: list[str] = []
        self.set_vocabulary(
            (c.get("vocabularies") or {}).get(cfg["clip"].get("active_domain", "")))

    def set_vocabulary(self, words) -> None:
        from ultralytics import YOLOWorld  # 지연 import (AI-C-04)

        words = list(words or [])
        self.vocab = words
        if not words:
            self.model = None      # 어휘가 없으면 이 눈은 쉰다(선택 기능, AI-C-05)
            return
        self.model = YOLOWorld(str(resolve(self._cfg, self._cfg["ovd2"]["weights"])))
        self.model.set_classes(words)

    def detect(self, bgr) -> list[dict]:
        if self.model is None:
            return []
        res = self.model.predict(bgr, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None:
            return []
        names = res.names
        return [{"box": [float(v) for v in b.xyxy[0].tolist()],
                 "label": names[int(b.cls[0])], "conf": float(b.conf[0])}
                for b in res.boxes]


# --------------------------------------------------------------------------
# CLIP — quantized ONNX 두 타워. 텍스트 임베딩은 프롬프트가 바뀔 때만 다시 만든다.
# --------------------------------------------------------------------------
class ClipScorer:
    def __init__(self, cfg: dict):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        c = cfg["clip"]
        self.size = int(c.get("input_size", 224))
        # 실행 provider를 코드에 고정하지 않는다(AI-B-08). 설정이 요구한 것 중 이 호스트가
        # 실제로 가진 것만 쓰고, 없으면 CPU로 내려간다 — 가속기 부재는 실패가 아니다(AI-C-05).
        want = list(c.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
        have = set(ort.get_available_providers())
        self.providers = [p for p in want if p in have] or ["CPUExecutionProvider"]
        self.vision = ort.InferenceSession(str(resolve(cfg, c["vision_onnx"])),
                                           providers=self.providers)
        self.text = ort.InferenceSession(str(resolve(cfg, c["text_onnx"])),
                                         providers=self.providers)
        self.tok = AutoTokenizer.from_pretrained(str(resolve(cfg, c["tokenizer_dir"])))
        self.prompts: list[str] = []
        self.text_emb: np.ndarray | None = None
        self._vis_in = self.vision.get_inputs()[0].name
        # CLIP 전처리 상수(OpenAI) — ImageNet 상수와 다르다. 섞으면 점수가 망가진다.
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    @staticmethod
    def _l2(a: np.ndarray) -> np.ndarray:
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)

    def set_prompts(self, prompts: list[str]) -> None:
        if prompts == self.prompts and self.text_emb is not None:
            return
        enc = self.tok(prompts, padding="max_length", max_length=77,
                       truncation=True, return_tensors="np")
        feed = {i.name: enc[i.name].astype(np.int64)
                for i in self.text.get_inputs() if i.name in enc}
        self.text_emb = self._l2(np.asarray(self.text.run(None, feed)[0], np.float32))
        self.prompts = list(prompts)

    def score(self, crops: list[np.ndarray], batch_size: int = 16) -> list[list[float]]:
        """crop들을 배치로 인코딩한다.

        **정정(2026-09-21)**: 배치를 지연 대책으로 넣었으나 효과가 없었다(900→950 ms).
        이전 판이 이미 전체를 한 번에 넣고 있었으므로 쪼갠 것이 오히려 호출을 늘렸다.
        900 ms는 quantized ViT-B/32로 crop 60개를 도는 고유 비용이고, 실제 대책은
        실행 provider(CUDA)다. batch_size는 메모리 상한 용도로만 남긴다.
        """
        if not crops or self.text_emb is None:
            return []
        prep = []
        for c in crops:
            r = cv2.resize(c, (self.size, self.size))
            x = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            prep.append(((x - self.mean) / self.std).transpose(2, 0, 1))
        embs = []
        for i in range(0, len(prep), max(1, batch_size)):
            chunk = np.ascontiguousarray(np.stack(prep[i:i + batch_size]))
            embs.append(np.asarray(self.vision.run(None, {self._vis_in: chunk})[0],
                                   np.float32))
        return (self._l2(np.concatenate(embs)) @ self.text_emb.T).tolist()


class TorchClipScorer:
    """torch CUDA fp16 CLIP. 2026-09-21 벤치의 승자다(`bench_clip.py`).

    crop 60장 기준 실측:

        int8 ONNX  CPU   823.4 ms   13.72 ms/crop   (이전 기본값)
        int8 ONNX  CUDA  969.2 ms   16.15 ms/crop   ← GPU가 **더 느리다**
        fp32 ONNX  CUDA   80.6 ms    1.34 ms/crop
        torch      fp16   34.6 ms    0.58 ms/crop   ← 기준선 대비 **24배**
        MobileCLIP2-S0 fp16  96.3 ms  1.61 ms/crop
        MobileCLIP-S1  fp16 165.6 ms  2.76 ms/crop

    두 가지가 드러났다. (1) **quantized 그래프는 GPU로 갈 수 없다** — CUDA EP에 해당
    연산이 없어 memcpy 노드가 끼고 CPU↔GPU를 오간다. 그래서 양자화는 CPU 전용 최적화다.
    (2) **작은 모델이 GPU에서 빠른 것이 아니다** — MobileCLIP2-S0는 파라미터가 7.7배
    적은데 2.8배 느리다. 지연을 정하는 것은 파라미터 수가 아니라 연산 깊이와 커널이다.

    **두 타워를 같은 체크포인트에서 쓴다.** 이미지 타워만 바꾸고 텍스트 임베딩을 예전
    int8 것으로 두면 두 벡터가 서로 다른 공간에 있어 유사도가 무의미해진다.

    **주의(AI-B-01)**: fp16 점수는 int8 점수와 척도가 다르다(같은 crop에서 코사인
    0.879). `background_margin`·`confident_margin`은 int8 분포에서 잡은 값이므로
    백엔드를 바꾸면 재확인 대상이다.
    """

    def __init__(self, cfg: dict):
        import torch
        from transformers import CLIPModel, CLIPTokenizerFast

        c = cfg["clip"]
        name = c.get("torch_model", "openai/clip-vit-base-patch32")
        want = c.get("device", "cuda")
        self.device = want if (want != "cuda" or torch.cuda.is_available()) else "cpu"
        # 가속기 부재는 실패가 아니다 — CPU로 내려가고 그 사실이 로그에 남는다(AI-C-05).
        self.half = bool(c.get("fp16", True)) and self.device == "cuda"
        self.torch = torch
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        if self.half:
            self.model = self.model.half()
        self.tok = CLIPTokenizerFast.from_pretrained(name)
        self.size = int(c.get("input_size", 224))
        self.prompts: list[str] = []
        self.text_emb = None
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
        self.providers = [f"torch/{self.device}{'/fp16' if self.half else '/fp32'}"]

    def set_prompts(self, prompts: list[str]) -> None:
        if prompts == self.prompts and self.text_emb is not None:
            return
        enc = self.tok(prompts, padding=True, truncation=True, max_length=77,
                       return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            t = embed_tensor(self.model.get_text_features(**enc)).float()
        self.text_emb = t / (t.norm(dim=-1, keepdim=True) + 1e-9)
        self.prompts = list(prompts)

    def score(self, crops: list[np.ndarray], batch_size: int = 64) -> list[list[float]]:
        if not crops or self.text_emb is None:
            return []
        prep = []
        for c in crops:
            r = cv2.resize(c, (self.size, self.size))
            x = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            prep.append(((x - self.mean) / self.std).transpose(2, 0, 1))
        embs = []
        for i in range(0, len(prep), max(1, batch_size)):
            x = self.torch.from_numpy(np.ascontiguousarray(
                np.stack(prep[i:i + batch_size]))).to(self.device)
            if self.half:
                x = x.half()
            with self.torch.inference_mode():
                f = embed_tensor(
                    self.model.get_image_features(pixel_values=x)).float()
            embs.append(f / (f.norm(dim=-1, keepdim=True) + 1e-9))
        return (self.torch.cat(embs) @ self.text_emb.T).cpu().numpy().tolist()


LIVE: dict[int, dict] = {}      # camera_id → {"jpeg": bytes, "at": float, "info": dict}
LIVE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Depth — MoGe2-Aerial. 미터 단위 metric depth.
# ─────────────────────────────────────────────────────────────────────────────
class AerialDepth:
    """MoGe-2 + AerialMetric LoRA. 프레임 한 장 → 미터 단위 depth 맵.

    ── 왜 이 모델인가 ─────────────────────────────────────────────────────────
    지상 데이터로 학습한 metric depth는 **항공 영상에서 무너진다.** AerialMetric
    (ECCV 2026)이 실제 UAV 영상(고도 80~120m AGL, pitch -90~-45도)에서 잰 값:

        모델              AbsRel↓   δ₁↑
        ZoeDepth-NK        97.1     0.0%
        DepthPro           97.8     0.0%     ← 지상 SOTA가 완전히 실패한다
        MoGe-2-L           48.4     5.1%
        UniDepthV2-L       31.0    34.1%     ← zero-shot 최강인데도 이 수준
        MoGe2-Aerial       10.3    89.3%     ← LoRA 미세조정 후

    라이선스도 같이 봤다: UniDepthV2는 CC BY-NC(상업 불가), Metric3Dv2는 BSD-2
    비상업이다. MoGe-2/MoGe2-Aerial만 MIT라 기기를 옮기고 배포해도 걸리지 않는다.

    ── intrinsics를 반드시 넘긴다 ─────────────────────────────────────────────
    위 표는 **GT intrinsics 없이** 잰 값이다. 말단이 `meta["camera"]["hfov_deg"]`를
    실어 보내므로 여기서 `given_fov_x`로 넘긴다. 아는 값을 안 넘겨서 모델이 추정하게
    두면 그만큼 잃는다.

    ── 거리는 근거가 아니라 관측이다 ───────────────────────────────────────────
    여기서 내는 미터 값은 **관측 시점의** 거리다. 말단까지 왕복하는 동안 기체가
    움직이므로 도착하면 이미 낡아 있고, track_id로 붙여도 고쳐지지 않는다
    (drone_rpi/records.py `Range` 참고). 그래서 응답에 `observed_at`을 같이 싣고
    말단이 `ego_compensated=False`로 표시한다.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("depth", {})
        self.ok = False
        self.error: str | None = None
        self.model = None
        self.torch = None
        self.infer = None
        if not self.cfg.get("enabled", False):
            self.error = "disabled"
            return
        try:
            import torch
            import sys
            self.torch = torch
            # AerialMetric이 포크한 MoGe를 쓴다 — microsoft/MoGe 본가가 아니다.
            # LoRA 체크포인트의 키가 이 포크 기준이라 본가로 로드하면 조용히
            # 어긋난 가중치가 붙는다. fetch_models.sh가 두 경로를 맞춰 놓는다.
            repo = Path(self.cfg.get("aerialmetric_repo", "third_party/AerialMetric"))
            if repo.exists():
                sys.path.insert(0, str(repo / "MoGe"))
                sys.path.insert(0, str(repo))
            self._load(repo)
            self.ok = True
        except Exception as exc:                  # 없으면 이 단계만 쉰다(AI-C-05)
            self.error = repr(exc)[:300]

    def _load(self, repo: Path) -> None:
        import json as _json

        from peft import LoraConfig, get_peft_model

        from moge.model import import_model_class_by_version

        torch = self.torch
        ckpt_path = Path(self.cfg["checkpoint"])
        lora_cfg_path = Path(self.cfg.get("lora_config", repo / "MoGe/configs/train/v2.json"))
        rank = int(self.cfg.get("lora_rank", 96))

        train_cfg = _json.loads(Path(lora_cfg_path).read_text(encoding="utf-8"))
        ModelCls = import_model_class_by_version(train_cfg.get("model_version", "v2"))
        model = ModelCls(**train_cfg["model"])
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=2 * rank, bias="none",
            target_modules=["qkv", "proj", "fc1", "fc2"],
            modules_to_save=["scale_head"]))

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        keys = set(model.state_dict().keys())
        remapped = {}
        for k, v in state.items():
            if k in keys:
                remapped[k] = v
                continue
            for prefix in ("base_model.model.", "model.", ""):
                pk = f"{prefix}{k}" if prefix else k
                if pk in keys:
                    remapped[pk] = v
                    break
        missing = model.load_state_dict(remapped, strict=False)
        # 몇 개가 안 붙었는지 드러낸다 — 조용히 절반만 로드되는 것이 최악이다.
        self.unmatched = len(getattr(missing, "missing_keys", []))

        self.device = self.cfg.get("device", "cuda")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"               # GPU 없으면 내려간다(AI-C-05)
        self.fp16 = bool(self.cfg.get("fp16", True)) and self.device.startswith("cuda")
        model.to(self.device).eval()
        if self.fp16:
            model = model.half()
        self.model = model

    def available(self) -> bool:
        return self.ok and self.model is not None

    @staticmethod
    def _fit14(w: int, h: int, longest: int | None):
        """DINOv2 백본이라 변 길이가 14의 배수여야 한다."""
        if longest and longest > 0:
            s = longest / max(h, w)
            w, h = int(w * s), int(h * s)
        return max(14, (w // 14) * 14), max(14, (h // 14) * 14)

    def depth_map(self, bgr, hfov_deg: float | None = None):
        """(H, W) float32 미터 depth. 실패하면 None — 이 단계만 빠진다."""
        if not self.available():
            return None
        torch = self.torch
        h0, w0 = bgr.shape[:2]
        w, h = self._fit14(w0, h0, self.cfg.get("longest_side", 700))
        img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (w0, h0) else bgr
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().div_(255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.fp16:
            x = x.half()
        kw = {"resolution_level": int(self.cfg.get("resolution_level", 9)),
              "force_projection": True, "apply_mask": True,
              # MoGe는 fp16을 인자로 받는다 — 모델을 .half()로 바꾼 것만으로는
              # 내부 경로가 따라오지 않는다(demo_infer.py도 둘 다 넘긴다).
              "use_fp16": self.fp16}
        fov = hfov_deg if hfov_deg is not None else self.cfg.get("fallback_hfov_deg")
        if fov:
            # **아는 값을 넘긴다.** AerialMetric 표는 intrinsics 없이 잰 것이고,
            # 안 넘기면 모델이 FoV를 추정하면서 그만큼 잃는다.
            kw["fov_x"] = float(fov)
        try:
            with torch.inference_mode():
                out = self.model.infer(x, **kw)
        except Exception as exc:
            self.error = repr(exc)[:300]
            return None
        # squeeze로 배치·채널 차원을 떨어뜨린다. 모델 판본마다 (1,H,W)나 (H,W)로
        # 갈려서 ndim 분기로 맞추면 한쪽에서 조용히 틀린다.
        d = out["depth"].float().squeeze().cpu().numpy()
        if d.shape[:2] != (h, w):
            d = d[:h, :w]                     # 14 정렬 패딩 제거
        if d.shape[:2] != (h0, w0):
            # 선형 보간 — NEAREST는 경계에서 거리가 계단처럼 튀고, 작은 객체의
            # 박스 안 백분위수가 그 계단에 걸린다.
            d = cv2.resize(d, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return d

    @staticmethod
    def range_for_box(depth, box, *, percentile: float = 25.0):
        """박스 안 depth의 대표값(미터).

        **중앙값이 아니라 하위 백분위수를 쓴다.** 박스에는 대상뿐 아니라 그 뒤
        배경이 함께 들어오고, 작은 객체일수록 배경 비율이 크다. 중앙값을 쓰면
        먼 배경으로 끌려간다 — 위험 판단에서는 **가까운 쪽으로 틀리는 편**이
        안전하므로 앞쪽 픽셀을 대표로 삼는다.
        """
        if depth is None:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        h, w = depth.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = depth[y1:y2, x1:x2]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size == 0:
            return None
        return float(np.percentile(patch, percentile))


class ZipDepth:
    """ZipDepth ONNX — **상대 역depth**. metric이 아니다.

    MoGe2-Aerial 체크포인트(1.48GB)를 받는 동안 파이프라인을 끝까지 돌려 보려고
    붙인 임시 provider다. torch가 필요 없고 CPU에서 256px 12ms / 384px 30ms라
    당장 쓸 수 있다.

    ── 미터를 내지 않는 이유 ──────────────────────────────────────────────────
    출력이 `affine_invariant_inverse`라 cm/m로 바꾸려면 보정 프로파일이 있어야
    한다. 저장소에 하나 있지만(`zipdepth_base_256.crop_bc.inv_affine_direct_cm.v1`)
    그것은 **464x400 실내 어안 로봇** 영상에 crop (114,80,351,321)을 적용한
    구성에서 맞춘 것이고 유효 범위가 센티미터다. 지금은 imx708 wide 1280x720에
    crop 없이, 대상 거리가 수십 미터다 — 조건이 하나도 겹치지 않는다.
    프로파일을 구성 간에 옮기지 않는 것이 AI-B-01이고, 그쪽 보고서도
    `status: provisional`에 계수가 프레임마다 CV 46.5%로 흔들린다고 적어 두었다.

    그래서 이 provider는 **깊이 맵만** 낸다. 화면의 depth 패널이 그것을 그리고,
    거리(m)는 클래스 크기 사전이 따로 낸다 — 그쪽은 focal_px와 박스 높이만 쓰므로
    보정 프로파일이 필요 없다. MoGe2-Aerial이 준비되면 provider만 바꾸면 된다.
    """

    metric = False        # **미터가 아니다.** 소비자가 이 플래그를 보고 갈라야 한다.

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("depth", {})
        self.ok = False
        self.error: str | None = None
        self.sess = None
        self.unmatched = 0
        try:
            path = resolve(cfg, self.cfg.get("zipdepth_model",
                                             "models/zipdepth_base_384x384.onnx"))
            o = ort.SessionOptions()
            o.log_severity_level = 3
            # CUDA EP는 이 환경에서 libcublasLt를 못 찾아 조용히 CPU로 내려간다.
            # 어차피 CPU 30ms라 굳이 매달리지 않는다.
            self.sess = ort.InferenceSession(str(path),
                                             providers=["CPUExecutionProvider"])
            i = self.sess.get_inputs()[0]
            self.iname = i.name
            self.H, self.W = int(i.shape[2]), int(i.shape[3])
            self.device, self.fp16 = "cpu", False
            self.ok = True
        except Exception as exc:
            self.error = repr(exc)[:300]

    def available(self) -> bool:
        return self.ok and self.sess is not None

    def depth_map(self, bgr, hfov_deg: float | None = None):
        """(H, W) float32 **상대 역depth**. 값이 클수록 가깝다 — 미터가 아니다."""
        if not self.available():
            return None
        h0, w0 = bgr.shape[:2]
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.W, self.H))
        # 파일명에 fp16이 붙어 있어도 **입력은 fp32**다(가중치만 fp16). 실제로
        # float16을 넣었다가 InvalidArgument를 받았다.
        x = np.ascontiguousarray(x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        try:
            out = self.sess.run(None, {self.iname: x})[0]
        except Exception as exc:
            self.error = repr(exc)[:300]
            return None
        d = np.asarray(out, np.float32).squeeze()
        if d.shape[:2] != (h0, w0):
            d = cv2.resize(d, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return d

    # 역depth라 "가까울수록 크다" — 대표값 규칙이 metric depth와 반대다.
    @staticmethod
    def range_for_box(depth, box, *, percentile: float = 25.0):
        """상대 역depth에서는 미터가 나오지 않는다. 항상 None."""
        return None


def class_prior_range(label: str, box, focal_px: float | None,
                      priors: dict) -> float | None:
    """클래스 크기 사전으로 낸 거리 — **추가 모델 없이** 나오는 두 번째 경로.

    거리 ≈ focal_px × 클래스_표준높이 / 박스_높이_px

    YOLOE가 이미 라벨을 내고 있으므로 이 경로는 사실상 공짜다. 사람·차량처럼
    크기 분산이 작은 클래스에서는 100m에서도 단안 네트워크보다 정확하다 — 100m의
    사람이 10px일 때 depth 네트워크는 그 영역에서 아무 말도 못 하지만, 높이
    10px과 1.7m라는 사전은 그대로 성립한다.

    크기 분산이 큰 클래스(가방, 상자, 부유물)에는 사전을 두지 않는다. 틀린 거리를
    자신 있게 내는 것보다 안 내는 편이 낫다.
    """
    if not focal_px or not priors:
        return None
    key = (label or "").strip().lower()
    height_m = priors.get(key)
    if not height_m:
        return None
    box_h = float(box[3]) - float(box[1])
    if box_h <= 1.0:
        return None
    return float(focal_px) * float(height_m) / box_h
