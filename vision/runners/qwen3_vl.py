#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qwen3_vl.py — Qwen3-VL-2B-Instruct 러너 (kind=vlm): 상황 분석 · 근거 설명 · 물체 위치 찾기(grounding) (2026-11-26).

러너 규약 (vlm):
  Runner(cfg: dict, device: str)          가중치를 읽어 GPU 에 올린다
  runner(bgr, prompt=None) -> (info, dets) info["text"] = 답. prompt 가 grounding 요청이면 dets 에 상자
  runner.describe(bgr, question) -> str    글 답만
  runner.ground(bgr, names) -> (info, dets) 이름 목록을 찾아 상자로 (픽셀 좌표, 입력 프레임 기준)
  runner.close()                           GPU 메모리를 돌려준다 (오케스트레이터가 GPU 가 모자라면 내린다)

det = {"cls": -1, "name": str, "conf": 1.0, "xyxy": [x1, y1, x2, y2]}  — VLM 은 확신도를 내지 않아 conf 는 1.0 고정.

cfg (모두 선택):
  model_path    가중치 폴더 (기본 models/qwen3-vl-2b-instruct, DS_HOME 기준)
  dtype         bf16 | fp16 (기본 bf16)
  quant         none | 4bit (bitsandbytes nf4 — 패키지가 있을 때만)
  min_pixels    모델에 넣는 영상 최소 화소 (기본 64*32*32)
  max_pixels    모델에 넣는 영상 최대 화소 (기본 640*32*32 ≈ 0.65 MP — 메모리 · 시간을 잡는다)
  max_new_tokens 기본 128 (글 답)          ground_max_new_tokens  기본 384 (상자 JSON 은 물체당 약 25 토큰)
  repetition_penalty  글 답에만 (기본 1.05 — 2B 가 「냉장고, 냉장고, …」 처럼 되풀이하는 것을 막는다). 상자 답은 1.0
  drop_full_frame 기본 0.95 — 화면의 이 비율 이상을 덮는 상자는 버린다 (없는 물체를 화면 전체로 답하는 버릇). null = 끔
  coord_mode    rel1000 (Qwen3-VL 기본: 0~1000 상대 좌표) | abs (모델 입력 영상의 픽셀 좌표, Qwen2.5-VL 방식)

디코딩은 결정적(greedy, do_sample=False)이다. 같은 프레임 · 같은 질문이면 같은 답이 나온다.
torch · transformers 는 Runner 를 만들 때만 import 한다 — 파싱 함수(parse_boxes)는 GPU 없이 시험할 수 있다.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

#: 한국어 질문 틀. {names} 는 찾을 물체 이름(쉼표로 이은 것)
PROMPTS = {
    "describe": "이 장면을 한국어 한 문장으로 설명해 줘.",
    "situation": "이 장면에서 사람·장애물·위험 요소가 있는지 한국어로 두세 문장으로 분석해 줘.",
    "explain": "이 장면에서 {names}(이)가 있다고 판단할 근거를 한국어로 한두 문장으로 설명해 줘.",
    "ground": ("이 이미지에서 다음 물체를 모두 찾아 줘: {names}. "
               "각 물체마다 {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"이름\"}} 형식의 JSON 배열로만 답해 줘. "
               "없으면 [] 로 답해 줘."),
}

#: 질문이 상자를 바라는지 — 「박스가 있어?」 같은 글 질문까지 상자 풀이로 가지 않게 좁게 잡는다
_GROUND_HINT = re.compile(r"bbox|grounding|좌표|\bjson\b", re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_NUM4 = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
_LABEL = re.compile(r"\"label\"\s*:\s*\"([^\"]*)\"")


def _ds_home():
    return Path(os.environ.get("DS_HOME") or Path(__file__).resolve().parents[2]).expanduser()


def _to_xyxy(box, w, h, mode, in_wh=None):
    """모델 좌표 → 입력 프레임 픽셀 좌표 (잘라서 화면 안으로). 뒤집힌 상자는 바로잡는다. 넓이 0 이면 None."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if mode == "abs" and in_wh:
        sx, sy = w / float(in_wh[0]), h / float(in_wh[1])
    else:
        sx, sy = w / 1000.0, h / 1000.0
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
    y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def _walk(obj, out, label=None):
    """JSON 값에서 (상자, 이름) 을 모은다 — 리스트 · 딕셔너리가 어떻게 겹쳐 있어도."""
    if isinstance(obj, dict):
        lab = obj.get("label") or obj.get("name") or obj.get("object") or label
        box = obj.get("bbox_2d") or obj.get("bbox") or obj.get("box") or obj.get("xyxy")
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            out.append((box, lab))
            return
        for k, v in obj.items():
            _walk(v, out, lab if k not in ("label", "name") else label)
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 4 and all(isinstance(v, (int, float)) for v in obj):
            out.append((obj, label))
            return
        for v in obj:
            _walk(v, out, label)


def parse_boxes(text, w, h, mode="rel1000", in_wh=None, default_name="object", max_area=None):
    """모델 답 글 → det 목록. JSON(코드 울타리 안 · 밖) 을 먼저, 안 되면 줄마다 숫자 넷을 찾는다.

    w, h     입력 프레임 크기 (픽셀)     mode  rel1000 | abs     in_wh  abs 일 때 모델 입력 영상 크기
    같은 이름 · 같은 상자는 한 번만 낸다. max_area(0~1) 를 주면 화면의 그 비율 이상을 덮는 상자는 버린다 —
    2B 는 없는 물체를 [0, 0, 1000, 1000] (화면 전체) 로 답하는 버릇이 있다 (2026-11-26 실측: pi12 · robot1)."""
    text = text or ""
    found = []
    blocks = _FENCE.findall(text) or [text]
    for b in blocks:
        b = b.strip()
        cand = [b]
        i, j = b.find("["), b.rfind("]")
        if 0 <= i < j:
            cand.append(b[i:j + 1])
        i, j = b.find("{"), b.rfind("}")
        if 0 <= i < j:
            cand.append("[" + b[i:j + 1] + "]")
        for c in cand:
            try:
                obj = json.loads(c)
            except (ValueError, TypeError):
                continue
            _walk(obj, found)
            break
    if not found:                                   # JSON 이 깨졌다 (잘린 답 · 따옴표 빠짐) — 줄 단위로 줍는다
        for chunk in re.split(r"\}\s*,?|\n", text):
            m = _NUM4.search(chunk)
            if m:
                lab = _LABEL.search(chunk)
                found.append(([float(g) for g in m.groups()], lab.group(1) if lab else None))
    dets, seen = [], set()
    for box, lab in found:
        xyxy = _to_xyxy(box, w, h, mode, in_wh)
        if xyxy is None:
            continue
        if max_area and (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) >= max_area * w * h:
            continue
        name = str(lab).strip() if lab else default_name
        key = (name, tuple(xyxy))
        if key in seen:
            continue
        seen.add(key)
        dets.append({"cls": -1, "name": name, "conf": 1.0, "xyxy": xyxy})
    return dets


class Runner:
    name = "qwen3_vl"
    kind = "vlm"

    def __init__(self, cfg: dict | None = None, device: str = "cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        cfg = dict(cfg or {})
        self.cfg = cfg
        self.device = device
        p = Path(str(cfg.get("model_path") or "models/qwen3-vl-2b-instruct")).expanduser()
        self.model_path = p if p.is_absolute() else _ds_home() / p
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError("Qwen3-VL 가중치가 없다: %s (tools/fetch_models.py --models qwen3_vl_2b)"
                                    % self.model_path)
        self.max_new_tokens = int(cfg.get("max_new_tokens", 128))
        self.ground_max_new_tokens = int(cfg.get("ground_max_new_tokens", 384))
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.05))
        self.coord_mode = str(cfg.get("coord_mode", "rel1000"))
        self.max_pixels = None
        self.drop_full_frame = cfg.get("drop_full_frame", 0.95)
        min_px = int(cfg.get("min_pixels", 64 * 32 * 32))
        max_px = int(cfg.get("max_pixels", 640 * 32 * 32))
        self.max_pixels = max_px
        dtype = torch.float16 if str(cfg.get("dtype", "bf16")) in ("fp16", "float16") else torch.bfloat16
        t0 = time.perf_counter()
        if str(device).startswith("cuda"):
            # cuBLAS 작업 공간(약 8MB)은 첫 행렬곱 때 캐시 할당기에서 잡혀 프로세스가 끝날 때까지 남는다. 가중치보다
            # 먼저 작은 조각에 잡아 둔다 — 아니면 4bit 가중치의 큰 조각(약 1.4GB) 안에 박혀 close() 뒤에도 그 조각
            # 전체가 놓이지 않는다 (2026-11-26 실측: 4bit close 후 reserved 1466MB → 이 줄로 해결)
            a = torch.ones(8, 8, device=device, dtype=dtype)
            (a @ a).sum().item()
            del a
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), size={"shortest_edge": min_px, "longest_edge": max_px})
        kw = {"dtype": dtype}
        if str(cfg.get("quant", "none")) == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            kw["device_map"] = {"": device}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_path), **kw)
        if "quantization_config" not in kw:
            self.model.to(device)
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        self._torch = torch

    # ------------------------------------------------------------ 추론
    def _generate(self, bgr, prompt, max_new_tokens=None, repetition_penalty=1.0):
        """한 프레임 + 질문 → (답 글, 모델 입력 영상 크기 (w, h))."""
        import cv2
        from PIL import Image
        torch = self._torch
        h, w = bgr.shape[:2]
        if self.max_pixels and h * w > self.max_pixels * 1.2:   # 큰 프레임은 미리 줄인다 (처리기의 CPU 축소보다 빠르다)
            s = (self.max_pixels / float(h * w)) ** 0.5
            bgr = cv2.resize(bgr, (max(32, round(w * s)), max(32, round(h * s))), interpolation=cv2.INTER_AREA)
        rgb = Image.fromarray(bgr[:, :, ::-1].copy())
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[rgb], return_tensors="pt").to(self.device)
        thw = inputs["image_grid_thw"][0].tolist()          # (t, h 패치, w 패치) — 패치 크기는 모델 설정
        ps = int(getattr(self.processor.image_processor, "patch_size", 16))
        in_wh = (thw[2] * ps, thw[1] * ps)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens or self.max_new_tokens),
                                      do_sample=False, repetition_penalty=float(repetition_penalty))
        ans = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return ans.strip(), in_wh

    def __call__(self, bgr, prompt: str | None = None):
        """prompt 가 없으면 한 문장 장면 설명. 상자를 바라는 질문(bbox · 좌표 · JSON)이면 답을 상자로 푼다."""
        t0 = time.perf_counter()
        prompt = prompt or PROMPTS["describe"]
        grounding = bool(_GROUND_HINT.search(prompt))
        if grounding:
            ans, in_wh = self._generate(bgr, prompt, self.ground_max_new_tokens, 1.0)
        else:
            ans, in_wh = self._generate(bgr, prompt, self.max_new_tokens, self.repetition_penalty)
        h, w = bgr.shape[:2]
        dets = parse_boxes(ans, w, h, self.coord_mode, in_wh, max_area=self.drop_full_frame) if grounding else []
        return {"text": ans, "model_input_wh": list(in_wh),
                "ms": round((time.perf_counter() - t0) * 1000, 1)}, dets

    def describe(self, bgr, question: str | None = None) -> str:
        return self(bgr, question or PROMPTS["describe"])[0]["text"]

    def ground(self, bgr, names):
        names = [names] if isinstance(names, str) else list(names)
        return self(bgr, PROMPTS["ground"].format(names=", ".join(names)))

    # ------------------------------------------------------------ 내리기
    def close(self):
        """모델을 지우고 CUDA 캐시를 비운다. 다시 쓰려면 Runner 를 새로 만든다."""
        torch = getattr(self, "_torch", None)
        for a in ("model", "processor"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
