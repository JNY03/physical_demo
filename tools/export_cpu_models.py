#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_cpu_models.py — 기기(CPU) 모델을 ONNX · NCNN 으로 굳힌다 (2026-11-27)

서버(x86 · ultralytics 가 있는 venv)에서 한 번 돌리고, 나온 파일을 기기에 싣는다. 기기 실행기
(vision/runners/yolo_seg_cpu.py)는 ultralytics 없이 onnxruntime(또는 ncnn)만 쓴다.

  venv/bin/python tools/export_cpu_models.py yolo26n_seg            # models/yolo26n-seg/yolo26n-seg_{320,640}.onnx (+ ncnn)
  venv/bin/python tools/export_cpu_models.py yoloe26n_vocab          # models/yoloe-26n-seg/yoloe-26n-seg-vocab_{320,640}.onnx
  venv/bin/python tools/export_cpu_models.py yoloe26n_vocab --vocab person car door "plastic bottle"

yoloe26n_vocab 은 **어휘를 굳힌** YOLOE-26n-seg 다 — set_classes 로 글 임베딩을 머리에 녹여 넣으므로 기기에는
글 인코더가 필요 없다(zeroshot_detect 기기 단계의 「미리 굳힌 프롬프트 임베딩만」). 새 어휘는 여기서 다시 굳힌다.
클래스 이름은 ONNX 메타데이터 names 에 들어간다 — 실행기가 거기서 읽는다.

가중치(.pt)는 레지스트리 source 의 URL 로 받는다(tools/fetch_models.py). 글 인코더 mobileclip2_b.ts 는 models/ 에서 찾는다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 기본 굳힌 어휘 — 로봇 순찰 · 하천 · 실내 이동에서 자주 찾는 것 (바꾸려면 --vocab)
DEFAULT_VOCAB = [
    "person", "car", "truck", "bus", "bicycle", "motorcycle", "dog", "cat",
    "door", "stairs", "chair", "table", "bench", "traffic cone", "fire extinguisher",
    "backpack", "suitcase", "box", "bag", "bottle", "plastic bottle", "trash",
    "floating debris", "boat", "tree", "pole", "fence", "wall", "window", "sign",
]
SIZES = (320, 640)


def _models_dir():
    from ds_resources import settings
    return Path(settings.resolve("models"))


def _export(model, out_dir, stem, fmt, imgsz):
    """ultralytics 로 내보내고 <stem>_<imgsz>.<확장자> 로 옮긴다"""
    # nms=False → 끝까지(end2end) 내보낸다: NMS 없는 one-to-one 머리 (YOLO26 · YOLOE-26). ncnn 은 TopK 가 없어 ultralytics 가 끈다
    got = Path(model.export(format=fmt, imgsz=imgsz, dynamic=False, simplify=True, opset=17, nms=False
                            ) if fmt == "onnx" else model.export(format=fmt, imgsz=imgsz))
    if fmt == "onnx":
        dst = out_dir / ("%s_%d.onnx" % (stem, imgsz))
    else:
        dst = out_dir / ("%s_ncnn_%d" % (stem, imgsz))
        if dst.exists():
            shutil.rmtree(dst)
    shutil.move(str(got), str(dst))
    print(dst)
    return dst


def yolo26n_seg(args):
    from ultralytics import YOLO
    d = _models_dir() / "yolo26n-seg"
    for sz in SIZES:
        m = YOLO(str(d / "yolo26n-seg.pt"))
        _export(m, d, "yolo26n-seg", "onnx", sz)
        if args.ncnn:
            m = YOLO(str(d / "yolo26n-seg.pt"))
            _export(m, d, "yolo26n-seg", "ncnn", sz)


def yoloe26n_vocab(args):
    from ultralytics import YOLOE
    d = _models_dir() / "yoloe-26n-seg"
    vocab = list(args.vocab or DEFAULT_VOCAB)
    cwd = os.getcwd()
    os.chdir(_models_dir())                       # 글 인코더(mobileclip2_b.ts)를 models/ 에서 찾게
    try:
        for sz in SIZES:
            m = YOLOE(str(d / "yoloe-26n-seg.pt"))
            m.set_classes(vocab, m.get_text_pe(vocab))
            _export(m, d, "yoloe-26n-seg-vocab", "onnx", sz)
    finally:
        os.chdir(cwd)
    (d / "yoloe-26n-seg-vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")


JOBS = {"yolo26n_seg": yolo26n_seg, "yoloe26n_vocab": yoloe26n_vocab}


def main(argv=None):
    ap = argparse.ArgumentParser(description="기기(CPU) 모델 ONNX · NCNN 굳히기")
    ap.add_argument("job", choices=sorted(JOBS))
    ap.add_argument("--vocab", nargs="+", help="yoloe26n_vocab 의 굳힐 어휘 (없으면 기본 목록)")
    ap.add_argument("--ncnn", action="store_true", help="yolo26n_seg 를 NCNN 으로도")
    a = ap.parse_args(argv)
    JOBS[a.job](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
