#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yolo26n_ncnn.py — 기기 단(라즈베리파이 5, CPU)용 YOLO26n 검출 실행기 (2026-11-26). provides [detect]

**혼자 도는 파일이다.** 저장소의 다른 모듈도, torch 도, ultralytics 도 import 하지 않는다 —
arm64 컨테이너에 이 파일 하나와 모델 폴더만 넣으면 된다. 필요한 것: numpy · opencv(-headless) 와
  · NCNN 모델 폴더 (yolo26n_ncnn_<imgsz>/model.ncnn.param · model.ncnn.bin · metadata.yaml) → pip ncnn
  · 또는 ONNX 파일 (yolo26n_<imgsz>.onnx · yolo26n_<imgsz>_int8.onnx)                     → pip onnxruntime
경로가 폴더 · .param 이면 NCNN, .onnx 면 onnxruntime 으로 돈다. (metadata.yaml 을 읽으려면 PyYAML, 없으면 COCO 80종.)

출력 두 모양을 다 받는다 (_decode):
  · ONNX end2end (export end2end=True) — (1, 300, 6) [x1 y1 x2 y2 conf cls]. NMS 가 없다(YOLO26 의 장점)
  · NCNN — PNNX 가 TopK 를 못 담아 ultralytics 가 one-to-many 머리 (1, 84, N) 로 내보낸다. 여기서 NMS(opencv)
둘 다 letterbox 된 입력 좌표라 원래 프레임 좌표로 되돌린다. **모델 입력 크기는 내보낼 때 고정**된다
(yolo26n_ncnn_320 은 320 만) — imgsz 를 따로 주지 말고 모델 폴더 · 파일을 골라라.

Runner(cfg, device) 규약:
  cfg   model (필수: 폴더 · .param · .onnx) · imgsz (기본: ONNX 입력 모양 → metadata.yaml → 640) ·
        conf 0.25 · iou 0.7 (NCNN 출력의 NMS) · max_det 100 · threads 4 (Pi 5 = 4코어) · classes (남길 COCO 이름, 비우면 전부)
  device  무시한다 (항상 CPU)
  __call__(bgr) -> ({"infer_ms", "model", "imgsz", "backend"}, [{"cls","name","conf","xyxy"}])

명령줄:
  기기에서 재기   python3 yolo26n_ncnn.py --model models/yolo26n_ncnn/yolo26n_ncnn_320 --images 'frames/*.jpg' --threads 4
  x86 에서 내보내기 (한 번)   YOLO_AUTOINSTALL=False python vision/runners/yolo26n_ncnn.py \
                  --export models/yolo26n.pt --sizes 320 640 --out models/yolo26n_ncnn
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()
COCO = tuple(n.replace("_", " ") for n in COCO)


def letterbox(bgr, size):
    """긴 변을 size 에 맞춰 줄이고 가운데에 114 회색으로 채운다 (ultralytics 와 같은 방식)."""
    import cv2
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else bgr
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    x = out[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0      # BGR→RGB, CHW, 0..1
    return np.ascontiguousarray(x), r, left, top


def _read_meta(folder):
    f = Path(folder) / "metadata.yaml"
    if not f.is_file():
        return {}
    try:
        import yaml
        return yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception:                                    # PyYAML 이 없으면 기본값
        return {}


class _Ncnn:
    def __init__(self, path, threads):
        import ncnn
        p = Path(path)
        param = p if p.suffix == ".param" else next(p.glob("*.param"))
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = int(threads)
        self.net.load_param(str(param))
        self.net.load_model(str(param.with_suffix(".bin")))
        self.ncnn = ncnn
        self.inp = self.net.input_names()[0]
        self.outs = sorted(self.net.output_names())
        self.folder = param.parent

    def __call__(self, x):
        with self.net.create_extractor() as ex:
            ex.input(self.inp, self.ncnn.Mat(x))
            return np.array(ex.extract(self.outs[0])[1])


class _Onnx:
    def __init__(self, path, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.folder = Path(path).parent
        shape = self.sess.get_inputs()[0].shape                  # [1, 3, H, W] — 고정이면 그 크기
        self.imgsz = shape[2] if isinstance(shape[2], int) else None

    def __call__(self, x):
        return self.sess.run(None, {self.inp: x[None]})[0][0]


class Runner:
    name = "yolo26n_ncnn"
    kind = "detect"
    provides = ("detect",)

    def __init__(self, cfg: dict, device: str = "cpu"):
        cfg = dict(cfg or {})
        path = Path(str(cfg.get("model") or "")).expanduser()
        if not path.exists():
            raise FileNotFoundError("YOLO26n 모델이 없다: %s" % path)
        threads = int(cfg.get("threads", 4))
        self.backend = "onnx" if path.suffix == ".onnx" else "ncnn"
        self.net = _Onnx(path, threads) if self.backend == "onnx" else _Ncnn(path, threads)
        meta = _read_meta(path if path.is_dir() else path.parent)
        names = meta.get("names")
        self.names = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(COCO))
        imgsz = cfg.get("imgsz") or getattr(self.net, "imgsz", None) or meta.get("imgsz") or 640
        self.imgsz = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
        self.conf = float(cfg.get("conf", 0.25))
        self.max_det = int(cfg.get("max_det", 100))
        self.iou = float(cfg.get("iou", 0.7))              # NMS 가 필요한 출력(NCNN)에서만 쓴다
        keep = {str(n).lower() for n in cfg.get("classes") or []}
        self.keep = {i for i, n in self.names.items() if n.lower() in keep} if keep else None
        self.model = path.name
        self(np.zeros((480, 640, 3), np.uint8))           # 첫 장 지연을 미리 치른다

    def __call__(self, bgr):
        t0 = time.perf_counter()
        x, r, left, top = letterbox(bgr, self.imgsz)
        y = self._decode(np.asarray(self.net(x), np.float32))
        y = y[y[:, 4] >= self.conf]
        if self.keep is not None:
            y = y[np.isin(y[:, 5].astype(int), list(self.keep))]
        y = y[np.argsort(-y[:, 4])][: self.max_det]
        h, w = bgr.shape[:2]
        dets = []
        for x1, y1, x2, y2, s, k in y:
            b = [(x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r]
            b = [min(max(b[0], 0), w), min(max(b[1], 0), h), min(max(b[2], 0), w), min(max(b[3], 0), h)]
            dets.append({"cls": int(k), "name": self.names.get(int(k), str(int(k))), "conf": round(float(s), 4),
                         "xyxy": [round(float(v), 1) for v in b]})
        ms = (time.perf_counter() - t0) * 1000
        return {"infer_ms": round(ms, 1), "model": self.model, "imgsz": self.imgsz, "backend": self.backend}, dets

    def _decode(self, out):
        """모델 출력 → (K, 6) [x1 y1 x2 y2 conf cls] (letterbox 좌표).

        end2end 로 내보낸 것(ONNX nms=True)은 (300, 6) 그대로. NCNN 은 TopK 를 못 담아 ultralytics 가
        one-to-many 머리 (4+nc, N) [cx cy w h 점수…] 로 내보낸다 — 그때는 여기서 NMS 를 한다(opencv)."""
        out = out.reshape(out.shape[-2], out.shape[-1]) if out.ndim == 3 else out
        if out.ndim == 2 and out.shape[-1] == 6:
            return out
        p = out.T if out.shape[0] < out.shape[1] else out                # (N, 4+nc)
        scores = p[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(p)), cls]
        m = conf >= self.conf
        p, cls, conf = p[m], cls[m], conf[m]
        if not len(p):
            return np.zeros((0, 6), np.float32)
        xyxy = np.concatenate([p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2], 1)
        import cv2
        # 클래스별 NMS — 클래스마다 좌표를 띄워 한 번에 (ultralytics agnostic=False 와 같은 효과)
        off = cls[:, None].astype(np.float32) * 4096.0
        b = xyxy + off
        keep = cv2.dnn.NMSBoxes(np.concatenate([b[:, :2], b[:, 2:] - b[:, :2]], 1).tolist(),
                                conf.tolist(), self.conf, self.iou)
        keep = np.asarray(keep, int).reshape(-1)
        return np.concatenate([xyxy[keep], conf[keep, None], cls[keep, None].astype(np.float32)], 1)

    def close(self):
        self.net = None


def export(weight, sizes, out):
    """yolo26n.pt → out/yolo26n_ncnn_<z>/ (NCNN) · out/yolo26n_<z>.onnx (end2end) — **x86 에서 한 번** 한다.

    이것만 ultralytics(+ ncnn · pnnx==20260526 · onnx · onnxslim) 가 필요하다. 기기는 결과 폴더만 받는다.
    YOLO_AUTOINSTALL=False 로 부른다 — 빠진 패키지를 ultralytics 가 몰래 설치하지 않게
    (onnxruntime 이 onnxruntime-gpu 를 덮어쓴 일이 있었다)."""
    import shutil
    import tempfile
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from ultralytics import YOLO
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for z in sizes:
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "yolo26n.pt"
            shutil.copy(weight, w)                         # 내보낸 것이 가중치 옆에 생기므로 임시 폴더에서
            d = Path(YOLO(str(w)).export(format="ncnn", imgsz=z, device="cpu"))
            dst = out / ("yolo26n_ncnn_%d" % z)
            dst.mkdir(exist_ok=True)
            for f in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
                shutil.copy(d / f, dst / f)
            o = YOLO(str(w)).export(format="onnx", imgsz=z, device="cpu", simplify=True, end2end=True)
            shutil.copy(o, out / ("yolo26n_%d.onnx" % z))
    print("내보냄:", out)


def main(argv=None):
    import argparse
    import glob
    import json
    ap = argparse.ArgumentParser(description="YOLO26n NCNN/ONNX 기기 실행기 — 재기 · 내보내기")
    ap.add_argument("--export", metavar="PT", help="yolo26n.pt 를 NCNN · ONNX 로 내보낸다 (--imgsz 여럿, --out)")
    ap.add_argument("--out", default="models/yolo26n_ncnn")
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 640])
    ap.add_argument("--model")
    ap.add_argument("--images", help="glob (따옴표로)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args(argv)
    if a.export:
        return export(a.export, a.sizes, a.out)
    if not (a.model and a.images):
        ap.error("--model 과 --images 가 필요하다 (또는 --export)")
    import cv2
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    run = Runner({"model": a.model, "threads": a.threads, "imgsz": a.imgsz or None, "conf": a.conf})
    files = sorted(glob.glob(a.images))[: a.limit]
    frames = [cv2.imread(f) for f in files]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit("이미지가 없다: %s" % a.images)
    for f in frames[: a.warmup]:
        run(f)
    ms, n = [], 0
    for f in frames:
        t0 = time.perf_counter()
        _, d = run(f)
        ms.append((time.perf_counter() - t0) * 1000)
        n += len(d)
    print(json.dumps({"model": run.model, "backend": run.backend, "imgsz": run.imgsz, "threads": a.threads,
                      "frames": len(frames), "p50_ms": round(float(np.percentile(ms, 50)), 1),
                      "p90_ms": round(float(np.percentile(ms, 90)), 1), "dets_per_frame": round(n / len(frames), 2)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
