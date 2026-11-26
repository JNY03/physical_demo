"""YOLO 분할(segment) CPU 실행기 — ONNX(onnxruntime) · NCNN (2026-11-27)

기기(파이 · GPU 없음)와 서버 CPU 여유분에서 돈다. **ultralytics · torch 를 쓰지 않는다** — numpy · cv2 ·
onnxruntime(또는 ncnn)만 있으면 되므로 이 파일만 기기 컨테이너에 복사해도 돈다(저장소 import 없음).

  Yolo26nSegCpuRunner     YOLO26n-seg (COCO 80종)                      models.yolo26n_seg_cpu
  Yoloe26nVocabCpuRunner  YOLOE-26n-seg, 어휘를 굳힌 ONNX (열린 어휘)     models.yoloe26n_vocab_cpu

가중치는 tools/export_cpu_models.py 로 굳힌다(입력 크기마다 파일 하나 — 정적 그래프가 CPU 에서 빠르다).
ONNX 는 end2end(NMS 없는 one-to-one 머리, (1, 300, 6+32))로 내보내고, NCNN 은 TopK 가 없어 원 머리
((1, 4+nc+32, A))로 나오므로 여기서 NMS 를 한다 — 두 꼴 모두 받는다.

cfg (vision_config.yaml models.<키>)
  model      "models/yolo26n-seg/yolo26n-seg_{imgsz}.onnx" — {imgsz} 를 채운다. NCNN 은 폴더 (model.ncnn.param · bin)
  backend    onnx | ncnn (기본: model 이 폴더면 ncnn)
  imgsz      320 (기기 기본) · 640
  conf       0.25        iou   0.5 (NCNN 원 머리에서만)     max_det  100
  threads    4 (파이 5 코어 수)     masks  true (외곽선 _poly · polygon 을 만든다)
  names      [..] 클래스 이름 (없으면 ONNX 메타데이터 names · NCNN 폴더의 metadata.yaml)
"""
from __future__ import annotations

import ast
import os
import time

import cv2
import numpy as np


def _names_from_meta(text):
    """ultralytics 메타데이터의 names 칸 ("{0: 'person', ...}") → [이름…]"""
    if not text:
        return None
    try:
        d = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(d, dict):
        return [str(d[k]) for k in sorted(d)]
    return [str(x) for x in d]


def _names_from_yaml(path):
    """NCNN 폴더의 metadata.yaml names 칸 — PyYAML 없이 「  0: person」 줄만 읽는다"""
    out, inside = {}, False
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("names:"):
                    inside = True
                    continue
                if inside:
                    if not line.startswith(" "):
                        break
                    k, _, v = line.strip().partition(":")
                    if k.isdigit():
                        out[int(k)] = v.strip().strip("'\"")
    except OSError:
        return None
    return [out[k] for k in sorted(out)] or None


def letterbox(bgr, size):
    """긴 변을 size 에 맞추고 회색(114)으로 채운다 → (그림, 배율, (왼쪽, 위) 채움)"""
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
    left, top = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def nms_raw(pred, conf, iou, max_det, nm):
    """원 머리 (4+nc+nm, A) → end2end 꼴 (K, 6+nm) [x1 y1 x2 y2 score cls coeffs] (letterbox px)"""
    p = pred.T                                   # (A, 4+nc+nm)
    nc = p.shape[1] - 4 - nm
    scores = p[:, 4:4 + nc]
    cls = scores.argmax(1)
    sc = scores[np.arange(len(p)), cls]
    keep = sc >= conf
    if not keep.any():
        return np.zeros((0, 6 + nm), np.float32)
    p, cls, sc = p[keep], cls[keep], sc[keep]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    # 클래스별 NMS — 클래스마다 상자를 멀리 밀어 한 번에 (ultralytics 와 같은 수)
    off = cls[:, None].astype(np.float32) * 4096.0
    b = xyxy + off
    idx = cv2.dnn.NMSBoxes(np.c_[b[:, :2], b[:, 2:] - b[:, :2]].tolist(), sc.tolist(), conf, iou)
    idx = np.asarray(idx, dtype=int).reshape(-1)[:max_det]
    return np.c_[xyxy[idx], sc[idx], cls[idx], p[idx, 4 + nc:]].astype(np.float32)


def resolve_path(path):
    """상대 경로(models/…)는 DS_HOME(없으면 저장소 루트 — 이 파일의 두 단계 위) 기준. 저장소 import 없이"""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    home = os.environ.get("DS_HOME") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(home, path)


class _OnnxBackend:
    def __init__(self, path, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = _names_from_meta(meta.get("names"))

    def __call__(self, x):
        out = self.sess.run(None, {self.inp: x})
        return out[0][0], out[1][0]              # (300, 38) 또는 (4+nc+32, A) · proto (32, h, w)


class _NcnnBackend:
    def __init__(self, folder, threads):
        import ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = int(threads)
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(os.path.join(folder, "model.ncnn.param"))
        self.net.load_model(os.path.join(folder, "model.ncnn.bin"))
        self.ncnn = ncnn
        self.names = _names_from_yaml(os.path.join(folder, "metadata.yaml"))

    def __call__(self, x):
        ex = self.net.create_extractor()
        ex.input("in0", self.ncnn.Mat(x[0]).clone())
        _, o0 = ex.extract("out0")
        _, o1 = ex.extract("out1")
        return np.array(o0), np.array(o1)


class YoloSegCpuRunner:
    """YOLO 분할 ONNX · NCNN 실행기 (detect · segment 약속)"""
    name = "yolo_seg_cpu"
    kind = "segment"
    default_model = None

    def __init__(self, cfg: dict, device: str = "cpu"):
        cfg = dict(cfg or {})
        self.imgsz = int(cfg.get("imgsz", 320))
        path = str(cfg.get("model") or self.default_model or "").replace("{imgsz}", str(self.imgsz))
        if not path:
            raise ValueError("%s: cfg.model 이 없다" % self.name)
        path = resolve_path(path)
        backend = cfg.get("backend") or ("ncnn" if os.path.isdir(path) else "onnx")
        threads = int(cfg.get("threads", 4))
        self.backend = backend
        self.path = path
        self.net = _NcnnBackend(path, threads) if backend == "ncnn" else _OnnxBackend(path, threads)
        self.names = [str(x) for x in (cfg.get("names") or self.net.names or [])]
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.5))
        self.max_det = int(cfg.get("max_det", 100))
        self.masks = bool(cfg.get("masks", True))

    def _decode(self, out, proto):
        nm = proto.shape[0]
        if out.shape[1] == 6 + nm:                                       # (300, 6+nm) — 앵커 수는 38 이 될 수 없다
            rows = out[out[:, 4] >= self.conf][: self.max_det]          # end2end
        else:
            rows = nms_raw(out, self.conf, self.iou, self.max_det, nm)
        return rows

    def __call__(self, bgr: np.ndarray):
        t0 = time.perf_counter()
        h0, w0 = bgr.shape[:2]
        img, r, (px, py) = letterbox(bgr, self.imgsz)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out, proto = self.net(x)
        rows = self._decode(out, proto)
        nm, ph, pw = proto.shape
        sx, sy = pw / self.imgsz, ph / self.imgsz                      # letterbox px → proto px
        dets = []
        for row in rows:
            x1, y1, x2, y2, sc, c = (float(v) for v in row[:6])
            ci = int(c)
            box = [max(0.0, (x1 - px) / r), max(0.0, (y1 - py) / r),
                   min(float(w0), (x2 - px) / r), min(float(h0), (y2 - py) / r)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            d = {"cls": ci, "name": self.names[ci] if 0 <= ci < len(self.names) else str(ci),
                 "conf": round(sc, 4), "xyxy": [round(v, 1) for v in box]}
            if self.masks:
                poly = self._poly(row[6:6 + nm], proto, (x1, y1, x2, y2), sx, sy, r, px, py)
                if poly is not None:
                    d["_poly"] = poly
                    d["polygon"] = [[round(float(a), 1), round(float(b), 1)] for a, b in poly]
            dets.append(d)
        info = {"infer_ms": round((time.perf_counter() - t0) * 1000.0, 2), "imgsz": self.imgsz,
                "backend": self.backend, "checkpoint": os.path.basename(self.path.rstrip("/"))}
        return info, dets

    @staticmethod
    def _poly(coef, proto, box, sx, sy, r, px, py):
        """상자 안 proto 영역만 곱해서 가장 큰 외곽선 하나 → 원본 px (N, 2). proto 해상도에서 끝내 싸다"""
        nm, ph, pw = proto.shape
        bx1, by1 = int(max(0, np.floor(box[0] * sx))), int(max(0, np.floor(box[1] * sy)))
        bx2, by2 = int(min(pw, np.ceil(box[2] * sx))), int(min(ph, np.ceil(box[3] * sy)))
        if bx2 <= bx1 or by2 <= by1:
            return None
        sub = proto[:, by1:by2, bx1:bx2].reshape(nm, -1)
        m = (coef @ sub).reshape(by2 - by1, bx2 - bx1) > 0.0             # sigmoid(x) > 0.5 ⇔ x > 0
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        c[:, 0] = ((c[:, 0] + bx1 + 0.5) / sx - px) / r
        c[:, 1] = ((c[:, 1] + by1 + 0.5) / sy - py) / r
        return c

    def close(self):
        self.net = None


class Yolo26nSegCpuRunner(YoloSegCpuRunner):
    """YOLO26n-seg — COCO 80종 분할 (기기 분할 · 탐지 슬롯)"""
    name = "yolo26n_seg_cpu"
    default_model = "models/yolo26n-seg/yolo26n-seg_{imgsz}.onnx"


class Yoloe26nVocabCpuRunner(YoloSegCpuRunner):
    """YOLOE-26n-seg, 어휘를 굳힌 ONNX — 열린 어휘 탐지 · 분할 (기기 zero-shot 슬롯)"""
    name = "yoloe26n_vocab_cpu"
    default_model = "models/yoloe-26n-seg/yoloe-26n-seg-vocab_{imgsz}.onnx"
