"""엣지 RPN — class-agnostic 후보 제안기.

implements: AI-B-08, AI-S-04, AI-E-04, AI-C-05

**왜 말단이 아니라 엣지인가**: 이 RPN은 시점 정렬용이 아니다(그건 KLT 플로우가
6ms에 한다). 여기서의 목적은 하나뿐 — **OVD가 설명하지 못한 것을 찾는 것**이다.
"무엇이 안 잡혔나"를 알려면 "무엇이든 있다"를 먼저 알아야 하고, 그것이 RPN이
내는 class-agnostic 후보다.

말단에서 걷어낸 그 코드를 그대로 옮겼다. Pi 5에서는 프레임당 350ms였지만
RTX 3060에서는 훨씬 싸고, 무엇보다 이 경로는 **동기 예산 밖**이다 — 미검출
후보 분석은 화면 하단 패널용이라 늦어도 된다.

전처리 상수는 export_p1_onnx.py와 같아야 한다. 바꾸면 기존 recall 실측치와
비교 불가다(AI-B-01).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve(cfg: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (Path(cfg["_root"]) / p).resolve()


def fast_nms(boxes, scores, iou_thresh):
    if not boxes:
        return []
    xywh = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, scores, 0.0, iou_thresh)
    if idx is None or len(idx) == 0:
        return []
    return [int(i) for i in (idx.flatten() if hasattr(idx, "flatten") else idx)]


def bucketed_nms(boxes, scores, frame_area, buckets):
    """**크기 구간별로 따로 NMS한다. 작은 박스는 큰 박스에 지워지지 않는다.**

    단일 IoU NMS의 문제(2026-09-21 실측):
      * 0.4로 조이면 겹친 작은 객체가 함께 지워진다 — 물 위 잔가지 사이의 페트병.
        하천 GT recall이 0.410 → 0.324로 떨어졌다.
      * 0.7로 풀면 작은 객체는 살지만 거대 중복 박스가 그대로 남는다.

    두 요구가 충돌하는 이유는 **한 임계를 크기가 다른 박스들에 같이 쓰기 때문**이다.
    구간을 나누면 각자 맞는 임계를 쓸 수 있고, 무엇보다 **구간이 다르면 서로
    억제하지 않는다** — 부유목 더미(큰 박스)가 그 위의 페트병(작은 박스)을 지우는
    일이 원천적으로 없어진다. 이건 부작용이 아니라 이 설계의 목적이다
    (사용자 2026-09-21: "부유목 안으로 다른 클래스가 흡수되지 않도록").

    buckets: [{"max_area_frac": 0.01, "iou": 0.7}, ...] — 작은 것부터. 마지막 구간이
    나머지를 받는다.
    """
    if not boxes:
        return []
    order = sorted(buckets, key=lambda b: float(b.get("max_area_frac", 1.0)))
    groups: list[list[int]] = [[] for _ in order]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        frac = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / (frame_area or 1.0)
        for gi, spec in enumerate(order):
            if frac <= float(spec.get("max_area_frac", 1.0)) or gi == len(order) - 1:
                groups[gi].append(i)
                break
    keep: list[int] = []
    for gi, members in enumerate(groups):
        if not members:
            continue
        sub_b = [boxes[i] for i in members]
        sub_s = [scores[i] for i in members]
        for k in fast_nms(sub_b, sub_s, float(order[gi].get("iou", 0.5))):
            keep.append(members[k])
    keep.sort(key=lambda i: -scores[i])
    return keep


class Rpn:
    """class-agnostic 후보 제안기. 두 종류의 그래프를 같은 계약으로 감싼다.

    `kind`로 갈린다 — 상위 코드는 어느 쪽인지 모르고 (boxes, scores)만 받는다
    (AI-B-08: 상위 기능은 모델 형식에 직접 의존하지 않는다).

      p1_rpn   torchvision fasterrcnn_mobilenet_v3_large_fpn의 RPN까지만 내보낸 것.
               출력이 이미 (boxes, scores)이고 전처리는 ImageNet 정규화.
      yolov8   yolov8n을 **클래스를 무시하고** 후보 제안기로 쓴다. 출력 [1,84,N]에서
               앞 4개가 xywh, 뒤 80개가 클래스 점수이며 그 **최대값을 objectness로**
               쓴다. 전처리는 0~1 스케일만(ImageNet 정규화 없음) — 섞으면 점수가 깨진다.

    2026-09-21 교체 근거: 같은 GPU 평가에서 yolov8n class-agnostic이 P1보다
    빠르고(12.0 vs 20.4 ms) recall도 높았다(R@100 0.424 vs 0.310).
    Pi CPU에서도 같은 순서인지는 이 교체로 처음 확인한다 — GPU 비율은 옮기지 않는다(AI-B-01).
    """

    def __init__(self, cfg: dict):
        r = cfg["edge_rpn"]
        self.kind = r.get("kind", "p1_rpn")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(r.get("threads", 4))
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(resolve(cfg, r["model"])), opts,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        # **정사각을 가정하지 않는다.** 2026-09-21에 비정사각 export(1280x736)를
        # 올리자 shape[-1]만 읽어 1280x1280으로 리사이즈해 차원 오류가 났다.
        # H와 W를 따로 읽는다 — 종횡비 왜곡을 없애려고 비정사각을 쓰는 것이므로
        # 여기서 다시 정사각으로 만들면 목적 자체가 사라진다.
        shape = self.sess.get_inputs()[0].shape
        self.in_h, self.in_w = int(shape[2]), int(shape[3])
        self.cfg_rpn = r

    def propose(self, bgr):
        h0, w0 = bgr.shape[:2]
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rs = cv2.resize(rgb, (self.in_w, self.in_h))
        # 전처리가 모델마다 다르다. p1은 export 시점에 ImageNet 정규화를 전제했고
        # yolov8은 0~1 스케일만 쓴다 — 바꿔 넣으면 점수가 조용히 망가진다.
        arr = ((rs - MEAN) / STD) if self.kind == "p1_rpn" else rs
        x = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        pre_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        out = self.sess.run(None, {self.inp: x})
        infer_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        sx, sy = w0 / self.in_w, h0 / self.in_h
        if self.kind == "yolov8":
            # [1,84,N] → xywh(픽셀, 입력 해상도 기준) + 80 클래스 점수.
            # 클래스는 버리고 최대값만 objectness로 쓴다 = class-agnostic 후보.
            pred = np.asarray(out[0])[0]
            xywh, cls = pred[:4], pred[4:]
            scores_all = cls.max(axis=0)
            pre_conf = float(self.cfg_rpn.get("yolo_pre_conf", 0.01))
            sel = np.nonzero(scores_all >= pre_conf)[0]
            order = sel[np.argsort(-scores_all[sel])]
            cx, cy, ww, hh = xywh[0], xywh[1], xywh[2], xywh[3]
            b = [((cx[i] - ww[i] / 2) * sx, (cy[i] - hh[i] / 2) * sy,
                  (cx[i] + ww[i] / 2) * sx, (cy[i] + hh[i] / 2) * sy) for i in order]
            b = [tuple(float(v) for v in bb) for bb in b]
            s = [float(scores_all[i]) for i in order]
            n_raw = int(len(scores_all))
        else:
            boxes, scores = out
            order = np.argsort(-scores)
            b = [(float(boxes[i][0] * sx), float(boxes[i][1] * sy),
                  float(boxes[i][2] * sx), float(boxes[i][3] * sy)) for i in order]
            s = [float(scores[i]) for i in order]
            n_raw = int(len(scores))
        nb = self.cfg_rpn.get("nms_buckets")
        if nb:
            keep = bucketed_nms(b, s, float(w0 * h0), nb)
        else:
            keep = fast_nms(b, s, float(self.cfg_rpn.get("nms_iou", 0.4)))
        b, s = [b[i] for i in keep], [s[i] for i in keep]
        b, s = filter_candidates(b, s, self.cfg_rpn, w0, h0)
        nms_ms = (time.perf_counter() - t2) * 1000.0
        return b, s, {"pre_ms": round(pre_ms, 2), "infer_ms": round(infer_ms, 2),
                      "nms_ms": round(nms_ms, 2), "raw_candidates": n_raw}




def filter_candidates(boxes, scores, cfg_rpn: dict, w: int, h: int):
    """objectness·면적 필터 + top_n.

    면적 필터가 있는 이유는 실측이다: 2026-09-21 VisDrone 10장에서 상위 100개가
    건물 전면 같은 거대 박스로 쏠렸다(VisDrone 실측). 이건 recall 문제가
    아니라 **랭킹** 문제라, 모델을 바꾸지 않고 후처리에서 먼저 확인해 본다.
    기본값을 끄고 싶으면 config에서 max_box_area_frac=1.0으로 둔다.
    """
    min_obj = float(cfg_rpn.get("min_objectness", 0.0))
    max_frac = float(cfg_rpn.get("max_box_area_frac", 1.0))
    top_n = int(cfg_rpn.get("top_n", 100))
    frame_area = float(w * h) or 1.0

    out_b, out_s = [], []
    for b, s in zip(boxes, scores):
        if s < min_obj:
            continue
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if area / frame_area > max_frac:
            continue
        out_b.append(b)
        out_s.append(s)
        if len(out_b) >= top_n:
            break
    return out_b, out_s
