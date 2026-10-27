#!/usr/bin/env python3
"""
UniDepth + Dual YOLO Image Server
수신/송신 경로 변경 없이 처리 속도만 개선:
  1. in_queue maxsize=2 + put_nowait  — 레이턴시 누적 제거
  2. publisher 스레드 분리            — out_queue blocking 드레인, main loop 단순화
  3. JPEG 인코딩 publisher로 이동     — Worker GPU 루프에서 CPU 작업 분리
  4. orjson 교체                      — json.dumps 대비 3~5× 직렬화 속도
  5. pinned memory + non_blocking     — CPU→GPU 비동기 전송
  6. calibrate_depth_to_cm 중복 제거  — 루프 내 2회 → 1회

프레임 프로파일링 (FrameProfiler):
  - 전체 처리 FPS / depth 실행 FPS / zeroshot 실행 FPS / 발행 FPS / 수신 FPS
  - 단계별 평균 처리 시간 (ms): decode, depth, finetune, zeroshot, fusion, result
  - 드롭 프레임 수 (in_queue Full 횟수)
  - REPORT_INTERVAL 초마다 자동 출력 (기본 10초)
"""

import os
import time
import traceback
import queue
import threading
import collections

import numpy as np
import cv2
import orjson
import redis
import requests
import torch
import yaml

from multiprocessing import Process, Queue, Event, set_start_method
from ultralytics import YOLOWorld
from unidepth.models import UniDepthV2
from torchvision.ops import nms


# ── 설정 로드 ──────────────────────────────────────────────────────────────
CONFIG_PATH = os.getenv(
    "CONFIG_PATH",
    "/home/dg/capstone-db/image_server/unidepth_dual_yolo_config.yaml",
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


# ── Redis ──────────────────────────────────────────────────────────────────
REDIS_HOST = CFG["redis"]["host"]
REDIS_PORT = int(CFG["redis"]["port"])
REDIS_DB   = int(CFG["redis"]["db"])
REDIS_PASS = os.environ.get("REDIS_PASSWORD") or CFG["redis"]["password"] or None

# ── 스트림 ─────────────────────────────────────────────────────────────────
CAMERA_ID        = CFG["streams"]["camera_id"]
IN_STREAM        = CFG["streams"]["input"]
OUT_STREAM       = CFG["streams"]["output"]
YOLO_VIS_STREAM  = CFG["streams"]["yolo_vis"]
DEPTH_VIS_STREAM = CFG["streams"]["depth_vis"]

# ── 런타임 ─────────────────────────────────────────────────────────────────
USE_FP16          = bool(CFG["runtime"]["use_fp16"])
NUM_WORKERS       = int(CFG["runtime"]["num_workers"])
DEPTH_INTERVAL    = int(CFG["runtime"].get("depth_interval", 4))
ZEROSHOT_INTERVAL = int(CFG["runtime"].get("zeroshot_interval", 10))
VIS_FPS_LIMIT     = float(CFG["runtime"].get("vis_fps_limit", 5))

# ── 이미지 ─────────────────────────────────────────────────────────────────
W_HALF       = int(CFG["image"]["w_half"])
JPEG_QUALITY = int(CFG["image"]["jpeg_quality"])

# ── 위험 threshold ─────────────────────────────────────────────────────────
NEAR_TH = float(CFG["risk"]["near_threshold"])
MID_TH  = float(CFG["risk"]["mid_threshold"])

# ── 모델 경로 ──────────────────────────────────────────────────────────────
UNIDEPTH_MODEL_NAME  = CFG["models"]["unidepth"]["model_name"]
UNIDEPTH_INFER_IMGSZ = CFG["models"]["unidepth"].get("infer_imgsz", None)

FINETUNE_WEIGHT_PATH = CFG["models"]["finetune_detector"]["weight"]
FINETUNE_CONF        = float(CFG["models"]["finetune_detector"]["conf"])
FINETUNE_IMGSZ       = int(CFG["models"]["finetune_detector"]["imgsz"])

ZEROSHOT_WEIGHT_PATH = CFG["models"]["zeroshot_detector"]["weight"]
ZEROSHOT_CONF        = float(CFG["models"]["zeroshot_detector"]["conf"])
ZEROSHOT_IMGSZ       = int(CFG["models"]["zeroshot_detector"].get("imgsz", FINETUNE_IMGSZ))

# ── Fusion ─────────────────────────────────────────────────────────────────
FUSION_IOU = float(CFG["fusion"]["iou"])

# ── 클래스 / 그룹 ──────────────────────────────────────────────────────────
FINETUNE_CLASSES = CFG["classes"]["finetune"]
ZEROSHOT_CLASSES = CFG["classes"]["zeroshot"]
CLASS_TO_GROUP   = CFG["group_mapping"]
GROUP_PRIORITY   = CFG["group_priority"]
CLASS_CONF       = CFG.get("class_conf", {})

# ── 시각화 색상 ────────────────────────────────────────────────────────────
RISK_TO_COLOR = {
    "near":    (0, 0, 255),
    "mid":     (0, 165, 255),
    "far":     (0, 255, 0),
    "unknown": (128, 128, 128),
}

# ── HTTP control (경로 변경 없음) ──────────────────────────────────────────
CONTROL_URL = os.getenv(
    "CONTROL_URL",
    "http://210.110.250.33:7864/command/go1",
)

# ── 프로파일러 설정 ────────────────────────────────────────────────────────
REPORT_INTERVAL = 10.0   # 통계 출력 주기 (초)


# ══════════════════════════════════════════════════════════════════════════════
# FrameProfiler — 프레임별 처리 속도 및 단계별 시간 측정
# ══════════════════════════════════════════════════════════════════════════════

class FrameProfiler:
    """
    Worker / Main / Publisher 각각 독립 인스턴스를 생성해 사용.

    측정 항목
    ─────────────────────────────────────────────────────────
    Worker 인스턴스:
      • 전체 처리 FPS          — process()로 프레임 완료 시 호출
      • depth 실행 FPS         — depth()로 depth 추론 시 호출
      • zeroshot 실행 FPS      — zeroshot()로 zeroshot 추론 시 호출
      • 단계별 평균 ms         — stage(name, seconds)로 각 단계 소요 시간 기록
          decode / depth / finetune / zeroshot / fusion / result

    Main 인스턴스:
      • 수신 FPS               — recv()로 in_queue put 성공 시 호출
      • 드롭 프레임 수         — drop()으로 put_nowait Full 시 호출

    Publisher 인스턴스:
      • 발행 FPS               — publish()로 xadd 완료 시 호출

    REPORT_INTERVAL 초마다 report()가 자동으로 통계를 출력하고 카운터를 초기화.
    """

    def __init__(self, label: str, auto_report: bool = True):
        self.label       = label
        self.auto_report = auto_report
        self._lock       = threading.Lock()
        self._reset()

    def _reset(self):
        self._t0              = time.perf_counter()
        self._total_frames    = 0
        self._depth_frames    = 0
        self._zeroshot_frames = 0
        self._publish_frames  = 0
        self._recv_frames     = 0
        self._drop_frames     = 0
        # 단계별 누적 시간 (초)
        self._stage_times: dict[str, list] = collections.defaultdict(list)

    # ── 카운터 메서드 ──────────────────────────────────────────────────────

    def process(self):
        """Worker: 프레임 처리 완료 1회 — 자동 출력 트리거"""
        with self._lock:
            self._total_frames += 1
            self._maybe_report()

    def depth(self):
        """Worker: UniDepth 추론 실행 1회"""
        with self._lock:
            self._depth_frames += 1

    def zeroshot(self):
        """Worker: Zeroshot YOLO 추론 실행 1회"""
        with self._lock:
            self._zeroshot_frames += 1

    def publish(self):
        """Publisher: Redis 발행 1회 — 자동 출력 트리거"""
        with self._lock:
            self._publish_frames += 1
            self._maybe_report()

    def recv(self):
        """Main: 프레임 수신(in_queue put 성공) 1회 — 자동 출력 트리거"""
        with self._lock:
            self._recv_frames += 1
            self._maybe_report()

    def drop(self):
        """Main: 프레임 드롭(in_queue Full) 1회"""
        with self._lock:
            self._drop_frames += 1

    def stage(self, name: str, elapsed: float):
        """
        단계별 소요 시간 기록.
        name: 'decode' | 'depth' | 'finetune' | 'zeroshot' | 'fusion' | 'result'
        elapsed: 소요 시간 (초)
        """
        with self._lock:
            self._stage_times[name].append(elapsed)

    # ── 리포트 ────────────────────────────────────────────────────────────

    def _maybe_report(self):
        """_lock 보유 상태에서 호출 — REPORT_INTERVAL 경과 시 출력"""
        elapsed = time.perf_counter() - self._t0
        if elapsed >= REPORT_INTERVAL:
            self._print_report(elapsed)
            self._reset()

    def _fps(self, count, elapsed):
        return f"{count / elapsed:.2f} FPS"

    def _fmt_ms(self, times: list):
        if not times:
            return "N/A"
        avg = sum(times) / len(times) * 1000
        mn  = min(times) * 1000
        mx  = max(times) * 1000
        return f"avg {avg:.1f}ms  min {mn:.1f}  max {mx:.1f}  (n={len(times)})"

    def _print_report(self, elapsed: float):
        W = 62
        bar  = "━" * W
        sep  = "─" * W

        # ── FPS 요약 3줄 ──────────────────────────────────────────────────
        recv_fps    = self._recv_frames    / elapsed if self._recv_frames    else 0.0
        process_fps = self._total_frames   / elapsed if self._total_frames   else 0.0
        pub_fps     = self._publish_frames / elapsed if self._publish_frames else 0.0
        drop_total  = self._recv_frames + self._drop_frames
        drop_pct    = self._drop_frames / max(1, drop_total) * 100

        lines = [f"\n{bar}"]

        # ── 헤더 ──────────────────────────────────────────────────────────
        if self.label == "Main":
            lines += [
                f"  [{self.label}]  {elapsed:.1f}s  |  수신 → 처리 → 발행 FPS 요약",
                sep,
                f"  {'항목':<18}{'FPS':>8}  {'프레임':>8}  {'비고':<20}",
                sep,
                f"  {'수신 (Redis in)':<18}{recv_fps:>8.2f}  {self._recv_frames:>8}프레임  드롭 {self._drop_frames}f ({drop_pct:.1f}%)",
            ]
        elif self.label.startswith("Worker"):
            depth_fps = self._depth_frames   / elapsed if self._depth_frames   else 0.0
            zero_fps  = self._zeroshot_frames / elapsed if self._zeroshot_frames else 0.0
            lines += [
                f"  [{self.label}]  {elapsed:.1f}s  |  수신 → 처리 → 발행 FPS 요약",
                sep,
                f"  {'항목':<18}{'FPS':>8}  {'프레임':>8}  {'비고':<20}",
                sep,
                f"  {'처리 (전체)':<18}{process_fps:>8.2f}  {self._total_frames:>8}프레임",
                f"  {'  └ depth':<18}{depth_fps:>8.2f}  {self._depth_frames:>8}프레임  매 {DEPTH_INTERVAL}f 실행",
                f"  {'  └ zeroshot':<18}{zero_fps:>8.2f}  {self._zeroshot_frames:>8}프레임  매 {ZEROSHOT_INTERVAL}f 실행",
            ]
            if self._stage_times:
                lines.append(sep)
                lines.append(f"  {'단계':<12}  {'avg':>8}   {'min':>8}   {'max':>8}   n")
                lines.append(sep)
                for name in ["decode", "depth", "finetune", "zeroshot", "fusion", "result"]:
                    if name in self._stage_times:
                        t = self._stage_times[name]
                        avg = sum(t)/len(t)*1000
                        mn  = min(t)*1000
                        mx  = max(t)*1000
                        lines.append(
                            f"  {name:<12}  {avg:>7.1f}ms  {mn:>7.1f}ms  {mx:>7.1f}ms  {len(t)}"
                        )
        elif self.label == "Publisher":
            lines += [
                f"  [{self.label}]  {elapsed:.1f}s  |  수신 → 처리 → 발행 FPS 요약",
                sep,
                f"  {'항목':<18}{'FPS':>8}  {'프레임':>8}  {'비고':<20}",
                sep,
                f"  {'발행 (Redis out)':<18}{pub_fps:>8.2f}  {self._publish_frames:>8}프레임  JSON+시각화",
            ]

        lines.append(bar)
        print("\n".join(lines), flush=True)

    def force_report(self):
        """주기 무관하게 즉시 출력 (종료 시 호출용)"""
        with self._lock:
            elapsed = time.perf_counter() - self._t0
            if elapsed > 0:
                self._print_report(elapsed)


# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════════

def encode_bgr_to_jpeg(img_bgr, quality=85):
    if img_bgr is None:
        return None
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def extract_left_frame(combined_bgr):
    h, w = combined_bgr.shape[:2]

    if w <= W_HALF:
        return combined_bgr.copy()

    frame = combined_bgr[:, :W_HALF].copy()

    PAD_LEFT = 0  # 왼쪽에 빈 공간 추가 → 영상 내용은 오른쪽으로 이동

    shifted = np.empty_like(frame)

    # 왼쪽 빈 공간은 검은색으로 채움
    shifted[:, :PAD_LEFT] = 0

    # 원본을 오른쪽으로 PAD_LEFT만큼 이동
    shifted[:, PAD_LEFT:] = frame[:, :W_HALF - PAD_LEFT]

    return shifted

def depth_to_colormap(depth):
    if depth is None:
        return None
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        d_min = float(np.percentile(depth[valid], 2))
        d_max = float(np.percentile(depth[valid], 98))
        if not np.isfinite(d_min) or not np.isfinite(d_max) or abs(d_max - d_min) < 1e-8:
            norm = np.zeros_like(depth, dtype=np.uint8)
        else:
            norm = ((depth - d_min) / (d_max - d_min) * 255.0)
            norm = np.nan_to_num(norm, nan=0.0, posinf=255.0, neginf=0.0)
            norm = norm.clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def clamp_bbox(x1, y1, x2, y2, frame_shape):
    h, w = frame_shape[:2]
    x1 = max(0, min(w - 1, int(float(x1))))
    y1 = max(0, min(h - 1, int(float(y1))))
    x2 = max(0, min(w,     int(float(x2))))
    y2 = max(0, min(h,     int(float(y2))))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def get_object_depth(depth, x1, y1, x2, y2):
    """
    bbox 내부 depth를 안정적으로 추출한다.

    변경점:
    - bbox 전체가 아니라 중앙부 ROI만 사용해서 배경/바닥/경계 픽셀 혼입 감소
    - p25 대신 p40 사용해서 가까운 노이즈 픽셀에 덜 민감하게 처리
    """
    if depth is None:
        return None

    w = int(x2 - x1)
    h = int(y2 - y1)
    if w <= 0 or h <= 0:
        return None

    # bbox 가장자리 18% 제거: 물체-배경 경계, 바닥, 그림자 혼입 완화
    mx = int(w * 0.18)
    my = int(h * 0.18)

    rx1 = int(x1 + mx)
    ry1 = int(y1 + my)
    rx2 = int(x2 - mx)
    ry2 = int(y2 - my)

    # 작은 bbox는 중앙부 crop이 너무 작아질 수 있으므로 원래 bbox로 fallback
    if rx2 <= rx1 or ry2 <= ry1 or (rx2 - rx1) * (ry2 - ry1) < 20:
        rx1, ry1, rx2, ry2 = int(x1), int(y1), int(x2), int(y2)

    roi = depth[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size == 0:
        return None

    # 기존 p25는 가까운 픽셀 노이즈에 민감했으므로 p40 사용
    return float(np.percentile(valid, 40))


def risk_from_depth(depth_value):
    if depth_value is None:
        return "unknown"
    if depth_value <= NEAR_TH:
        return "near"
    if depth_value <= MID_TH:
        return "mid"
    return "far"


def calibrate_depth_to_cm(rel_depth, bbox=None, image_w=464, image_h=400, name=None):
    """
    UniDepth 신규 모델 보정식 + 어안렌즈 중심 기준 위치 보정.

    출력 형식은 기존과 동일하게 cm float를 반환한다.
    risk 판단은 기존 config의 near_threshold/mid_threshold(raw depth 기준)를 그대로 사용한다.
    """
    if rel_depth is None or not np.isfinite(rel_depth):
        return None

    # 1) depth 값 기반 기본 거리 보정
    distance_cm = (
        -4.61006769 * (rel_depth ** 3)
        + 28.91334586 * (rel_depth ** 2)
        - 11.16168081 * rel_depth
        + 15.20770323
    )

    # 2) 어안렌즈 보정: 좌/우가 아니라 이미지 중심에서의 반지름 r 기준
    if bbox is not None:
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        nx = (cx - image_w / 2.0) / (image_w / 2.0)
        ny = (cy - image_h / 2.0) / (image_h / 2.0)

        # 0.0 = 이미지 중심, 1.0 = 외곽/코너 방향
        r = (nx ** 2 + ny ** 2) ** 0.5
        r = min(1.0, max(0.0, r))

        # 중심~중간부는 더 줄임
        center_start = 0.55
        if r < center_start:
            center_ratio = 1.0 - (r / center_start)
            center_gain = 1.0 - 0.16 * center_ratio
            distance_cm *= center_gain

        # 외곽부 보정은 약하게 적용
        edge_start = 0.42
        if r > edge_start:
            edge_ratio = (r - edge_start) / (1.0 - edge_start)
            edge_ratio = min(1.0, max(0.0, edge_ratio))

            edge_gain = 1.0 + 0.45 * (edge_ratio ** 1.35)
            edge_gain = min(edge_gain, 1.35)

            distance_cm *= edge_gain

    return max(0.0, float(distance_cm))



def make_track_key(det, frame_w, frame_h):
    """
    간단한 temporal smoothing용 key.
    동일 객체를 완벽히 tracking하지는 않지만, 클래스/소스/화면 위치 grid를 묶어
    프레임 간 거리 튐을 줄인다.
    """
    x1, y1, x2, y2 = det["bbox_xyxy"]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    gx = int(np.clip(cx / max(1, frame_w) * 4, 0, 3))
    gy = int(np.clip(cy / max(1, frame_h) * 3, 0, 2))

    return f"{det['source']}:{det['name']}:{gx}:{gy}"


def smooth_distance_cm(distance_cm, track_key, smooth_state, alpha=0.35):
    """
    EMA smoothing.
    alpha가 낮을수록 안정적이고 반응은 느려진다.
    alpha=0.35는 흔들림 완화와 반응성 사이의 절충값.
    """
    if distance_cm is None:
        return None

    now = time.time()
    prev = smooth_state.get(track_key)

    if prev is None:
        smoothed = float(distance_cm)
    else:
        prev_value = float(prev["value"])
        smoothed = alpha * float(distance_cm) + (1.0 - alpha) * prev_value

    smooth_state[track_key] = {"value": smoothed, "t": now}

    # 오래된 key 정리
    stale_keys = [k for k, v in smooth_state.items() if now - v.get("t", now) > 2.0]
    for k in stale_keys:
        smooth_state.pop(k, None)

    return smoothed


def extract_depth_from_outputs(outputs):
    if isinstance(outputs, dict):
        for key in ["depth", "metric_depth", "pred", "prediction", "out"]:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
        raise RuntimeError(f"UniDepth depth key not found. keys={list(outputs.keys())}")
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def tensor_to_depth_numpy(depth_tensor):
    if depth_tensor is None:
        return None
    if depth_tensor.ndim == 4:
        if depth_tensor.shape[1] == 1:
            depth_tensor = depth_tensor[0, 0]
        elif depth_tensor.shape[-1] == 1:
            depth_tensor = depth_tensor[0, :, :, 0]
        else:
            depth_tensor = depth_tensor[0, 0]
    elif depth_tensor.ndim == 3:
        if depth_tensor.shape[0] == 1:
            depth_tensor = depth_tensor[0]
        elif depth_tensor.shape[-1] == 1:
            depth_tensor = depth_tensor[:, :, 0]
        else:
            depth_tensor = depth_tensor[0]
    elif depth_tensor.ndim == 2:
        pass
    else:
        raise RuntimeError(f"Unexpected depth tensor shape: {tuple(depth_tensor.shape)}")
    return depth_tensor.detach().float().cpu().numpy()


def safe_class_name(class_list, cls_id):
    return class_list[cls_id] if 0 <= cls_id < len(class_list) else f"cls{cls_id}"


def get_class_min_conf(source, class_name):
    source_conf_cfg = CLASS_CONF.get(source, {})
    return float(source_conf_cfg.get(class_name, source_conf_cfg.get("default", 0.0)))


def parse_yolo_results(results, class_list, source, frame_shape):
    detections = []
    if len(results) == 0 or results[0].boxes is None:
        return detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = clamp_bbox(
            box.xyxy[0][0], box.xyxy[0][1],
            box.xyxy[0][2], box.xyxy[0][3],
            frame_shape,
        )
        cls_id = int(box.cls[0])
        name   = safe_class_name(class_list, cls_id)
        conf   = float(box.conf[0]) if box.conf is not None else 0.0
        min_conf = get_class_min_conf(source, name)
        if conf < min_conf:
            continue
        group = CLASS_TO_GROUP.get(name, "UNKNOWN_OBSTACLE")
        detections.append({
            "name":       name,
            "group":      group,
            "source":     source,
            "confidence": conf,
            "bbox_xyxy":  [x1, y1, x2, y2],
        })
    return detections


def fuse_detections(detections):
    if not detections:
        return []
    boxes_t = torch.tensor([d["bbox_xyxy"] for d in detections], dtype=torch.float32)
    scores  = []
    for d in detections:
        group_bonus  = GROUP_PRIORITY.get(d["group"], 1.0) / 100.0
        source_bonus = 0.03 if d["source"] == "finetune" else 0.0
        scores.append(float(d["confidence"]) + group_bonus + source_bonus)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep     = nms(boxes_t, scores_t, FUSION_IOU)
    return [detections[i] for i in keep.tolist()]


def should_publish_vis(last_vis_time):
    if VIS_FPS_LIMIT <= 0:
        return True
    return (time.time() - last_vis_time) >= (1.0 / VIS_FPS_LIMIT)


def fetch_state_change_from_http():
    """HTTP 경로 변경 없이 원본 그대로 유지"""
    try:
        response = requests.get(CONTROL_URL, timeout=3)
        response.raise_for_status()
        payload      = response.json()
        state_change = bool(payload.get("state_change", True))
        # [muted] control json received
        return state_change
    except Exception as exc:
        # [muted] control json fetch failed
        return True


# ══════════════════════════════════════════════════════════════════════════════
# [최적화 2+3] Publisher 스레드 — out_queue blocking 드레인 + JPEG 인코딩 담당
# ══════════════════════════════════════════════════════════════════════════════

def publisher_thread(out_queue: Queue, rds: redis.Redis, stop_event: threading.Event):
    """
    out_queue에서 결과를 blocking으로 꺼내 Redis에 발행.
    JPEG 인코딩 / depth colormap 생성도 여기서 처리 → Worker GPU 루프 단순화.

    out_queue 메시지 형식:
      {
        "timestamp":    str,
        "result":       dict,            # [최적화 4] orjson 직렬화 전 dict
        "state_change": bool,
        "yolo_vis_np":  np.ndarray | None,
        "depth_np":     np.ndarray | None,
      }
    """
    # [muted] Publisher 시작
    profiler = FrameProfiler("Publisher")

    while not stop_event.is_set():
        try:
            item = out_queue.get(timeout=0.1)   # blocking — CPU 0%
        except queue.Empty:
            continue
        if item is None:
            break

        try:
            ts           = item["timestamp"].encode("utf-8")
            result       = item["result"]
            state_change = bool(item.get("state_change", True))

            # [최적화 4] orjson 직렬화
            result_bytes = orjson.dumps(result)

            # [최적화 3] JPEG 인코딩은 publisher에서
            yolo_jpeg  = encode_bgr_to_jpeg(item.get("yolo_vis_np"), quality=JPEG_QUALITY)
            depth_jpeg = encode_bgr_to_jpeg(
                depth_to_colormap(item.get("depth_np")), quality=JPEG_QUALITY
            )

            # print(f"[Main] send check timestamp={item['timestamp']} state_change={state_change}")

            # OUT_STREAM: state_change 조건부 발행 (원본 동작 유지)
            state_change = True
            if state_change:
                rds.xadd(
                    OUT_STREAM,
                    {b"data": result_bytes, b"timestamp": ts},
                    maxlen=300,
                    approximate=True,
                )
            else:
                pass  # skip: state_change=False

            # 시각화 스트림: state_change 무관하게 항상 발행 (원본 동작 유지)
            if yolo_jpeg is not None:
                rds.xadd(
                    YOLO_VIS_STREAM,
                    {b"image": yolo_jpeg, b"timestamp": ts},
                    maxlen=100,
                    approximate=True,
                )
            if depth_jpeg is not None:
                rds.xadd(
                    DEPTH_VIS_STREAM,
                    {b"image": depth_jpeg, b"timestamp": ts},
                    maxlen=100,
                    approximate=True,
                )

            profiler.publish()

        except Exception as exc:
            print(f"[Publisher] error: {repr(exc)}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# Worker 프로세스
# ══════════════════════════════════════════════════════════════════════════════

def worker_task(worker_id, in_queue, out_queue, stop_event):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        # [muted] CUDA Initialized

    yolo_device = 0 if device == "cuda" else "cpu"

    print(f"[Worker {worker_id}] Loading Models...")

    finetune_model = YOLOWorld(FINETUNE_WEIGHT_PATH)
    finetune_model.set_classes(FINETUNE_CLASSES)

    zeroshot_model = YOLOWorld(ZEROSHOT_WEIGHT_PATH)
    zeroshot_model.set_classes(ZEROSHOT_CLASSES)

    depth_model = UniDepthV2.from_pretrained(UNIDEPTH_MODEL_NAME).to(device).eval()

    print(f"[Worker {worker_id}] Models Loaded.")

    last_depth         = None
    last_zeroshot_dets = []
    last_vis_time      = 0.0
    frame_count        = 0
    distance_smooth_state = {}
    profiler           = FrameProfiler(f"Worker{worker_id}")

    while not stop_event.is_set():
        try:
            item = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception:
            continue

        if item is None:
            break

        msg_id, jpeg_bytes, timestamp_str, state_change = item
        frame_count += 1

        try:
            # ── decode ────────────────────────────────────────────────────
            _t = time.perf_counter()
            arr          = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            combined_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if combined_bgr is None:
                continue
            frame    = extract_left_frame(combined_bgr)
            frame_h, frame_w = frame.shape[:2]
            profiler.stage("decode", time.perf_counter() - _t)

            # ── Depth: interval 기반 ───────────────────────────────────────
            if frame_count % DEPTH_INTERVAL == 0 or last_depth is None:
                _t = time.perf_counter()
                depth_input  = frame
                infer_w, infer_h = frame_w, frame_h

                if UNIDEPTH_INFER_IMGSZ is not None:
                    infer_w = int(UNIDEPTH_INFER_IMGSZ[0])
                    infer_h = int(UNIDEPTH_INFER_IMGSZ[1])
                    depth_input = cv2.resize(
                        frame, (infer_w, infer_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                rgb = cv2.cvtColor(depth_input, cv2.COLOR_BGR2RGB)

                # [최적화 5] pinned memory + non_blocking 비동기 전송
                rgb_tensor = (
                    torch.from_numpy(rgb.copy())
                    .permute(2, 0, 1)
                    .pin_memory()
                    .to(device, non_blocking=True)
                )

                with torch.inference_mode():
                    if device == "cuda" and USE_FP16:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = depth_model.infer(rgb_tensor, camera=None, normalize=True)
                    else:
                        outputs = depth_model.infer(rgb_tensor, camera=None, normalize=True)

                depth_t  = extract_depth_from_outputs(outputs)
                depth_np = tensor_to_depth_numpy(depth_t)

                if depth_np is not None and depth_np.shape[:2] != (frame_h, frame_w):
                    depth_np = cv2.resize(
                        depth_np, (frame_w, frame_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                last_depth = depth_np
                profiler.stage("depth", time.perf_counter() - _t)
                profiler.depth()

            # ── Finetune YOLO: 매 프레임 ───────────────────────────────────
            _t = time.perf_counter()
            f_res = finetune_model.predict(
                frame, conf=FINETUNE_CONF,
                imgsz=FINETUNE_IMGSZ, device=yolo_device, verbose=False,
            )
            finetune_dets = parse_yolo_results(f_res, FINETUNE_CLASSES, "finetune", frame.shape)
            profiler.stage("finetune", time.perf_counter() - _t)

            # ── Zeroshot YOLO: interval 기반 ──────────────────────────────
            if frame_count % ZEROSHOT_INTERVAL == 0 or frame_count == 1:
                _t = time.perf_counter()
                z_res = zeroshot_model.predict(
                    frame, conf=ZEROSHOT_CONF,
                    imgsz=ZEROSHOT_IMGSZ, device=yolo_device, verbose=False,
                )
                last_zeroshot_dets = parse_yolo_results(
                    z_res, ZEROSHOT_CLASSES, "zeroshot", frame.shape,
                )
                profiler.stage("zeroshot", time.perf_counter() - _t)
                profiler.zeroshot()

            # ── Fusion ────────────────────────────────────────────────────
            _t = time.perf_counter()
            raw_detections = fuse_detections(finetune_dets + last_zeroshot_dets)
            profiler.stage("fusion", time.perf_counter() - _t)

            # ── 결과 조합 ─────────────────────────────────────────────────
            _t               = time.perf_counter()
            publish_vis      = should_publish_vis(last_vis_time)
            yolo_vis_np      = frame.copy() if publish_vis else None
            final_detections = []
            has_near         = False

            for i, det in enumerate(raw_detections):
                x1, y1, x2, y2 = det["bbox_xyxy"]

                obj_depth = get_object_depth(last_depth, x1, y1, x2, y2)
                risk      = risk_from_depth(obj_depth)  # config의 near/mid threshold(raw depth 기준) 유지

                # [최적화 6] calibrate_depth_to_cm 1회만 호출 + bbox 기반 어안 보정 적용
                calibrated_cm_raw = calibrate_depth_to_cm(
                    obj_depth,
                    bbox=(x1, y1, x2, y2),
                    image_w=frame_w,
                    image_h=frame_h,
                    name=det["name"],
                )

                track_key = make_track_key(det, frame_w, frame_h)
                calibrated_cm = smooth_distance_cm(
                    calibrated_cm_raw,
                    track_key,
                    distance_smooth_state,
                    alpha=0.35,
                )

                if risk == "near":
                    has_near = True

                if yolo_vis_np is not None:
                    color    = RISK_TO_COLOR.get(risk, RISK_TO_COLOR["unknown"])
                    dist_str = f"{calibrated_cm:.1f}cm" if calibrated_cm is not None else "N/A"
                    label    = f"{det['name']} {dist_str} ({risk})"
                    cv2.rectangle(yolo_vis_np, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        yolo_vis_np, label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
                    )

                final_detections.append({
                    "id":              i + 1,
                    "name":            det["name"],
                    "group":           det["group"],
                    "rel_depth":       obj_depth,
                    "distance_cm":     calibrated_cm,
                    "distance_cm_raw": calibrated_cm_raw,
                    "risk_level":      risk,
                    "bbox_xyxy":       [int(x1), int(y1), int(x2), int(y2)],
                })

            if publish_vis:
                last_vis_time = time.time()

            result = {
                "timestamp":         timestamp_str,
                "camera_id":         CAMERA_ID,
                "detections":        final_detections,
                "has_near_obstacle": has_near,
                "state_change":      state_change,
            }
            profiler.stage("result", time.perf_counter() - _t)

            # [최적화 3] numpy 배열 그대로 전달 → JPEG 인코딩은 publisher에서
            out_queue.put({
                "timestamp":    timestamp_str,
                "result":       result,
                "state_change": state_change,
                "yolo_vis_np":  yolo_vis_np,
                "depth_np":     last_depth if publish_vis else None,
            })
            profiler.process()

        except Exception as exc:
            print(f"[Worker {worker_id}] error: {repr(exc)}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # [최적화 1] in_queue maxsize=2
    in_queue   = Queue(maxsize=2)
    out_queue  = Queue()
    stop_event = Event()

    workers = [
        Process(target=worker_task, args=(i, in_queue, out_queue, stop_event))
        for i in range(NUM_WORKERS)
    ]
    for p in workers:
        p.start()

    rds = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT,
        db=REDIS_DB, password=REDIS_PASS,
    )
    rds.ping()

    # [최적화 2] publisher 전용 스레드 시작
    pub_stop   = threading.Event()
    pub_thread = threading.Thread(
        target=publisher_thread,
        args=(out_queue, rds, pub_stop),
        daemon=True,
    )
    pub_thread.start()

    print(f"System Running. Camera: {CAMERA_ID}")

    main_profiler = FrameProfiler("Main")
    last_id = "$"

    try:
        while True:
            resp = rds.xread({IN_STREAM: last_id}, count=1, block=10)

            if resp:
                for msg_id, data in resp[0][1]:
                    last_id = msg_id

                    if b"image" not in data:
                        continue

                    timestamp    = data.get(b"timestamp", b"0").decode("utf-8")
                    state_change = fetch_state_change_from_http()
                    item         = (msg_id, data[b"image"], timestamp, state_change)

                    # [최적화 1] put_nowait 즉시 드롭 — 오래된 프레임 누적 방지
                    try:
                        in_queue.put_nowait(item)
                        main_profiler.recv()
                    except queue.Full:
                        main_profiler.drop()

    except KeyboardInterrupt:
        print("Stopping...")
        main_profiler.force_report()
        stop_event.set()

        for _ in workers:
            in_queue.put(None)
        for p in workers:
            p.join()

        out_queue.put(None)
        pub_stop.set()
        pub_thread.join()


if __name__ == "__main__":
    main()