#!/usr/bin/env python3
"""GPU 실측 — 호스트 venv 와 ds-vision-gpu 컨테이너에서 같은 스크립트로 잰다 (Docker 계획 D5).

재는 것 (단계마다 새 프로세스를 띄워 서로 섞이지 않게 한다):
  context  torch.cuda.init() + 작은 할당(1 MiB) 뒤 기기 사용량 증가 · torch 예약량, 그 뒤 matmul(cuBLAS 적재) 뒤 값
  onnx     onnxruntime 제공자 목록 · CUDAExecutionProvider 로 ONNX 한 번 실행 (CLIP vision 이 있으면 그것, 없으면 1-op 모델)
  yoloe    YOLOE 적재 + 640×640 빈 영상 추론. 예열 뒤 20회 중앙값 지연, torch 예약 최대치, 기기 사용량 증가.
           「컨텍스트 몫 = 기기 증가 − torch 예약 최대치」 — model_registry.yaml gpu_context_mb 의 근거와 같은 셈법.

기기 사용량은 nvidia-smi --query-gpu=memory.used (기기 전체). WSL 은 프로세스별 값이 [N/A] 라 기기 전체를 쓴다.
그래서 재기 전에 GPU 에 다른 프로세스가 없어야 한다 (시작할 때 확인하고 바쁘면 멈춘다).

  python measure_gpu.py                      # 전부, JSON 한 줄씩 + 요약
  python measure_gpu.py --mode yoloe --runs 20 --models-dir /ds/models
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

MB = 1024 * 1024


def device_used_mb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
    return used, total


def settle_used(wait=0.6):
    time.sleep(wait)
    return device_used_mb()[0]


def where():
    return "container" if os.path.exists("/.dockerenv") else "host"


def mode_context(a):
    base = settle_used()
    import torch
    t0 = time.perf_counter()
    torch.cuda.init()
    x = torch.empty(MB, dtype=torch.uint8, device="cuda")  # 1 MiB
    torch.cuda.synchronize()
    init_s = time.perf_counter() - t0
    used1 = settle_used()
    r1 = torch.cuda.memory_reserved() / MB
    # cuBLAS 를 깨우는 작은 matmul
    m = torch.randn(512, 512, device="cuda")
    y = (m @ m).sum().item()
    torch.cuda.synchronize()
    used2 = settle_used()
    r2 = torch.cuda.memory_reserved() / MB
    del x
    return {
        "mode": "context", "where": where(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "cuda_available": torch.cuda.is_available(),
        "matmul_ok": bool(y == y), "init_s": round(init_s, 3),
        "base_used_mb": base,
        "after_alloc": {"device_delta_mb": used1 - base, "torch_reserved_mb": round(r1, 1),
                        "context_mb": round(used1 - base - r1, 1)},
        "after_matmul": {"device_delta_mb": used2 - base, "torch_reserved_mb": round(r2, 1),
                         "context_mb": round(used2 - base - r2, 1)},
    }


def _tiny_onnx(path):  # onnx 패키지 필요 (베이스 이미지에는 없음 — CLIP 이 없을 때만 씀)
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1024])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1024])
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "tiny", [x], [y])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)


def mode_onnx(a):
    import numpy as np
    import onnxruntime as ort
    base = settle_used()
    provs = ort.get_available_providers()
    # torch 를 먼저 import 하지 않은 상태에서도 nvidia-*-cu12 휠의 라이브러리를 찾게 한다
    preload = None
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
            preload = "ok"
        except Exception as e:  # noqa: BLE001
            preload = f"fail: {e}"
    path = os.path.join(a.models_dir, "clip-vit-base-patch32", "onnx", "vision_model.onnx")
    model = "clip-vit-base-patch32/vision_model.onnx"
    if not os.path.exists(path):
        path = "/tmp/tiny_relu.onnx"
        _tiny_onnx(path)
        model = "tiny relu (1-op)"
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        if "pixel" in i.name:
            shape = [1, 3, 224, 224]
        dt = np.int64 if "int64" in i.type else np.float32
        feeds[i.name] = np.zeros(shape, dt)
    sess.run(None, feeds)
    lat = []
    for _ in range(a.runs):
        t = time.perf_counter()
        sess.run(None, feeds)
        lat.append((time.perf_counter() - t) * 1000)
    used = settle_used()
    return {"mode": "onnx", "where": where(), "ort": ort.__version__, "available_providers": provs,
            "preload_dlls": preload, "model": model, "session_providers": sess.get_providers(),
            "cuda_used": sess.get_providers()[0] == "CUDAExecutionProvider",
            "median_ms": round(statistics.median(lat), 2), "device_delta_mb": used - base}


def mode_yoloe(a):
    import numpy as np
    base = settle_used()
    import torch
    from ultralytics import YOLOE
    w = os.path.join(a.models_dir, a.yoloe)
    t0 = time.perf_counter()
    model = YOLOE(w)
    img = np.zeros((640, 640, 3), np.uint8)
    model.predict(img, imgsz=640, device=0, verbose=False)  # 첫 추론(적재 · 예열)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    for _ in range(3):
        model.predict(img, imgsz=640, device=0, verbose=False)
    lat = []
    peak_used = 0
    for _ in range(a.runs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.predict(img, imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    peak_used = max(peak_used, device_used_mb()[0])
    used = settle_used()
    peak_used = max(peak_used, used)
    rmax = torch.cuda.max_memory_reserved() / MB
    return {"mode": "yoloe", "where": where(), "weight": a.yoloe, "runs": a.runs,
            "load_plus_first_s": round(load_s, 2),
            "median_ms": round(statistics.median(lat), 2),
            "p90_ms": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2),
            "torch_reserved_peak_mb": round(rmax, 1),
            "torch_reserved_now_mb": round(torch.cuda.memory_reserved() / MB, 1),
            "device_delta_mb": peak_used - base,
            "context_mb": round(peak_used - base - rmax, 1)}


MODES = {"context": mode_context, "onnx": mode_onnx, "yoloe": mode_yoloe}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["all", *MODES], default="all")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--models-dir", default=os.path.join(os.environ.get("DS_HOME", "."), "models"))
    p.add_argument("--yoloe", default="yoloe-11l-seg-pf.pt")
    p.add_argument("--repeat", type=int, default=1, help="all 일 때 단계마다 몇 번 되풀이할지")
    p.add_argument("--idle-mb", type=int, default=300, help="시작 때 기기 사용량이 이보다 크면 바쁜 것으로 보고 멈춤")
    a = p.parse_args()
    if a.mode != "all":
        print(json.dumps(MODES[a.mode](a), ensure_ascii=False))
        return
    used, total = device_used_mb()
    if used > a.idle_mb:
        sys.exit(f"GPU 가 바쁨: {used}/{total} MiB 사용 중 — 다른 프로세스가 끝난 뒤 다시 재세요")
    for mode in MODES:
        for _ in range(a.repeat):
            cmd = [sys.executable, os.path.abspath(__file__), "--mode", mode, "--runs", str(a.runs),
                   "--models-dir", a.models_dir, "--yoloe", a.yoloe]
            r = subprocess.run(cmd, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [""])[-1]
            if r.returncode != 0:
                print(json.dumps({"mode": mode, "error": r.stderr.strip()[-800:]}, ensure_ascii=False))
            else:
                print(line)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
