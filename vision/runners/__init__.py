"""vision/runners — 모델 실행기(runner) 꾸러미와 적재기 (M2, 2026-11-27).

모델 하나를 레지스트리(vision/model_registry.yaml)에 **한 번** 등록하면 그 능력(provides)이 맞는 역할 슬롯마다
선택지로 뜨고(ds_resources/slots.py), 고르면 비전(vision_infer.py)이 이 적재기로 실행기를 만든다.
이 파일은 torch 를 import 하지 않는다 — 실행기 모듈은 **고를 때만** 읽는다(lazy).

레지스트리 등록 (models.<키>)
    models:
      my_model:
        runner: my_model:MyRunner          # vision/runners/my_model.py 의 MyRunner (점이 있으면 절대 모듈 이름)
        kind: detect                       # detect | segment | depth | track | vlm | embed (없으면 provides 로 짐작)
        provides: [detect, open_vocab]     # 능력 — 슬롯 needs 와 맞춘다
        runs_on: [gpu]                     # gpu | cpu_x86 | cpu_arm64
        size: small                        # 기기(파이) 슬롯에 뜨려면 (선택)
        gpu_mb: 800                        # 배치 어림 (MB)
        p90_ms: {s: 40, m: 60, l: 90, xl: 150}
        license: Apache-2.0                # 안내만 — 라이선스로 막지 않는다
        files: [models/my_model.pt]        # 있어야 쓸 수 있는 파일 (가중치 유무 판정)
        source: [...]                      # tools/fetch_models.py 가 받는 곳 (기존 규칙 그대로)
    매개변수는 vision/vision_config.yaml models.<키> 에 둔다 — 그 사전이 실행기의 cfg 로 간다
    (없으면 레지스트리 models.<키>.config, 그것도 없으면 빈 사전).

실행기 프로토콜 (README.md 에 같은 글)
    class MyRunner:
        name = "my_model"                  # 레지스트리 키와 같게
        kind = "detect"                    # detect | segment | depth | track | vlm | embed

        def __init__(self, cfg: dict, device: str): ...
            # device: "cuda:0" · "cpu". 무거운 import(torch · ultralytics …)는 여기서 한다

        # detect · segment
        def __call__(self, bgr: np.ndarray) -> tuple[dict, list[dict]]:
            # bgr: HxWx3 uint8 (OpenCV BGR). 돌려주는 것 (info, dets)
            #   info  {"infer_ms": float, ...} — 결과 기록에 그대로 붙는다 (JSON 으로 쓸 수 있는 값만)
            #   dets  [{"cls": int, "name": str, "conf": float, "xyxy": [x1, y1, x2, y2] (원본 px),
            #           선택: "_poly": np.ndarray (N,2) 분할 외곽 (밑줄 칸은 저장 전에 빠진다),
            #                 "polygon": [[x, y], ...] (저장할 외곽), "mask", "track_id": int}]

        # depth — UniDepthRunner · MogeAerialRunner 와 같은 약속
        def __call__(self, bgr: np.ndarray, fov_x: float | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
            # fov_x: 가로 화각(도). 주면 그 화각으로, 없으면 모델이 추정
            # 돌려주는 것 (info, depth, K)
            #   info   {"infer_ms": float, "checkpoint": str, ...}
            #   depth  HxW float32, **미터**, 원본 크기로 되돌린 것. 0 또는 비유한값 = 무효 픽셀
            #   K      3x3 float 내부 행렬 (원본 px 기준 — 줄여 돌렸으면 되돌린 것)

        # track · vlm · embed — 지금 비전 루프는 부르지 않는다(오케스트레이터 계획에만 실린다).
        #   track:  update(dets: list[dict], ts: float, bgr: np.ndarray | None = None) -> list[dict]  (track_id 를 붙인다)
        #   vlm:    __call__(bgr, prompt: str) -> (info, text: str)
        #   embed:  __call__(bgr, boxes: list[list[float]]) -> (info, np.ndarray (N, D) L2 정규화)

        def close(self): ...               # 선택 — GPU 메모리를 놓는다
"""
from __future__ import annotations

import importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parent / "model_registry.yaml"
KINDS = ("detect", "segment", "depth", "track", "vlm", "embed")
#: 비전 루프가 부르는 종류 — 검출 꼴(detect · segment)과 깊이 꼴(depth)
DETECT_KINDS = ("detect", "segment")
DEPTH_KINDS = ("depth",)


class RunnerError(RuntimeError):
    """실행기를 못 만든다 — 메시지는 사람이 읽는 한 줄 (모듈 · 클래스 · 까닭)"""


def is_runner_ref(runner):
    """「모듈:클래스」 꼴인가 (그 밖은 옛 선택지 키 — RUNNERS 가 맡는다)"""
    return isinstance(runner, str) and ":" in runner


def runner_kind(spec):
    """레지스트리 항목 → 종류. kind 가 없으면 provides 로 짐작한다 (ds_resources/slots.py 와 같은 규칙)"""
    k = (spec or {}).get("kind")
    if k:
        return str(k)
    caps = set((spec or {}).get("provides") or ())
    if caps & {"depth_metric", "depth_relative", "depth"}:
        return "depth"
    for k in ("segment", "detect", "track", "vlm", "embed"):
        if k in caps:
            return k
    return None


def runner_models(reg=None):
    """{키: 레지스트리 항목} — runner 가 「모듈:클래스」 인 모델만. reg 가 없으면 레지스트리 파일을 읽는다"""
    if reg is None:
        try:
            import yaml
            with open(REGISTRY, encoding="utf-8") as f:
                reg = yaml.safe_load(f) or {}
        except (ImportError, OSError):
            return {}
    options = set(reg.get("options") or {})
    return {str(k): dict(m) for k, m in (reg.get("models") or {}).items()
            if isinstance(m, dict) and is_runner_ref(m.get("runner")) and k not in options}


def runner_class(spec):
    """레지스트리 항목의 runner → 클래스 (모듈을 이때 import 한다)"""
    ref = (spec or {}).get("runner")
    if not is_runner_ref(ref):
        raise RunnerError("runner 는 「모듈:클래스」 꼴이어야 한다 (받은 것: %r)" % (ref,))
    mod_name, _, cls_name = ref.partition(":")
    full = mod_name if "." in mod_name else "%s.%s" % (__name__, mod_name)
    try:
        mod = importlib.import_module(full)
    except ImportError as exc:
        raise RunnerError("실행기 모듈 %s 를 못 읽음: %s" % (full, exc)) from exc
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise RunnerError("%s 에 %s 가 없다" % (full, cls_name))
    return cls


def load_runner(key, spec, device, cfg=None):
    """레지스트리 키 · 항목 · 장치 → 실행기 인스턴스. cfg 가 없으면 항목의 config (없으면 빈 사전)"""
    cls = runner_class(spec)
    conf = dict(cfg if cfg is not None else ((spec or {}).get("config") or {}))
    inst = cls(conf, device)
    kind = getattr(inst, "kind", None) or runner_kind(spec)
    if kind not in KINDS:
        raise RunnerError("%s 실행기의 kind %r 를 모른다 (가능: %s)" % (key, kind, ", ".join(KINDS)))
    if not callable(inst) and kind != "track":
        raise RunnerError("%s 실행기는 부를 수 있어야 한다 (__call__)" % key)
    if not getattr(inst, "name", None):
        try:
            inst.name = key
        except AttributeError:
            pass
    return inst
