"""비전 프로세스 안의 작업 나눔 — 같은 장 · 같은 일은 한 번 (work/1, 2026-11-27).

ds_resources/workgraph.py 가 계획에서 「같은 소스 · 같은 모델 · 같은 입력 크기의 일은 노드 하나」로 묶으면,
이 모듈이 비전 프로세스 안에서 그것을 지킨다. 한 장(프레임)마다 FrameWork 하나를 열고 결과를 열쇠로 담아
두었다가 같은 일을 바라는 다른 기능에 그대로 준다.

  depth   지정 물체 탐지(targeted)의 거리 근거(depth_track.DepthTracker)와 독립 depth 모델(unidepth 선택지 ·
          da3_metric_l 실행기)이 **같은 가중치**면 지도 하나를 나눠 쓴다. 정체 = (종류, 가중치 경로) —
          거리 근거는 따로 모델을 적재하지 않고(SharedDepthProvider) 독립 모델이 잰 지도를 받는다.
          먼저 부른 쪽이 재고 뒤에 부른 쪽은 받는다. 독립 모델이 그 장에 돌 것이면 거리 근거는 주기(5장)를
          기다리지 않고 매 장 받는다(공짜다). 설정(resolution_level · fp16 · 화각 힌트)은 독립 모델 것을 쓴다.
  검출    탐지 모델은 한 프로세스에서 모델마다 한 번 돈다(runners 사전). 그 검출을 이 장의 FrameWork 에
          담아 두면 물체 추적(object_track)이 다시 돌리지 않고 받는다.

나눈 횟수는 stats 에 쌓여 vision_telemetry 의 work 칸 · 로그로 나간다 — 「한 장에 한 번」의 근거다.
"""
from __future__ import annotations

import time
from pathlib import Path

METRIC = "metric"


def norm_path(p):
    """가중치 경로를 견줄 수 있게 — 상대(models/…) · ~ 는 절대로, 허브 이름은 그대로"""
    if p is None:
        return None
    s = str(p)
    if s.startswith(("models/", "external/", "~", "/")):
        try:
            from ds_resources import settings
            return str(Path(settings.resolve(s)).resolve())
        except Exception:                                   # noqa: BLE001 — 못 풀면 글자 그대로
            return str(Path(s).expanduser())
    return s


def standalone_identity(m, spec_runner, mcfg):
    """독립 depth 모델의 정체 — ("unidepth", 경로) · ("depth_anything_3", 경로). 나눌 수 없는 것은 None"""
    mcfg = mcfg or {}
    if m == "unidepth":
        if mcfg.get("infer_long_side"):                     # 줄여 잰 지도는 다른 일이다
            return None
        return ("unidepth", norm_path(mcfg.get("model_name")))
    ref = str((spec_runner or {}).get("runner") or "")
    if ref.startswith("da3_metric:"):
        path = mcfg.get("model_path") or mcfg.get("model_id") or mcfg.get("model_name") or "models/da3-metric-large"
        return ("depth_anything_3", norm_path(path))
    return None


def tracker_identity(dt_module, dspec):
    """거리 근거(DepthTracker spec)의 정체 — depth_track._kind 로 종류를 가린다"""
    kind = dt_module._kind(dspec)
    if kind in ("unidepth", "depth_anything_3"):
        return (kind, norm_path(dspec.get("model_id")))
    return None


class FrameWork:
    """한 장의 결과 담기 — 열쇠 → 값. 장마다 새로 연다"""

    def __init__(self, device, n, fov=None, planned=()):
        self.device, self.n, self.fov = device, n, fov
        self.planned = set(planned)                         # 이 장에 돌 독립 depth 정체
        self.items = {}


class WorkShare:
    """프로세스 하나의 나눔 장부"""

    def __init__(self, log=print):
        self.log = log
        self.depth = {}                                     # 정체 → (모델 키, 실행기)
        self.stats = {}                                     # (기기, 열쇠) → {runs, reused, consumers}
        self.frame = None

    # -- 장
    def begin(self, device, n, fov=None, depth_models=()):
        planned = {ident for ident, (m, _r) in self.depth.items() if m in set(depth_models)}
        self.frame = FrameWork(device, n, fov, planned)
        return self.frame

    def _stat(self, key, consumer, reused):
        st = self.stats.setdefault((self.frame.device if self.frame else None, key),
                                   {"runs": 0, "reused": 0, "consumers": []})
        st["reused" if reused else "runs"] += 1
        if consumer not in st["consumers"]:
            st["consumers"].append(consumer)

    def put(self, key, value, consumer):
        if self.frame is not None:
            self.frame.items[key] = value
            self._stat(key, consumer, False)

    def get(self, key, consumer):
        if self.frame is None or key not in self.frame.items:
            return None
        self._stat(key, consumer, True)
        return self.frame.items[key]

    # -- depth
    def register_depth(self, m, runner, ident):
        if ident is not None:
            self.depth[ident] = (m, runner)

    def model_of(self, ident):
        return (self.depth.get(ident) or (None, None))[0]

    def will_run(self, ident):
        return self.frame is not None and ident in self.frame.planned

    def depth_for(self, ident, bgr, consumer):
        """(info, depth, K) — 이 장에 이미 잰 것이 있으면 그것, 없으면 독립 모델로 재서 담는다"""
        key = ("depth", ident)
        hit = self.get(key, consumer)
        if hit is not None:
            return hit
        _m, runner = self.depth[ident]
        t0 = time.perf_counter()
        info, depth, K = runner(bgr, fov_x=self.frame.fov if self.frame else None)
        info = dict(info, infer_ms=info.get("infer_ms", round((time.perf_counter() - t0) * 1000, 1)))
        val = (info, depth, K)
        self.put(key, val, consumer)
        return val

    def summary(self, reset=True):
        """telemetry 칸 — {기기: {열쇠: {runs, reused, consumers}}}"""
        out = {}
        for (dv, key), st in sorted(self.stats.items(), key=lambda kv: str(kv[0])):
            name = ":".join(str(x) for x in key[:1]) + ":" + (self.model_of(key[1]) if key[0] == "depth" else str(key[1]))
            out.setdefault(str(dv), {})[name] = dict(st)
        if reset:
            self.stats = {}
        return out


class SharedDepthProvider:
    """DepthTracker 의 provider 자리에 끼운다 — 모델을 따로 적재하지 않고 WorkShare 에서 지도를 받는다"""

    space = METRIC

    def __init__(self, share, ident, consumer="targeted"):
        self.share, self.ident, self.consumer = share, ident, consumer
        self.name = "shared:%s" % (share.model_of(ident) or ident[0])
        self.intrinsics = None

    def fresh(self):
        """이 장에 독립 모델이 돌 것인가 — 그러면 거리 근거는 주기를 기다리지 않고 받는다"""
        return self.share.will_run(self.ident)

    def __call__(self, bgr):
        info, depth, K = self.share.depth_for(self.ident, bgr, self.consumer)
        try:
            if K is not None and float(K[0][0]) > 0:
                self.intrinsics = (float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]))
        except (TypeError, IndexError, ValueError):
            pass
        return depth
