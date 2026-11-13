"""위험물 후보 검증 — **라벨 없이, 학습 없이.**

implements: AI-S-03 (근거 충분도는 confidence 와 따로), AI-S-04 (확정하지 못한 것은
가설로), AI-E-04 (근거가 부족할 때만 보조 기능을 돌린다), AI-C-13 (자원 최소화)
관련: AI-L-01 (학습 후보), AI-C-16 (가중치는 내부 경로), AI-C-19

## 무엇을 하는가

탐지기가 낸 위험물 후보(나무 상자·부탄)의 crop 을 **기준 사진에서 만든 프로토타입**과
견준다. 경사하강이 없고 라벨이 없다 — 기준 crop 몇 장을 임베딩해 두고 코사인 유사도를
재는 **one-shot 검색**이다.

## 왜 이것이 비지도인가, 그리고 무엇을 주지 못하는가

운영 영상에는 이 두 물체의 정답이 사실상 없다(하천 부탄 정답 0개, 실내 1개는 오라벨).
그래서 지도학습이 설 자리가 없다. 대신 **라벨 없이 쓸 수 있는 불변량**에 기댄다 —
같은 물건은 자기지도 표현 공간에서 서로 가깝다는 것 하나다.

**주지 못하는 것**: "이 군집이 목함지뢰다" 는 사람이 정한다. 여기서 나오는 것은
"기준과 얼마나 닮았는가" 뿐이고, 그래서 **검출을 지우지 않는다**(AI-S-04, AI-C-19).

## 다중 모델인 이유

한 임베딩이 틀리는 자리는 모델마다 다르다. 서로 다른 학습 자료를 본 인코더 둘 이상이
같은 방향을 말할 때만 근거가 선다. 모델 목록은 **설정 데이터**이고 이 파일은 어떤
벤더도 알지 못한다 — ONNX 세션 하나와 전처리 셋이면 무엇이든 꽂힌다(AI-C-04).

## 실시간

전 화면이 아니라 **위험물 후보 crop 에만** 돌고, 한 프레임의 후보를 **한 번에 묶어**
세션에 넣는다. 후보가 없으면 비용이 0 이다. 측정값은 `verify_ms` 로 나간다(AI-O-01).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

#: 내려받지 않는다 — 없으면 그 모델만 빠진다 (AI-C-16, AI-C-05)
_MISSING = "가중치가 없습니다"


class _Encoder:
    """ONNX 비전 인코더 하나. 전처리는 선언으로 받는다 — 모델마다 다르다."""

    def __init__(self, spec):
        import onnxruntime as ort
        path = Path(spec["path"]).expanduser()
        if not path.exists():
            raise FileNotFoundError("%s: %s" % (_MISSING, path))
        providers = spec.get("providers") or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        avail = set(ort.get_available_providers())
        # **GPU 가 먼저다**(2026-11-13). onnxruntime-gpu 는 CUDA 판이 torch 와 맞아야 선다 —
        # torch cu12.6 에는 onnxruntime-gpu 1.24.x(CUDA 12), 1.25 이후는 CUDA 13 을 요구한다.
        # CUDA 라이브러리는 torch 가 깔아 둔 nvidia-* 패키지에서 미리 올린다(preload_dlls).
        # CPU 로 떨어지면 실 8 개 · 묶음으로 crop 당 ~11 ms 다 — **어디서 돌았는지는 로그로
        # 나간다**(`provider`, AI-O-01).
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(spec.get("threads") or 8)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so,
                                         providers=[p for p in providers if p in avail])
        self.name = str(spec.get("name") or path.parent.parent.name)
        self.size = int(spec.get("size") or 224)
        self.mean = np.array(spec.get("mean") or [0.0, 0.0, 0.0], np.float32).reshape(3, 1, 1)
        self.std = np.array(spec.get("std") or [1.0, 1.0, 1.0], np.float32).reshape(3, 1, 1)
        self.input = self.sess.get_inputs()[0].name
        # fp16 모델은 입력도 fp16 이다 — 모델이 원하는 형으로 넘긴다
        self.in_dtype = np.float16 if "float16" in self.sess.get_inputs()[0].type else np.float32
        self.output = self.sess.get_outputs()[0].name
        self.provider = self.sess.get_providers()[0]

    def __call__(self, crops):
        """BGR crop 목록 → (N, D) 정규화 임베딩."""
        batch = np.empty((len(crops), 3, self.size, self.size), np.float32)
        for i, c in enumerate(crops):
            rgb = cv2.cvtColor(cv2.resize(c, (self.size, self.size),
                                          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch[i] = (x - self.mean) / self.std
        v = self.sess.run([self.output], {self.input: batch.astype(self.in_dtype)})[0].astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-8)


class HazardVerifier:
    """후보 crop 과 프로토타입의 거리. **판정이 아니라 근거를 낸다.**

    점수는 `pos - neg` 다 — 그 클래스 기준과의 최대 유사도에서, **음성 기준**(헷갈리는
    물건, 예: 음료캔)과의 최대 유사도를 뺀다. 유사도 자체를 쓰면 "무엇과도 비슷한"
    흐릿한 crop 이 높게 나온다. 차이를 쓰면 **가르는 힘**만 남는다.
    """

    def __init__(self, spec, log=print):
        self.classes = tuple(spec.get("classes") or ())
        # **이름이 아직 없는 후보도 잰다**(`candidates`, 예: unconfirmed). 위험물 이름을
        # 받은 것만 재면 "나무라고 불리지 않은 나무 상자" 는 아예 비교되지 않는다.
        # 후보마다 **위험물 클래스 전부**에 대해 잰다(`verify_by`) — 점수로 견주려면
        # 같은 단서를 모든 클래스에 대해 같은 방식으로 재야 한다.
        self.candidates = tuple(spec.get("candidates") or ())
        self.encoders, self.missing = [], []
        for m in spec.get("models") or []:
            try:
                self.encoders.append(_Encoder(m))
            except Exception as exc:                 # 한 모델이 없어도 나머지로 돈다
                self.missing.append((str(m.get("name") or m.get("path")), str(exc)))
        # **선언이 먼저, 동봉이 다음.** 배포 기계는 자기 자산 경로를 적고, 그 경로가
        # 없는 기계(막 clone 한 곳)는 코드 옆에 동봉된 39 KB 로 물러선다 — 그래야
        # `match` 가 "기계마다 되기도 하고 안 되기도 하는" 단서가 되지 않는다.
        # **어느 쪽을 썼는지 기록에 남긴다**: 조용히 바뀌면 숫자가 어느 프로토타입의
        # 것인지 알 수 없다.
        bundled = Path(__file__).resolve().parent / "assets/butane_ref/prototypes.npz"
        declared = Path(str(spec.get("prototypes") or "")).expanduser()
        bank = declared if declared.exists() else bundled
        self.bank_path, self.bank_bundled = bank, (bank == bundled)
        self.bank = {}
        if bank.exists():
            z = np.load(bank, allow_pickle=False)
            self.bank = {k: z[k] for k in z.files}
        else:
            self.missing.append((str(declared or bundled), _MISSING))
        self.min_models = int(spec.get("min_models") or 1)
        # **시간을 되사는 두 가지.** (a) 앞 장과 거의 같은 자리의 같은 클래스는 다시
        # 임베딩하지 않는다 — 고정 카메라에서 같은 물체가 장마다 다시 계산된다.
        # (b) 한 장에서 보는 후보 수에 상한을 둔다. 둘 다 사용자 설계 8단계
        # ("기존 특징을 캐싱", "후보 발생 시에만")를 그대로 옮긴 것이다.
        self.cache_iou = float(spec.get("cache_iou") or 0.7)
        self.max_per_frame = int(spec.get("max_per_frame") or 6)
        self._last = []                              # [(cls, xyxy, verify)]
        self.structure = bool(spec.get("structure", True))
        # **의존하지 않는 것이 의존하는 것 때문에 죽지 않게 한다.**
        # `structure` 는 기준 사진에서 손으로 읽은 규칙이라 프로토타입이 **필요 없다**.
        # 그런데 준비 판정을 하나로 두면 npz 가 없는 기계에서 구조 단서까지 같이 꺼진다 —
        # 그 기계는 쓸 수 있는 근거를 가지고도 못 쓰게 된다(AI-C-05 의 격리가 뜻하는 것이
        # 바로 이것이다). 그래서 **배운 쪽과 안 배운 쪽을 따로 센다.**
        self.learned = (bool(self.encoders) and bool(self.bank)
                        and len(self.encoders) >= self.min_models)
        self.ready = self.learned or self.structure
        if log:
            parts = []
            if self.learned:
                parts.append("match(모델 %d · 프로토타입 %d묶음%s · %s)"
                             % (len(self.encoders), len(self.bank),
                                " · **동봉본**" if self.bank_bundled else "",
                                ",".join(sorted({e.provider.replace("ExecutionProvider", "")
                                                 for e in self.encoders}))))
            if self.structure:
                parts.append("structure(프로토타입 불필요)")
            log("  verify %s%s%s"
                % (" + ".join(parts) if parts else "**비활성**",
                   "" if self.learned else " · **match 없음**",
                   (" · 빠짐 %d" % len(self.missing)) if self.missing else ""))

    @staticmethod
    def _structure(crop, cls):
        """**모양 단서 — 그 물건의 형태인가.**

        나무 상자는 **직육면체**다. 긴 축에 수직으로 평균한 밝기 프로파일이 평평한
        면에서는 평탄하다(`flat` = 이웃 차이가 0.02 미만인 비율). 실측:

            진짜 나무 상자 **0.998**  ·  실내 오검출 0.684  ·  하천 오검출 0.368
            부탄 통 0.576 / 0.669     ·  음료캔 0.640

        **부탄의 '원통' 은 아직 못 잰다.** 둥근 음영을 재 보려고 두 가지를 시도했고
        둘 다 갈리지 않았다 — 2차 미분 평균은 사실 **모서리 활동량**이었고(합성 원통
        0.098 < 합성 상자 0.631 로 거꾸로 나왔다), 돔 상관 × (1−flat) 은 **하천
        오검출이 0.230 으로 가장 높았다**(부탄 0.179/0.116, 합성 원통 0.120).
        그래서 부탄은 **구성 서명**(뚜껑 → 어깨 → 몸통)을 그대로 쓴다. 모양은
        아니지만 실제로 갈리고, 갈리지 않는 수를 내는 것보다 낫다.

        앞서 버린 것들도 적어 둔다: 직교 모서리 집중도(하늘 0.515 > 진짜 상자 0.440),
        경계/속 기울기비(실내 오검출 0.540 로 최고), 허프 직선(진짜 상자 0 개).
        """
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 16:
            return None
        if cls == "wooden_box_mine":
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if g.shape[1] < g.shape[0]:
                g = g.T                                  # 긴 축을 가로로
            prof = cv2.GaussianBlur(g.mean(axis=0).reshape(1, -1), (5, 1), 0).ravel()
            span = max(1e-6, float(prof.max() - prof.min()))
            p = (prof - prof.min()) / span
            if len(p) < 12:
                return None
            return round(float((np.abs(np.diff(p)) < 0.02).mean()), 4)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (sat >= 80) & (val >= 60)
        chrome = (sat < 60) & (val >= 120)
        H = crop.shape[0]
        if cls == "butane_canister":
            # 긴 축을 세로로 본다 — 누운 통은 가로가 길다
            if crop.shape[1] > H:
                hsv = cv2.rotate(hsv, cv2.ROTATE_90_CLOCKWISE)
                h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
                lit = (sat >= 80) & (val >= 60)
                chrome = (sat < 60) & (val >= 120)
                H = hsv.shape[0]
            k = max(1, H // 8)
            # **양 끝을 다 본다.** 뚜껑이 위에 있다고 가정하면 뒤집힌 통이 0.71 로
            # 떨어진다(시험이 잡았다). 강에 뜬 통은 어느 쪽이 위인지 정해져 있지 않다.
            best = 0.0
            for flip in (False, True):
                hh, ll, cc = (h, lit, chrome)
                if flip:
                    hh, ll, cc = hh[::-1], ll[::-1], cc[::-1]
                cap_lit = float(ll[:k].mean())
                cap_hue = hh[:k][ll[:k]]
                # 뚜껑은 **단일색**이다: 색상 분산이 작아야 한다
                single = 1.0 - min(1.0, float(np.std(cap_hue)) / 30.0) if cap_hue.size else 0.0
                shoulder = float(cc[k:2 * k].mean()) if H > 2 * k else 0.0
                body = float(ll[2 * k:].mean()) if H > 2 * k else 0.0
                best = max(best, min(cap_lit, single) * 0.5 + shoulder * 0.25 + body * 0.25)
            return round(float(best), 4)
        return None

    def _protos(self, enc_name, cls):
        return self.bank.get("%s/%s" % (enc_name, cls))

    def __call__(self, bgr, dets):
        """후보에 `verify_by`(위험물 클래스마다의 근거)와 `verify`(자기 이름의 근거)를
        붙인다. 후보가 없으면 비용 0."""
        if not self.ready:
            return 0.0
        want = [d for d in dets if d["name"] in self.classes or d["name"] in self.candidates]
        if not want:
            return 0.0
        t0 = time.perf_counter()

        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            i = (x2 - x1) * (y2 - y1)
            return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)

        def own(d):
            d["verify"] = (d.get("verify_by") or {}).get(d["name"])

        # 확신이 높은 것부터 본다 — 상한에 걸려 잘릴 때 **약한 것이 잘리도록**
        want.sort(key=lambda d: -d["conf"])
        crops, keep, reused = [], [], 0
        H, W = bgr.shape[:2]
        for d in want:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            # 캐시는 **자리로만** 찾는다 — 근거는 클래스마다 다 들고 있으므로 이름이
            # 바뀌어도(가설 → 위험물) 다시 잴 필요가 없다
            hit = next((v for box, v in self._last
                        if iou(box, (x1, y1, x2, y2)) >= self.cache_iou), None)
            if hit is not None:
                d["verify_by"] = {c: dict(v, cached=True) for c, v in hit.items()}
                own(d)
                reused += 1
                continue
            c = bgr[y1:y2, x1:x2]
            if c.size == 0 or min(c.shape[:2]) < 8:
                d["verify_by"] = None                # 너무 작아 **재지 않는다** (0 이 아니다)
                own(d)
                continue
            if len(crops) >= self.max_per_frame:
                d["verify_by"] = None                # 이 장의 예산을 넘었다 — 다음 장에서 본다
                own(d)
                continue
            crops.append(c)
            keep.append(d)
        if not crops:
            self.last_reused = reused
            return round((time.perf_counter() - t0) * 1000, 1)
        per = {}                                     # (id(d), cls) → {모델: 근거}
        # 프로토타입이 없으면 인코더를 **돌리지 않는다** — 돌려도 견줄 것이 없고
        # 그 비용은 그대로 든다
        for enc in (self.encoders if self.learned else ()):
            emb = enc(crops)                         # 한 프레임의 후보를 한 번에
            neg = self._protos(enc.name, "_negative")
            for i, d in enumerate(keep):
                n = float((emb[i] @ neg.T).max()) if neg is not None and len(neg) else 0.0
                for cls in self.classes:
                    pos = self._protos(enc.name, cls)
                    if pos is None:
                        continue
                    p = float((emb[i] @ pos.T).max())
                    per.setdefault((id(d), cls), {})[enc.name] = {
                        "pos": round(p, 4), "neg": round(n, 4), "margin": round(p - n, 4)}
        for i, d in enumerate(keep):
            by = {}
            for cls in self.classes:
                rows = per.get((id(d), cls)) or {}
                if self.structure:
                    st = self._structure(crops[i], cls)
                    if st is not None:
                        # 학습하지 않은 신호다. margin 과 자릿수를 맞춰 넣되 **출처를 남긴다**
                        rows["structure"] = {"pos": st, "neg": 0.0, "margin": round(st - 0.15, 4),
                                             "learned": False}
                if not rows:
                    continue
                margins = [r["margin"] for r in rows.values()]
                by[cls] = {"margin": round(float(np.mean(margins)), 4),
                           "agree": int(sum(m > 0 for m in margins)),
                           "models": len(margins), "per_model": rows, "cached": False}
            d["verify_by"] = by or None
            own(d)
        self._last = [(tuple(int(v) for v in d["xyxy"]), d["verify_by"])
                      for d in want if d.get("verify_by")]
        ms = round((time.perf_counter() - t0) * 1000, 1)
        self.last_reused = reused
        return ms


def build_bank(ref_dir, out, models, log=print):
    """기준 crop 폴더 → 프로토타입 묶음.

    폴더 구조: `<ref_dir>/<클래스>/*.png`. `_negative/` 는 **헷갈리는 물건**이다
    (음료캔 등). 운영 영상은 **넣지 않는다** — 그것은 시험 자료다.
    """
    ref_dir, out = Path(ref_dir), Path(out)
    encs = []
    for m in models:
        try:
            encs.append(_Encoder(m))
        except Exception as exc:
            log("  건너뜀 %s: %s" % (m.get("name") or m.get("path"), exc))
    bank = {}
    for cls_dir in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
        crops = [c for c in (cv2.imread(str(f)) for f in files) if c is not None]
        if not crops:
            continue
        for enc in encs:
            bank["%s/%s" % (enc.name, cls_dir.name)] = enc(crops)
        log("  %-18s %d장" % (cls_dir.name, len(crops)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **bank)
    log("  프로토타입: %s (%d묶음)" % (out, len(bank)))
    return bank
