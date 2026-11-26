#!/usr/bin/env python3
"""모델 하나가 **실제로 적재되는지** 본다 — 가중치를 바꾼 뒤의 자동 시험 (2026-11-26).

    venv/bin/python tools/model_check.py --models yoloe [--json]

왜 따로 있나: 파일이 제자리에 있고 sha256 이 맞아도(그건 tools/fetch_models.py --check 가 본다) 그 파일이
그 자리에서 **열리는지**는 다른 문제다. 라이브러리 판이 다르거나, 받다 만 파일이거나, 다른 모델의 가중치를
같은 이름으로 넣었을 수 있다. 런처가 「모델 바꾸기」로 파일을 넣은 직후 이것을 돌려 사람에게 결과를 알린다.

하는 일: 레지스트리가 그 선택지(또는 모델 키)에 적어 둔 파일마다 확장자에 맞는 길로 한 번 열어 본다.
  .pt    ultralytics (YOLO · YOLOE) — 가중치만 읽고 추론은 하지 않는다
  .onnx  onnxruntime 세션 하나 (CPU 로)
  .json  json.load (토크나이저 사전 등)
그 밖(폴더 · git)은 있는지만 본다. 아무것도 받지 않고 고치지 않는다 — 읽기만 한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def load_pt(path):
    from ultralytics import YOLO
    YOLO(str(path))                                  # 적재만 (추론 없음)


def load_onnx(path):
    import onnxruntime as ort
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_json(path):
    with open(path, encoding="utf-8") as f:
        json.load(f)


LOADERS = {".pt": load_pt, ".onnx": load_onnx, ".json": load_json}


def check_file(path):
    """(ok, 걸린 ms, 까닭) — 열어 보고 돌려준다"""
    p = Path(path)
    if not p.exists():
        return False, 0, "파일이 없습니다"
    fn = LOADERS.get(p.suffix.lower())
    if fn is None:
        return True, 0, "확인 없음 (여는 길을 모르는 형식 — 있는 것만 봄)"
    t0 = time.time()
    try:
        fn(p)
    except Exception as e:  # noqa: BLE001 — 어떤 까닭이든 사람에게 그대로 보여 준다
        return False, int((time.time() - t0) * 1000), "%s: %s" % (type(e).__name__, str(e)[:200])
    return True, int((time.time() - t0) * 1000), ""


def check(models, dest=None):
    from tools import fetch_models as fm
    out = {"ok": True, "files": []}
    for m in models:
        try:
            keys, _trained = fm.needed([m])
            items = fm.items_for(keys, dest=dest)
        except SystemExit as e:
            out["ok"] = False
            out["files"].append({"model": m, "file": "-", "ok": False, "why": str(e)})
            continue
        for it in items:
            if it["kind"] in ("git", "hf_dir", "nosource") or not it.get("dest"):
                continue
            ok, ms, why = check_file(it["dest"])
            out["ok"] = out["ok"] and ok
            out["files"].append({"model": it["model"], "file": Path(it["dest"]).name, "ok": ok, "ms": ms,
                                 "why": why})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="모델이 실제로 적재되는지 본다 (읽기만)")
    ap.add_argument("--models", nargs="+", required=True, metavar="이름", help="선택지(yoloe …) 또는 모델 키")
    ap.add_argument("--dest", help="models/ 대신 볼 곳")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = check(a.models, a.dest)
    if a.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        for f in res["files"]:
            print("%-5s %-16s %-40s %s" % ("열림" if f["ok"] else "못 엶", f["model"], f["file"],
                                           f.get("why") or ("%d ms" % f.get("ms", 0))))
        print("→ %s" % ("모두 열립니다" if res["ok"] else "못 여는 파일이 있습니다"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
