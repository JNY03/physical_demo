"""원하는 상태 (ds_desired/1) — 웹 런처가 쓰고 감독자가 맞춘다 (core, 2026-11-25).

Docker 계획 §5. 런처는 프로세스를 직접 띄우지 않고 이 문서만 쓴다. 감독자(ds_resources.supervisor)가 실제를 맞춘다.
지금의 WinForms 런처가 하던 일을 그대로 담는다 — 데이터 서버(소스 · 로봇 카메라) · 360 슬라이스 · 비전(직접 · 자원 배분).

저장: Redis 키 `ds_desired`(JSON 글자) + `state/desired.json`(원자적). 읽을 때는 둘 중 version 이 큰 쪽.
version 은 쓸 때마다 1 씩 오른다. 칸은 추가만 한다(schema 이름이 바뀌면 계약 버전이 바뀐 것).
쓰기는 `<파일>.lock` 에 배타 잠금(fcntl.flock, state.py 와 같은 방식)을 잡고 **읽기 → version 올리기 → 파일 → Redis**
를 한 번에 한다 — 탭이 둘이어도 한쪽 뜻이 조용히 덮이지 않는다. 잠금을 몇 초 안에 못 잡으면 실패를 돌려준다.

    {
      "schema": "ds_desired/1",
      "version": 7, "updated_at": 1795600000.0, "by": "web",
      "running": true,                                   # 전체 켜기 (false 면 모두 끔)
      "server": {"sources": ["pi13", "a2"], "robot_cam": 1},
      "slice":  {"enabled": false, "u": null, "v": -30},
      "vision": {"enabled": true, "orchestrate": true, "models": ["yoloe"],
                 "profile": {...},                       # 지금 런처의 profile.json 과 같은 모양 (profile_version 은 감독자가 단다)
                 "max_fps": {}},                         # 선택 — 기기별 상한
      "ui": {...}                                        # 선택 — 웹 런처 화면 상태. 감독자는 보지 않는다
    }
"""
import copy
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path

SCHEMA = "ds_desired/1"
LOCK_WAIT_S = 5.0                    # 쓰기 잠금을 이만큼만 기다린다 (영원히 막히지 않게 — 넘으면 실패)
REDIS_KEY = "ds_desired"
SLICE_SOURCE = "cam360_slice"
SLICE_NEEDS = "cam360"
DEFAULT = {"schema": SCHEMA, "version": 0, "updated_at": 0.0, "by": "", "running": False,
           "server": {"sources": [], "robot_cam": 1},
           "slice": {"enabled": False, "u": None, "v": -30},
           "vision": {"enabled": False, "orchestrate": True, "models": [], "profile": {}, "max_fps": {}}}


def normalize(d):
    """빠진 칸을 기본값으로 채운 새 사전 (모르는 칸은 그대로 둔다 — 칸 추가 호환)"""
    out = copy.deepcopy(DEFAULT)
    for k, v in (d or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(copy.deepcopy(v))
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate(d):
    """지금 런처의 Check-Inputs 와 같은 규칙. 문제 목록(비면 정상)."""
    d = normalize(d)
    errs = []
    if d.get("schema") != SCHEMA:
        errs.append("schema 가 %s 가 아님" % SCHEMA)
    srcs = d["server"]["sources"]
    if not isinstance(srcs, list) or not all(isinstance(s, str) and s for s in srcs):
        errs.append("server.sources 는 소스 이름 목록")
    if not d["running"]:
        return errs
    if not srcs:
        errs.append("소스를 하나 이상 고르세요.")
    v = d["vision"]
    if v["enabled"] and not v["models"]:
        errs.append("비전 모델을 하나 이상 고르세요.")
    if v["enabled"] and "targeted" in v["models"]:
        ts = ((v.get("profile") or {}).get("models") or {}).get("targeted", {}).get("targets")
        if ts is not None and len(ts) == 0:
            errs.append("지정 클래스 탐지에는 대상이 하나 이상 필요합니다.")
    s = d["slice"]
    if s["enabled"] and SLICE_NEEDS not in srcs:
        errs.append("360 슬라이스는 cam360 소스도 켜야 합니다.")
    if s["enabled"] and s.get("u") not in (None, ""):
        try:
            float(s["u"])
        except (TypeError, ValueError):
            errs.append("슬라이스 u 는 숫자로 적으세요.")
    return errs


def vision_sources(d):
    """비전이 볼 소스 — 서버 소스 + (슬라이스가 켜졌으면) cam360_slice. 런처의 onAll 과 같다."""
    d = normalize(d)
    return list(d["server"]["sources"]) + ([SLICE_SOURCE] if d["slice"]["enabled"] else [])


def component_keys(d):
    """구성 요소마다 '이 값이 바뀌면 다시 띄운다' 열쇠. 같은 열쇠면 다시 띄우지 않는다(프로필은 파일로 반영).
    지금 런처의 Current-Config / Apply-Changes 규칙과 같다:
      server 가 바뀌면 모두 다시 · slice 는 slice 만 · vision 은 켜기 · 모델 · 소스 · 자원 배분이 바뀔 때만."""
    d = normalize(d)
    if not d["running"]:
        return {"server": None, "slice": None, "vision": None}
    srv = {"sources": d["server"]["sources"], "robot_cam": d["server"]["robot_cam"], "slice": d["slice"]["enabled"]}
    slc = {"u": d["slice"].get("u"), "v": d["slice"].get("v")} if d["slice"]["enabled"] else None
    v = d["vision"]
    vis = ({"models": sorted(v["models"]), "sources": vision_sources(d), "orchestrate": bool(v["orchestrate"]),
            "max_fps": v.get("max_fps") or {}} if v["enabled"] else None)
    return {"server": _key(srv), "slice": _key(slc) if slc is not None else None,
            "vision": _key(vis) if vis is not None else None}


def _key(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


# ------------------------------------------------------------ 읽기 · 쓰기
def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".desired-", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _load_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_redis(r):
    if r is None:
        return None
    try:
        raw = r.get(REDIS_KEY)
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001 — Redis 가 없으면 파일로
        return None


def read(r, path):
    """(원하는 상태, 출처 'redis'|'file'|'default'). 둘 다 있으면 version 이 큰 쪽."""
    a, b = _load_redis(r), _load_file(path)
    cands = [(x, src) for x, src in ((a, "redis"), (b, "file")) if isinstance(x, dict)]
    if not cands:
        return normalize({}), "default"
    x, src = max(cands, key=lambda c: int(c[0].get("version") or 0))
    return normalize(x), src


def _lock(path, wait_s=LOCK_WAIT_S, sleep=time.sleep):
    """`<파일>.lock` 에 배타 잠금 (state.py 와 같은 방식). wait_s 안에 못 잡으면 None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(str(path) + ".lock", "a+")
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.monotonic() >= deadline:
                f.close()
                return None
            sleep(0.02)


def _unlock(f):
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def write(r, path, d, by="web", clock=time.time, wait_s=LOCK_WAIT_S):
    """검사 → (잠금) 읽기 → version 올림 → 파일(원자적) → Redis. (저장된 문서, 문제 목록). 문제가 있으면 쓰지 않는다.

    읽기부터 Redis 까지 한 잠금 안에서 한다 — 탭이 둘이어도 같은 version 을 둘 다 +1 해서 덮는 일이 없다."""
    errs = validate(d)
    if errs:
        return None, errs
    lk = _lock(path, wait_s)
    if lk is None:
        return None, ["다른 쪽이 원하는 상태를 쓰고 있습니다. 잠시 뒤 다시 저장하세요."]
    try:
        cur, _ = read(r, path)
        out = normalize(d)
        out.update(schema=SCHEMA, version=int(cur.get("version") or 0) + 1, updated_at=float(clock()), by=by)
        _atomic_write(path, out)
        if r is not None:
            try:
                r.set(REDIS_KEY, json.dumps(out, ensure_ascii=False))
            except Exception:  # noqa: BLE001 — 파일에는 남았다. 감독자는 파일도 본다
                pass
        return out, []
    finally:
        _unlock(lk)
