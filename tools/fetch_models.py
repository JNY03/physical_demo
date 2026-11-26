#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_models.py — 고른 프로필에 필요한 가중치 · 외부 코드만 받고 해시로 검증한다 (2026-11-25, Docker 계획 D3).

implements: Docker_배포_계획_261125.md §6 「가중치 받기」 · §7 D3

  venv/bin/python tools/fetch_models.py --check --all                 지금 있는 것만 검증 (받지 않는다)
  venv/bin/python tools/fetch_models.py --models yoloe river          선택지(또는 모델 키)에 필요한 것을 받는다
  venv/bin/python tools/fetch_models.py --profile 프로필.json          런처 프로필 그대로 (대상 · 근거 · 기능까지 본다)
  venv/bin/python tools/fetch_models.py --all --dest /ds/models        레지스트리의 전부
  venv/bin/python tools/fetch_models.py --hash models/새가중치.pt       새 항목에 적을 sha256 (폴더면 manifest)

**무엇을 받는가 — vision/model_registry.yaml models.<키>.source 한 곳.** 모델을 늘릴 때 이 파일은 고치지 않는다.
source 항목 하나는 아래 넷 중 하나다 (path 는 DS_HOME 기준. models/… 는 --dest 로 옮겨진다):

  {path, url, sha256}                            파일 하나를 URL 에서 (GitHub 릴리스 · openaipublic 등)
  {path, hf, rev, file, sha256}                  HF 저장소의 파일 하나 (hf: 소유자/이름 · datasets/소유자/이름)
  {path, hf, rev, include, sha256}               HF 저장소의 파일 여럿을 폴더로. sha256 은 manifest (아래)
  {hf, rev, include, cache: true, sha256}        HF 캐시(HF_HUB_CACHE)에 받는다 — from_pretrained(저장소 id) 로 읽는 것
  {path, repo, commit}                           git 저장소를 그 커밋으로 (external/…)
  {path, manual, sha256?}                        받을 곳이 없다 (직접 학습 · 손본 파일). 안내만 하고, 해시가 있으면 검증

manifest = sha256( 정렬된 "상대경로\\t파일sha256\\n" 줄들 ) — include 에 적은 파일만 센다.

판정: ok · missing · mismatch · manual(없음) · unverified(해시 칸 없음) · failed(받기 실패) · info(안내만). 받기는 없는 것 · 틀린 것만 받는다.
받은 것은 임시 이름으로 쓰고 해시가 맞을 때만 제자리로 옮긴다(os.replace) — 중간에 끊겨도 반쪽 파일이 남지 않는다.

의존성: 표준 라이브러리 + PyYAML(레지스트리 읽기, cue_catalog 와 같다). **torch 를 import 하지 않는다.**
HF 는 huggingface_hub 없이 resolve URL 로 받는다 (HF_TOKEN 이 있으면 붙인다). git 항목을 받을 때만 git 이 필요하다.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "vision"))

import cue_catalog as cc  # noqa: E402  (torch 없이 레지스트리 · 단서 의존 관계를 읽는다)

HF_URL = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 1 << 20
UA = "data_stream-fetch_models/1"


# ---------------------------------------------------------------- 경로
def ds_home():
    return Path(os.environ.get("DS_HOME") or REPO).expanduser()


def hf_cache_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def resolve_path(path, home=None, dest=None):
    """레지스트리 경로 → 실제 경로. 절대 경로는 그대로, models/… 는 dest 아래로, 나머지는 DS_HOME 아래로.
    옛 절대 경로(…/models/x)도 dest 를 주면 그 아래로 옮겨 푼다."""
    home = Path(home) if home else ds_home()
    p = Path(str(path)).expanduser()
    parts = p.parts
    if dest is not None and "models" in parts:
        i = parts.index("models")
        if i == 0 or p.is_absolute():
            return Path(dest).joinpath(*parts[i + 1:])
    return p if p.is_absolute() else home / p


def hf_repo_dir(repo):
    """'owner/name' · 'datasets/owner/name' → HF 캐시 폴더 이름 (models--owner--name)."""
    kind, rid = _hf_split(repo)
    return "%ss--%s" % (kind, rid.replace("/", "--"))


def _hf_split(repo):
    r = str(repo).strip("/")
    for k in ("datasets", "spaces"):
        if r.startswith(k + "/"):
            return k[:-1], r[len(k) + 1:]
    return "model", r


def hf_file_url(repo, rev, file):
    kind, rid = _hf_split(repo)
    pre = "" if kind == "model" else kind + "s/"
    return "%s/%s%s/resolve/%s/%s" % (HF_URL, pre, rid, rev, file)


# ---------------------------------------------------------------- 해시
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def manifest_hash(root, include=None):
    """폴더 → manifest sha256. include 가 없으면 폴더 안 전부(.cache · .git 은 뺀다). 빠진 파일이 있으면 None."""
    root = Path(root)
    if include is None:
        names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                       if p.is_file() and not ({".cache", ".git"} & set(p.relative_to(root).parts)))
    else:
        names = sorted(str(n) for n in include)
    lines = []
    for n in names:
        f = root / n
        if not f.is_file():
            return None
        lines.append("%s\t%s\n" % (n, sha256_file(f)))
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def git_head(path):
    """git 없이 HEAD 커밋을 읽는다 (검사는 git 이 없는 컨테이너에서도 돈다)."""
    g = Path(path) / ".git"
    if g.is_file():                                     # worktree · submodule: "gitdir: …"
        g = (Path(path) / g.read_text().split(":", 1)[1].strip()).resolve()
    try:
        head = (g / "HEAD").read_text().strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head
    ref = head.split(None, 1)[1]
    if (g / ref).is_file():
        return (g / ref).read_text().strip()
    packed = g / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return None


# ---------------------------------------------------------------- 필요한 모델
@contextlib.contextmanager
def _assume_present():
    """cue_catalog 의 사용 가능 판정을 「다 있다」로 두고 의존 관계만 본다 — 빈 models/ 에서도 받을 목록이 나와야 한다."""
    saved = cc.have, cc._trained_ok
    cc.have = lambda key, files: True
    cc._trained_ok = lambda t: True
    try:
        yield
    finally:
        cc.have, cc._trained_ok = saved


def needed(options=(), profile=None, cfg=None):
    """선택지 · 모델 키 · 프로필 → (모델 키 집합, 대상 학습 가중치 [(대상, 경로)]).

    선택지의 cue_models, targeted 는 cue_catalog.resolve_profile + needed_models 로 (근거 needs · OVD · 거리 포함),
    features.reid → reid. 모델 키(models.<키>)를 직접 줘도 된다."""
    reg = cc._REG
    profile = dict(profile or {})
    opts = list(options) + [m for m in (profile.get("models") or {}) if m not in options]
    need, trained = set(), []
    for o in opts:
        if o in reg["models"]:
            need.add(o)
            continue
        if o not in reg["options"]:
            raise SystemExit("모르는 선택지 · 모델 %r (선택지: %s · 모델: %s)"
                             % (o, ", ".join(reg["options"]), ", ".join(reg["models"])))
        cm = reg["options"][o].get("cue_models")
        if isinstance(cm, dict):
            for ms in cm.values():
                need.update(ms or [])
    if "targeted" in opts or (profile.get("features") or {}).get("reid"):
        tcfg = ((cfg or {}).get("models") or {}).get("targeted") or {}
        with _assume_present():
            r = cc.resolve_profile(profile, cfg or {}, [o for o in opts if o in reg["options"]])
            need |= cc.needed_models(r["targeted"], tcfg)
        for t in (r["targeted"] or {}).get("targets", []):
            if t.get("detector") == "trained":
                tr = t.get("trained") or {}
                trained += [(t["name"], tr[k]) for k in ("weight", "fit") if tr.get(k)]
        if (r.get("features") or {}).get("reid"):
            need.add("reid")
    return need, trained


# ---------------------------------------------------------------- 항목
def items_for(models, home=None, dest=None, hf_cache=None):
    """모델 키들 → 받을 항목 목록 (같은 대상은 한 번만). git 항목이 먼저 — 그 안에 받을 가중치가 있을 수 있다."""
    out, seen = [], set()
    for m in sorted(models):
        spec = cc._REG["models"].get(m)
        if spec is None:
            raise SystemExit("레지스트리에 없는 모델 %r" % m)
        srcs = spec.get("source")
        if srcs is None:
            if spec.get("files"):
                out.append({"model": m, "kind": "nosource", "path": None,
                            "where": ", ".join(map(str, spec["files"])), "sha256": None})
            continue
        for s in srcs:
            it = dict(s)
            it["model"] = m
            if "repo" in s:
                it["kind"] = "git"
            elif "manual" in s:
                it["kind"] = "manual"
            elif "url" in s:
                it["kind"] = "url"
            elif "hf" in s:
                it["kind"] = "hf_file" if "file" in s else "hf_dir"
            else:
                raise SystemExit("모델 %s 의 source 항목을 모름: %r" % (m, s))
            if s.get("cache"):
                it["dest"] = Path(hf_cache or hf_cache_dir()) / hf_repo_dir(s["hf"]) / "snapshots" / str(s["rev"])
            elif "path" not in s:                     # 경로 없는 안내 (예: trained — 경로는 대상 파일이 가진다)
                it["dest"] = None
            else:
                it["dest"] = resolve_path(s["path"], home, dest)
            it["where"] = "-" if it["dest"] is None else str(it["dest"])
            key = (it["where"], it.get("sha256"), it.get("commit"), tuple(it.get("include") or ()))
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
    out.sort(key=lambda i: i["kind"] != "git")
    return out


def check_item(it):
    """→ (상태, 설명). 상태: ok · missing · mismatch · manual · unverified · nosource · info(경로 없는 안내)"""
    k = it["kind"]
    if k == "nosource":
        return "nosource", "레지스트리에 source 가 없음"
    d = it["dest"]
    if d is None:
        return "info", it.get("manual", "")
    if k == "git":
        if not d.is_dir():
            return "missing", "없음"
        head = git_head(d)
        if head is None:
            return "mismatch", "git 저장소가 아님"
        return ("ok", head[:12]) if head == it["commit"] else ("mismatch", "커밋 %s ≠ %s" % (head[:12], it["commit"][:12]))
    want = it.get("sha256")
    if k == "hf_dir":
        if not d.is_dir():
            got = None
        else:
            got = manifest_hash(d, it.get("include"))
        if it.get("cache"):
            ref = d.parent.parent / "refs" / "main"
            if got is not None and ref.is_file() and ref.read_text().strip() != str(it["rev"]):
                return "mismatch", "refs/main 이 %s 를 가리키지 않음" % str(it["rev"])[:12]
    else:
        got = sha256_file(d) if d.is_file() else None
    if got is None:
        return ("manual", it.get("manual", "")) if k == "manual" else ("missing", "없음")
    if not want:
        return "unverified", "해시 칸 없음 (sha256 %s)" % got
    return ("ok", got[:12]) if got == want else ("mismatch", "sha256 %s ≠ %s" % (got[:12], want[:12]))


# ---------------------------------------------------------------- 받기
def _open(url):
    headers = {"User-Agent": UA}
    tok = os.environ.get("HF_TOKEN")
    if tok and url.startswith(HF_URL):
        headers["Authorization"] = "Bearer " + tok
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def download(url, out, want=None, log=print):
    """url → out (임시 이름에 쓰고 해시가 맞으면 os.replace). 틀리면 지우고 ValueError."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + out.name + ".", suffix=".part", dir=str(out.parent))
    h, n = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as f, _open(url) as r:
            total = int(r.headers.get("Content-Length") or 0)
            for b in iter(lambda: r.read(CHUNK), b""):
                f.write(b)
                h.update(b)
                n += len(b)
        got = h.hexdigest()
        if want and got != want:
            raise ValueError("sha256 이 다름: %s (받은 %s ≠ 레지스트리 %s)" % (url, got[:12], want[:12]))
        os.replace(tmp, out)
        log("  받음 %s (%.1f MB%s)" % (out, n / 1e6, "" if not total or total == n else ", 길이 %d≠%d" % (n, total)))
        return got
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def fetch_item(it, log=print):
    k, d = it["kind"], it["dest"]
    if k == "url":
        download(it["url"], d, it.get("sha256"), log)
    elif k == "hf_file":
        download(hf_file_url(it["hf"], it["rev"], it["file"]), d, it.get("sha256"), log)
    elif k == "hf_dir":
        # 임시 폴더에 다 받고 manifest 가 맞을 때만 파일별로 제자리에 옮긴다
        d.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".fetch-", dir=str(d)))
        try:
            for f in it["include"]:
                download(hf_file_url(it["hf"], it["rev"], f), stage / f, None, log)
            got = manifest_hash(stage, it["include"])
            if it.get("sha256") and got != it["sha256"]:
                raise ValueError("manifest 가 다름: %s@%s (받은 %s ≠ 레지스트리 %s)"
                                 % (it["hf"], it["rev"], got[:12], it["sha256"][:12]))
            for f in it["include"]:
                (d / f).parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / f, d / f)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        if it.get("cache"):                              # from_pretrained(저장소 id) 가 오프라인에서 이 판을 찾도록
            ref = d.parent.parent / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(str(it["rev"]))
    elif k == "git":
        if d.exists():
            raise ValueError("%s 가 이미 있고 커밋이 다르다 — 손대지 않는다. 맞추려면: git -C %s fetch && git -C %s checkout %s"
                             % (d, d, d, it["commit"]))
        if shutil.which("git") is None:
            raise ValueError("git 이 없다 — git clone %s %s && git -C %s checkout %s" % (it["repo"], d, d, it["commit"]))
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="." + d.name + ".", dir=str(d.parent)))
        try:
            subprocess.run(["git", "clone", "--quiet", "--no-checkout", it["repo"], str(tmp)], check=True)
            subprocess.run(["git", "-C", str(tmp), "checkout", "--quiet", "--detach", it["commit"]], check=True)
            os.replace(tmp, d)
            log("  받음 %s @ %s" % (d, it["commit"][:12]))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    else:
        raise ValueError("받을 수 없는 항목: %s" % k)


# ---------------------------------------------------------------- 명령줄
def run(items, do_fetch, log=print):
    """항목마다 검사(→ 받기 → 다시 검사). → (행 목록, 종료 코드)"""
    rows, code = [], 0
    for it in items:
        st, why = check_item(it)
        if do_fetch and st in ("missing", "mismatch") and it["kind"] not in ("manual", "nosource"):
            log("받는 중: %s → %s" % (it["model"], it["where"]))
            try:
                fetch_item(it, log)
                st, why = check_item(it)
            except Exception as exc:   # 하나가 실패해도 나머지는 받는다
                st, why = "failed", str(exc)
        if st == "manual":
            log("직접 놓아야 함: %s → %s\n    %s" % (it["model"], it["where"], it.get("manual", "")))
        if st not in ("ok", "unverified", "info") and not it.get("optional"):
            code = 1
        rows.append({"model": it["model"], "kind": it["kind"], "path": it["where"], "status": st, "detail": why})
    return rows, code


# ---------------------------------------------------------------- 손으로 넣기 (런처 업로드)
def place_slots(dest=None, home=None):
    """레지스트리가 바라는 **파일 자리** → [{model, name, path, sha256}].

    이름(basename)으로 찾는다 — 사람이 가져오는 파일은 보통 이름이 같다(yoloe-11l-seg-pf.pt 등).
    git 저장소(external/…)와 HF 캐시 폴더는 파일 하나로 메울 수 없어 뺀다."""
    out = []
    for m in sorted(cc._REG["models"]):
        for it in items_for([m], home=home, dest=dest):
            if it["kind"] in ("git", "hf_dir", "nosource") or not it.get("dest"):
                continue
            d = Path(it["dest"])
            out.append({"model": it["model"], "name": d.name, "path": str(d), "sha256": it.get("sha256"),
                        "kind": it["kind"]})
    return out


def place_files(paths, dest=None, home=None, force=False, log=print):
    """고른 파일(.zip 이면 그 안의 파일들)을 레지스트리 자리에 넣는다 → {ok, placed, skipped, errors}.

    왜 이름으로 찾나: 레지스트리가 「어느 자리에 무슨 이름의 파일」인지 이미 적고 있다. 사람은 그 파일을
    어디서든 가져올 수 있으니(다른 기기 · USB · 사내 저장소), 받는 길을 늘리는 대신 **놓는 길**을 연다.
    sha256 이 적혀 있는데 다르면 넣지 않는다 — 조용히 틀린 가중치로 도는 것이 가장 나쁘다(force 로만 덮는다)."""
    import shutil
    import tempfile
    import zipfile
    slots = place_slots(dest, home)
    by_name = {}
    for s in slots:
        by_name.setdefault(s["name"], []).append(s)
    placed, skipped, errors = [], [], []
    files = []
    tmp = None
    for raw in paths:
        f = Path(raw)
        if not f.exists():
            errors.append({"file": str(f), "why": "파일이 없습니다"})
            continue
        if f.suffix.lower() == ".zip":
            try:
                tmp = tmp or tempfile.mkdtemp(prefix="ds-weights-")
                with zipfile.ZipFile(f) as z:
                    for n in z.namelist():
                        if n.endswith("/"):
                            continue
                        got = z.extract(n, tmp)
                        files.append(Path(got))
            except (OSError, zipfile.BadZipFile) as e:
                errors.append({"file": str(f), "why": "zip 을 열지 못했습니다: %s" % e})
            continue
        files.append(f)
    for f in files:
        cands = by_name.get(f.name) or []
        if not cands:
            errors.append({"file": f.name,
                           "why": "어느 모델의 파일인지 모릅니다 — 레지스트리가 바라는 이름이어야 합니다",
                           "expect": sorted(by_name)[:40]})
            continue
        for s in cands:
            d = Path(s["path"])
            want = s.get("sha256")
            if want:
                got = sha256_file(f)
                if got != want:
                    errors.append({"file": f.name, "model": s["model"],
                                   "why": "sha256 이 레지스트리와 다릅니다 (받은 %s… · 바라는 %s…)"
                                          % (got[:12], want[:12])})
                    continue
            if d.exists() and not force:
                skipped.append({"model": s["model"], "path": str(d), "why": "이미 있습니다"})
                continue
            try:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d)
            except OSError as e:
                errors.append({"file": f.name, "model": s["model"], "why": "복사 실패: %s" % e})
                continue
            placed.append({"model": s["model"], "path": str(d), "verified": bool(want)})
            log("넣음: %s → %s" % (f.name, d))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": bool(placed) and not errors, "placed": placed, "skipped": skipped, "errors": errors}


def main(argv=None):
    ap = argparse.ArgumentParser(description="가중치 · 외부 코드 받기와 sha256 검증 (vision/model_registry.yaml source)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="레지스트리의 모든 모델")
    g.add_argument("--hash", metavar="경로", help="새 source 항목에 적을 sha256 (폴더면 manifest, --include 로 고른 파일만)")
    ap.add_argument("--models", nargs="+", default=[], metavar="이름", help="선택지(yoloe river targeted …) 또는 모델 키")
    ap.add_argument("--profile", metavar="JSON", help="런처 프로필 (대상 · 근거 · features 까지 본다)")
    ap.add_argument("--config", default=str(REPO / "vision" / "vision_config.yaml"), help="targeted 기본 대상을 읽는 설정")
    ap.add_argument("--check", action="store_true", help="받지 않고 검증만")
    ap.add_argument("--dest", metavar="폴더", help="models/ 대신 받을 곳 (기본 DS_HOME/models)")
    ap.add_argument("--hf-cache", metavar="폴더", help="HF 캐시 (기본 HF_HUB_CACHE · HF_HOME/hub · ~/.cache/huggingface/hub)")
    ap.add_argument("--include", nargs="+", help="--hash 폴더에서 셀 파일")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로")
    ap.add_argument("--place", nargs="+", metavar="파일", help="손으로 가져온 가중치를 레지스트리 자리에 넣는다 "
                                                              "(.zip 이면 그 안의 파일들). 런처 「가중치 넣기」")
    ap.add_argument("--slots", action="store_true", help="레지스트리가 바라는 파일 이름 · 자리 목록")
    ap.add_argument("--force", action="store_true", help="--place: 이미 있는 파일도 덮는다")
    a = ap.parse_args(argv)
    if a.slots:
        rows = place_slots(a.dest)
        print(json.dumps(rows, ensure_ascii=False, indent=1) if a.json
              else "\n".join("%-16s %-44s %s" % (r["model"], r["name"], r["path"]) for r in rows))
        return 0
    if a.place:
        res = place_files(a.place, dest=a.dest, force=a.force, log=(lambda *_: None) if a.json else print)
        if a.json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            for x in res["placed"]:
                print("넣음   %s → %s%s" % (x["model"], x["path"], " (sha256 확인)" if x["verified"] else ""))
            for x in res["skipped"]:
                print("건너뜀 %s — %s" % (x["path"], x["why"]))
            for x in res["errors"]:
                print("못 넣음 %s — %s" % (x.get("file"), x["why"]))
        return 0 if res["ok"] else 1

    if a.hash:
        p = Path(a.hash).expanduser()
        print(manifest_hash(p, a.include) if p.is_dir() else sha256_file(p))
        return 0
    profile = None
    if a.profile:
        profile = json.loads(Path(a.profile).read_text(encoding="utf-8-sig"))
    if a.all:
        models, trained = set(cc._REG["models"]), []
    elif a.models or profile:
        try:
            models, trained = needed(a.models, profile, cc._load_yaml(a.config))
        except cc.ProfileError as exc:
            print("프로필 오류: %s" % exc, file=sys.stderr)
            return 2
    else:
        ap.error("--all · --models · --profile 중 하나를 준다")
    log = (lambda *x: print(*x, file=sys.stderr)) if a.json else print
    items = items_for(models, dest=a.dest, hf_cache=a.hf_cache)
    for name, path in trained:                      # 대상별 학습 가중치 — 대상 파일이 경로를 갖는다
        items.append({"model": "trained:" + name, "kind": "manual", "dest": resolve_path(path, dest=a.dest),
                      "where": str(resolve_path(path, dest=a.dest)), "sha256": None,
                      "manual": "대상 %s 의 학습 가중치 (vision/assets/targets/%s.json trained)" % (name, name)})
    rows, code = run(items, not a.check, log)
    if a.json:
        print(json.dumps({"ok": code == 0, "models": sorted(models), "items": rows}, ensure_ascii=False, indent=1))
    else:
        print("\n%-16s %-8s %-10s %s" % ("모델", "종류", "상태", "경로 · 설명"))
        for r in rows:
            print("%-16s %-8s %-10s %s  %s" % (r["model"], r["kind"], r["status"], r["path"], r["detail"]))
        print("\n필요한 모델: %s → %s" % (", ".join(sorted(models)), "모두 맞음" if code == 0 else "맞지 않는 것이 있음"))
    return code


if __name__ == "__main__":
    sys.exit(main())
