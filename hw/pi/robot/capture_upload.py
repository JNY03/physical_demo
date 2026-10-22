# -*- coding: utf-8 -*-
"""
피지컬팀 mk2 — 촬영 산출물 업로더 (HW-R-07 / 아키텍처 §3)
============================================================
`bench/go1_capture_teleop.py` 가 만든 촬영 세션을 **서버가 받을 수 있는 형태**로 묶는다.

    python3 -m robot.capture_upload ~/captures/20260910-134818            # 매니페스트만
    python3 -m robot.capture_upload ~/captures/* --stage /mnt/usb/upload  # 묶어서 적재
    python3 -m robot.capture_upload ~/captures/20260910-134818 --put https://…/  # 업로드

## 왜 매니페스트인가 — 픽셀은 DB 에 넣지 않는다

10fps 로 한 시간이면 36,000장이다. 프레임마다 DB 행을 만들면 메타가 데이터보다 커지고
백업이 불가능해진다. 그래서 **픽셀은 객체 저장소, DB 에는 참조(매니페스트)만** 둔다.
프레임 하나하나의 시각도 박지 않는다 — `t0 + n/interval` 로 복원한다.

## 없는 값은 null 이다

`pose_track` 처럼 아직 만들 수 없는 것은 **키를 지우지 않고 `null` 로 남긴다.** 트윈이
`a,b,c` 를 기대하는데 이 소스가 `c` 를 못 주면 서버가 blank 로 내려보내고, 나중에 다른
센서가 채우면 그때 값이 실린다(docs/ARCHITECTURE_ALIGNMENT.md §1). 0 이나 빈 문자열로
채우면 "쟀더니 0" 과 구별되지 않는다.

## 목적지는 아직 정해지지 않았다

백엔드 회신(BACKEND_AGENDA #15) 전까지는 **묶어서 로컬에 적재**만 한다. 목적지가 정해지면
`--put <base-url>` 로 HTTP PUT 하거나, sink 하나를 더 붙이면 된다 — 매니페스트 형식은
그대로다. 그래서 지금 찍은 데이터도 나중에 그대로 올릴 수 있다.
"""
import argparse
import glob
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, schema      # noqa: E402

MANIFEST_VERSION = "1.0"


def _jpeg_files(session_dir):
    return sorted(f for f in os.listdir(session_dir) if f.endswith(".jpg"))


def _find_sidecar(session_dir, kind):
    """세션 안의 `*_session.json` / `*_manifest.json` 을 찾는다.

    파일명 앞에 노드·개체·시각이 붙기 시작했으므로(common/schema.py artifact_name)
    고정 이름으로 찾으면 못 찾는다. **옛 이름도 계속 받는다** — 이미 찍어 둔 세션이
    디스크에 남아 있고, 그걸 못 읽으면 업로더가 매니페스트를 새로 만들어 버려
    스캔 세션의 shots[] 대응표가 사라진다(§8-9 ⑦ 에서 겪은 그 사고다).
    """
    hits = sorted(glob.glob(os.path.join(session_dir, f"*_{kind}.json")))
    if hits:
        return hits[-1]                       # 이름에 시각이 있어 마지막이 최신이다
    legacy = os.path.join(session_dir, f"{kind}.json")
    return legacy if os.path.exists(legacy) else None


def _frame_naming(frames):
    """프레임 이름 규칙을 **실제 파일에서** 낸다.

    예전에는 `"frame_%06d.jpg"` 를 상수로 박아 뒀다. 이름이 바뀌면 매니페스트가
    조용히 거짓을 말한다 — 서버는 그 규칙으로 파일을 찾을 것이므로 전부 404 다.
    """
    if not frames:
        return None
    first = frames[0]
    stem, ext = os.path.splitext(first)
    n = len(stem) - len(stem.rstrip("0123456789"))
    return (stem[:len(stem) - n] + "%0" + str(n) + "d" + ext) if n else first


def build_manifest(session_dir, identity=None):
    """세션 폴더 -> 매니페스트(dict). 폴더 안의 사실만 적는다 — 모르는 것은 null."""
    session_dir = os.path.abspath(session_dir)
    name = os.path.basename(session_dir.rstrip("/"))

    meta_path = _find_sidecar(session_dir, "session")
    meta = {}
    if meta_path:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    frames = _jpeg_files(session_dir)
    total = sum(os.path.getsize(os.path.join(session_dir, f)) for f in frames)
    fps = meta.get("fps")
    t0 = meta.get("started_at")

    motion_csv = os.path.join(session_dir, "teleop_log.csv")
    ident = identity or schema.Identity.resolve("robot")

    return {
        "schema_version": MANIFEST_VERSION,
        "kind": "capture_session",
        "session_id": f"{ident.entity_id}/{name}",
        "source_id": ident.entity_id,
        "node_id": ident.node_id,
        "zone_id": ident.zone_id,
        "started_at": meta.get("started_at_iso"),
        # 촬영 길이는 프레임 수와 fps 로 낸다. 둘 중 하나라도 모르면 null 이다.
        "duration_s": round(len(frames) / fps, 1) if (fps and frames) else None,
        "sensor": {
            "type": "camera",
            "position": {1: "front", 2: "chin", 3: "left", 4: "right",
                         5: "belly"}.get(meta.get("camera")),
            "camera_id": meta.get("camera"),
            "fps": fps,
            "format": "jpeg",
            "jpeg_quality": meta.get("jpeg_quality"),
        },
        "frames": {
            "count": len(frames),
            "bytes": total,
            "naming": _frame_naming(frames),
            "session_file": os.path.basename(meta_path) if meta_path else None,
            "t0_unix": t0,
            # 프레임 n(1부터)의 시각 = t0_unix + (n-1)*interval_s
            "interval_s": round(1.0 / fps, 4) if fps else None,
            "uri": None,               # 업로드 후 채워진다
        },
        "motion": ({"file": "teleop_log.csv",
                    "columns": ["ts_unix", "vx", "vy", "wz", "estop"],
                    "rows": sum(1 for _ in open(motion_csv, encoding="utf-8")) - 1}
                   if os.path.exists(motion_csv) else None),
        # 아래는 아직 만들 수 없는 것들. 지우지 않고 null 로 남긴다 — 다른 소스가 채운다.
        "pose_track": None,            # 위치 궤적(SLAM/odom 붙으면)
        "frame_ref_base": None,        # 엣지가 디코드 시점에 발급(v8 §6-9)
        "labels": None,                # AI 파트 산출물
    }


def pack(session_dir, out_dir):
    """세션을 tar.gz 로 묶는다. JPEG 는 이미 압축돼 있어 이득이 크지 않지만,
    파일 3만 개를 그대로 옮기는 것보다 한 덩어리가 훨씬 빠르고 안전하다."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.abspath(session_dir).rstrip("/"))
    path = os.path.join(out_dir, f"{name}.tar.gz")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(session_dir, arcname=name)
    return path


def _put(url, body, content_type, token, timeout=120):
    """PUT 한 번. 토큰은 헤더로 보낸다 — URL 쿼리에 실으면 서버 접근 로그에 남는다."""
    req = urllib.request.Request(url, data=body, method="PUT")
    req.add_header("Content-Type", content_type)
    if token:
        req.add_header("X-MK2-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(400).decode("utf-8", "replace")


def upload_session(archive, man, base_url, token, entity_path=False, timeout=120):
    """회신 §8-8 — **PUT 이 두 번이고 매니페스트가 먼저다.**

    서버는 매니페스트를 받아야 그 세션을 받을지 판정한다. 아카이브만 올리면 409 다.
    매니페스트가 2xx 가 아니면 아카이브를 보내지 않는다 — 판정 없이 올린 바이트는
    서버에서 오갈 데가 없다.

    entity_path=True 면 `<base>/<entity_id>/<세션>.tar.gz` 로 올린다(§8-10 20).
    끄면 `<base>/<세션>.tar.gz` — 백엔드가 실측한 형태다.
    """
    stem = os.path.basename(archive)
    if stem.endswith(".tar.gz"):
        stem = stem[:-7]

    prefix = base_url.rstrip("/")
    if entity_path:
        ent = man.get("source_id") or ""
        if ent:
            prefix = prefix + "/" + ent

    man_url = prefix + "/" + stem + ".manifest.json"
    arc_url = prefix + "/" + stem + ".tar.gz"

    body = json.dumps(man, ensure_ascii=False).encode("utf-8")
    st, msg = _put(man_url, body, "application/json", token, timeout)
    print("[매니페스트] " + man_url + " (" + str(st) + ") " + msg.strip()[:120])
    if not (200 <= st < 300):
        return arc_url, st, False

    with open(archive, "rb") as f:
        st2, msg2 = _put(arc_url, f.read(), "application/gzip", token, timeout)
    print("[아카이브]   " + arc_url + " (" + str(st2) + ") " + msg2.strip()[:120])
    return arc_url, st2, 200 <= st2 < 300


def http_put(path, base_url, timeout=120):
    """구형 단일 PUT. 매니페스트를 먼저 보내지 않아 409 가 난다 — upload_session 을 쓸 것."""
    url = base_url.rstrip("/") + "/" + os.path.basename(path)
    with open(path, "rb") as f:
        st, _ = _put(url, f.read(), "application/gzip", None, timeout)
    return url, st


def main():
    ap = argparse.ArgumentParser(description="촬영 세션 -> 매니페스트/적재/업로드")
    ap.add_argument("sessions", nargs="+", help="세션 디렉터리 (여러 개 가능)")
    ap.add_argument("--stage", default=None,
                    help="tar.gz 로 묶어 이 디렉터리에 적재 (목적지 미정일 때)")
    ap.add_argument("--put", default=None,
                    help="이 base URL 로 HTTP PUT (S3 호환). --stage 와 같이 쓸 수 있다")
    ap.add_argument("--token", default=os.environ.get("MK2_CAPTURE_TOKEN", ""),
                    help="업로드 토큰. 환경변수 MK2_CAPTURE_TOKEN 이 기본값이다 "
                         "(명령줄에 그대로 치면 셸 히스토리와 ps 에 남는다)")
    ap.add_argument("--entity-path", action="store_true",
                    help="<base>/<entity_id>/<세션> 형태로 올린다 (회신 §8-10 20). "
                         "끄면 <base>/<세션> — 백엔드가 실측한 형태다")
    ap.add_argument("--delete-after", action="store_true",
                    help="업로드가 확인된 세션만 원본 삭제")
    args = ap.parse_args()

    for sess in args.sessions:
        if not os.path.isdir(sess):
            print(f"[건너뜀] 디렉터리가 아니다: {sess}")
            continue

        # 이미 매니페스트가 있고 **다른 종류의 세션**이면(예: 8방향 스캔 촬영) 그 도구가
        # 만든 것을 그대로 존중한다. 여기서 덮어쓰면 사진과 방위의 대응이 사라진다.
        mpath = _find_sidecar(sess, "manifest")
        man = None
        broken = False
        if mpath:
            try:
                with open(mpath, encoding="utf-8") as f:
                    man = json.load(f)
            except ValueError:
                # 회신 §8-9 ⑦: 여기서 재생성하면 스캔 세션의 8방향 shots[] 대응표가
                # 조용히 사라진다. 깨진 것은 고치거나 지우고 다시 올리는 편이 낫다.
                man = None
                broken = True

        if broken:
            print("[거부] 매니페스트 JSON 이 깨졌다: " + mpath)
            print("       재생성하면 스캔 세션의 shots[] 가 사라진다 (§8-9 ⑦).")
            continue

        if man is None or man.get("kind") == "capture_session":
            man = build_manifest(sess)

        if mpath is None:
            # 매니페스트가 아직 없던 세션이다. 새 규칙으로 이름을 짓는다 —
            # 시각은 세션의 시작 시각을 쓴다(지금 시각을 쓰면 나중에 다시 돌릴 때마다
            # 이름이 달라져 같은 세션의 매니페스트가 여러 벌 쌓인다).
            mpath = os.path.join(sess, schema.artifact_name(
                man.get("node_id"), man.get("source_id"), parts=("manifest",),
                unix_ts=(man.get("frames") or {}).get("t0_unix"), ext="json"))

        if man.get("kind") == "capture_session":
            n = man["frames"]["count"]
            size_bytes = man["frames"]["bytes"]
        else:
            shots = man.get("shots") or []
            n = len(shots)
            size_bytes = sum(sh.get("bytes") or 0 for sh in shots)
        if n == 0:
            print(f"[건너뜀] 사진 0장: {sess}")
            continue

        archive = None
        if args.stage:
            archive = pack(sess, args.stage)

        uploaded = False
        if args.put:
            if archive is None:
                archive = pack(sess, "/tmp")
            try:
                uri, status, ok = upload_session(
                    archive, man, args.put, args.token, args.entity_path)
                if man.get("kind") == "capture_session":
                    man["frames"]["uri"] = uri
                else:
                    man["uri"] = uri
                uploaded = ok
            except Exception as e:
                print(f"[업로드 실패] {type(e).__name__}: {e}")

        # 매니페스트는 세션 옆에 남긴다. 목적지가 정해지면 이 파일만 보내면 된다.
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)

        label = man.get("session_id") or man.get("session") or os.path.basename(sess)
        print(f"[{label}] 사진 {n}장 {size_bytes/1e6:.1f}MB "
              f"길이 {man.get('duration_s')}s -> {mpath}"
              + (f" / 적재 {archive}" if archive else ""))

        if args.delete_after and uploaded:
            import shutil
            shutil.rmtree(sess)
            print(f"[삭제] {sess} (업로드 확인됨)")
        elif args.delete_after:
            print(f"[보존] {sess} — 업로드가 확인되지 않아 지우지 않는다")


if __name__ == "__main__":
    main()
