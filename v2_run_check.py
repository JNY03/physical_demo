#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2_run_check.py — 데이터 스트림 저장 폴더의 실행 하나를 소스마다 한 화면으로 요약한다.

2026-10-14 · 통합데이터규약 v2 적용 시험용 (DS · 360 을 WSL 노트북에 올린 뒤 「말단 → 엣지」「말단 → 서버」 확인).
표준 라이브러리만 쓴다 — 파이썬 3.8 이상이면 어느 venv 로 돌려도 된다. 읽기만 하고 아무것도 쓰지 않는다.

쓰는 법
  python3 v2_run_check.py <저장 폴더>                              소스마다 가장 최근 실행
  python3 v2_run_check.py <저장 폴더> --run 20261014_101500         그 실행
  python3 v2_run_check.py <저장 폴더> --run <새 실행> --compare 20261008_161514
                                                                   옛 실행(10-08 등)과 나란히 본다
  python3 v2_run_check.py <저장 폴더> --streams drone cam360         소스 고르기

  <저장 폴더> 는 server_stream_multi_source.py 의 --save (기본: 그 파일 옆 stream_data).
  폴더 꼴은 둘 다 읽는다: <저장 폴더>/<소스>/<실행>/  (DS 가 쓰는 꼴)
                          <저장 폴더>/<실행>_<소스>/   (서버 river_data 처럼 옮겨 둔 꼴. go1 = robot1, 360 = cam360)

보는 것 (소스마다)
  기록     frames.jsonl 줄 수 · 사진 파일 수 · 받은 시간 · 장/초
           [DS 몫 — v2 DS 면 0 이어야 한다]  received_at 없음 · 옛 이름(source/file/host) · sender._file ·
                                             JSON 상태를 글자로 저장
  장비     sender 가 v2 프레임 헤더인 줄 / 옛 모양인 줄 (장비가 아직 옛 판이면 옛 모양이 정상)
           v2 장비면 번호(sequence_id) 건너뜀 · 같은 번호 반복 · 거꾸로 감, 받은 시각 − 장비 시각(ms)
  넘기기   edge 칸이 있는 줄(다른 기기의 DS 가 --forward 로 넘긴 것) · 넘긴 쪽 번호(edge.n) 건너뜀 ·
           이 기기가 받은 시각 − 넘긴 기기가 받은 시각(ms)
  상태     states.jsonl 줄 수 · enc 별 · v2 상태 · 드론 frame 상태 중 frame_ref 있는 것
  비전     vision/<모델>/results.jsonl 줄 수 · v2 결과 · emitted_at · frame_ref · 결과 − 원본 시각(ms) · 오류 줄

끝 코드: 0 = DS 몫 위반 없음, 1 = 있음, 2 = 읽을 실행이 없음.
"""
import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

STREAMS = ("cam360", "drone", "robot1", "robot2")
ALIASES = {"cam360": ("cam360", "360"), "drone": ("drone",), "robot1": ("robot1", "go1"), "robot2": ("robot2",)}


def to_epoch(v):
    """ISO 글자 또는 epoch 숫자(초·밀리초) → epoch 초. 모르면 None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) / 1000.0 if v > 1e11 else float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v).timestamp()
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return None
    return None


def read_jsonl(path):
    rows, bad = [], 0
    if not path.exists():
        return rows, bad
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                break                                   # 쓰는 중인 마지막 줄
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    return rows, bad


def stats_ms(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return "-"
    p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
    return "중앙 %d · p90 %d · 최소 %d · 최대 %d (%d개)" % (statistics.median(v), p90, v[0], v[-1], len(v))


def sender_time(s):
    """장비 시각(epoch 초) — v2 timestamp · 옛 360 timestamp(epoch 글자) · 드론 ts · captured_at · GO1 X-Timestamp(ms)."""
    if not isinstance(s, dict):
        return None
    for k in ("timestamp", "ts", "captured_at", "observed_at", "X-Timestamp-Iso", "X-Timestamp"):
        t = to_epoch(s.get(k))
        if t is not None:
            return t / 1000.0 if t > 1e11 else t
    return None


def is_v2_sender(s):
    return (isinstance(s, dict) and str(s.get("schema_version", "")).startswith("2")
            and isinstance(s.get("camera_id"), str) and isinstance(s.get("sequence_id"), int)
            and not isinstance(s.get("sequence_id"), bool))


def run_dirs(save, stream):
    """(실행 이름, 경로) 목록 — 두 폴더 꼴을 다 찾는다."""
    out = {}
    for alias in ALIASES[stream]:
        d = save / alias
        if d.is_dir():
            for p in d.iterdir():
                if p.is_dir() and (p / "frames.jsonl").exists():
                    out.setdefault(p.name, p)
    if save.is_dir():
        for p in save.iterdir():
            for alias in ALIASES[stream]:
                suf = "_" + alias
                if p.is_dir() and p.name.endswith(suf) and (p / "frames.jsonl").exists():
                    out.setdefault(p.name[: -len(suf)], p)
    return sorted(out.items())


def seq_report(keyed):
    """{(카메라, 세션): [번호…]} → 건너뜀 · 반복 · 거꾸로"""
    gaps = missing = repeats = back = 0
    for seqs in keyed.values():
        for a, b in zip(seqs, seqs[1:]):
            if b == a:
                repeats += 1
            elif b < a:
                back += 1
            elif b > a + 1:
                gaps += 1
                missing += b - a - 1
    return gaps, missing, repeats, back


def summarize(run_path, root_for_files):
    S = {"path": str(run_path)}
    rows, bad = read_jsonl(run_path / "frames.jsonl")
    S["frames"], S["bad_lines"] = len(rows), bad
    imgs = run_path / "frames"
    S["images"] = sum(1 for p in imgs.iterdir() if p.is_file()) if imgs.is_dir() else 0
    # DS 몫
    S["no_received_at"] = sum(1 for r in rows if not r.get("received_at"))
    S["old_names"] = sum(1 for r in rows if any(k in r for k in ("source", "file", "host")))
    S["sender_file"] = sum(1 for r in rows if isinstance(r.get("sender"), dict) and "_file" in r["sender"])
    # 받은 시간
    rt = [to_epoch(r.get("received_at")) for r in rows]
    rt = [t for t in rt if t is not None]
    st_old = [sender_time(r.get("sender")) for r in rows]
    st_old = [t for t in st_old if t is not None]
    if len(rt) >= 2:
        S["duration_s"], S["time_from"] = max(rt) - min(rt), "received_at"
    elif len(st_old) >= 2:                                  # 옛 DS 기록 — 받은 시각이 없어 장비 시각으로
        S["duration_s"], S["time_from"] = max(st_old) - min(st_old), "장비 시각"
    elif S["images"] >= 2:
        ms = sorted(p.stat().st_mtime for p in imgs.iterdir() if p.is_file())
        S["duration_s"], S["time_from"] = ms[-1] - ms[0], "사진 파일 시각(복사한 폴더면 틀린다)"
    else:
        S["duration_s"], S["time_from"] = None, "-"
    S["fps"] = (S["frames"] - 1) / S["duration_s"] if S["duration_s"] and S["duration_s"] >= 1.0 else None
    # 장비 (sender)
    v2 = [r for r in rows if is_v2_sender(r.get("sender"))]
    S["sender_v2"], S["sender_old"] = len(v2), len(rows) - len(v2)
    keyed = {}
    lat = []
    for r in v2:
        s = r["sender"]
        keyed.setdefault((s["camera_id"], s.get("session_id")), []).append(s["sequence_id"])
        a, b = to_epoch(r.get("received_at")), to_epoch(s.get("timestamp"))
        if a is not None and b is not None:
            lat.append((a - b) * 1000.0)
    S["cameras"] = sorted({k[0] for k in keyed})
    S["sessions"] = len(keyed)
    S["seq"] = seq_report(keyed)
    S["lat"] = stats_ms(lat)
    old_keys = set()
    for r in rows:
        s = r.get("sender")
        if isinstance(s, dict) and not is_v2_sender(s):
            old_keys.update(s.keys())
    S["old_sender_keys"] = sorted(old_keys)[:12]
    # 넘기기 (edge)
    fw = [r for r in rows if isinstance(r.get("edge"), dict)]
    S["edge"] = len(fw)
    en = [r["edge"].get("n") for r in fw if isinstance(r["edge"].get("n"), int)]
    S["edge_seq"] = seq_report({"e": en}) if en else (0, 0, 0, 0)
    fd = []
    for r in fw:
        a, b = to_epoch(r.get("received_at")), to_epoch(r["edge"].get("received_at"))
        if a is not None and b is not None:
            fd.append((a - b) * 1000.0)
    S["edge_delay"] = stats_ms(fd)
    # 상태
    st, sbad = read_jsonl(run_path / "states.jsonl")
    S["states"], S["states_bad"] = len(st), sbad
    enc = {}
    json_as_text = v2_states = frame_states = frame_states_ref = 0
    for r in st:
        e = r.get("enc")
        enc[e] = enc.get(e, 0) + 1
        p = r.get("payload")
        if e == "utf8" and isinstance(p, str):
            try:
                if isinstance(json.loads(p), (dict, list)):
                    json_as_text += 1
            except ValueError:
                pass
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except ValueError:
                p = None
        if isinstance(p, dict):
            if str(p.get("schema_version", "")).startswith("2"):
                v2_states += 1
            if p.get("reason") == "frame":
                frame_states += 1
                if isinstance(p.get("frame_ref"), dict):
                    frame_states_ref += 1
    S["enc"] = enc
    S["json_as_text"] = json_as_text
    S["v2_states"], S["frame_states"], S["frame_states_ref"] = v2_states, frame_states, frame_states_ref
    S["no_received_at_state"] = sum(1 for r in st if not r.get("received_at"))
    S["old_names_state"] = sum(1 for r in st if any(k in r for k in ("source", "host")))
    # 비전
    vis = {}
    vdir = run_path / "vision"
    if vdir.is_dir():
        for m in sorted(p for p in vdir.iterdir() if p.is_dir()):
            vr, vbad = read_jsonl(m / "results.jsonl")
            if not vr and not vbad:
                continue
            lag = []
            for r in vr:
                a, b = to_epoch(r.get("emitted_at")), to_epoch(r.get("timestamp"))
                if a is not None and b is not None:
                    lag.append((a - b) * 1000.0)
            vis[m.name] = {
                "lines": len(vr), "bad": vbad,
                "v2": sum(1 for r in vr if r.get("schema_version") == "2.0"),
                "emitted_at": sum(1 for r in vr if r.get("emitted_at")),
                "frame_ref": sum(1 for r in vr if isinstance(r.get("frame_ref"), dict)),
                "errors": sum(1 for r in vr if r.get("error")),
                "dets": sum(len(r.get("detections") or []) for r in vr),
                "old_names": sum(1 for r in vr if any(k in r for k in ("frame", "image_wh", "lag_ms"))),
                "lag": stats_ms(lag),
            }
    S["vision"] = vis
    S["ds_violations"] = S["no_received_at"] + S["old_names"] + S["sender_file"] + json_as_text \
        + S["no_received_at_state"] + S["old_names_state"]
    return S


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


def print_one(stream, name, S):
    g, miss, rep, back = S["seq"]
    eg, emiss, erep, eback = S["edge_seq"]
    print("── %s · %s  (%s)" % (stream, name, S["path"]))
    print("  기록   %d줄 (깨진 줄 %d) · 사진 %d장 · %s초 (%s 기준) · %s장/초"
          % (S["frames"], S["bad_lines"], S["images"], fmt(S["duration_s"]), S["time_from"], fmt(S["fps"], 2)))
    print("  DS 몫  received_at 없음 %d · 옛 이름 %d · sender._file %d · JSON 상태를 글자로 %d · 상태 received_at 없음 %d · 상태 옛 이름 %d"
          % (S["no_received_at"], S["old_names"], S["sender_file"], S["json_as_text"],
             S["no_received_at_state"], S["old_names_state"]))
    print("  장비   v2 프레임 헤더 %d줄 · 옛 모양 %d줄%s" % (
        S["sender_v2"], S["sender_old"], ("  (옛 칸 예: %s)" % ", ".join(S["old_sender_keys"])) if S["sender_old"] else ""))
    if S["sender_v2"]:
        print("         카메라 %s · 세션 %d · 번호 건너뜀 %d번(빠진 %d장) · 같은 번호 반복 %d · 거꾸로 %d"
              % (", ".join(S["cameras"]), S["sessions"], g, miss, rep, back))
        print("         받은 시각 − 장비 시각(ms): %s" % S["lat"])
    if S["edge"]:
        print("  넘기기 edge 칸 %d줄 · 넘긴 쪽 번호 건너뜀 %d번(빠진 %d장) · 반복 %d · 거꾸로 %d"
              % (S["edge"], eg, emiss, erep, eback))
        print("         이 기기 받은 시각 − 넘긴 기기 받은 시각(ms): %s" % S["edge_delay"])
    if S["states"] or S["states_bad"]:
        print("  상태   %d줄 · enc %s · v2 상태 %d · frame 상태 %d (frame_ref 있음 %d)"
              % (S["states"], json.dumps(S["enc"], ensure_ascii=False), S["v2_states"],
                 S["frame_states"], S["frame_states_ref"]))
    for m, v in S["vision"].items():
        print("  비전   %-12s %d줄 · v2 %d · emitted_at %d · frame_ref %d · 옛 이름 %d · 오류 %d · 검출 %d"
              % (m, v["lines"], v["v2"], v["emitted_at"], v["frame_ref"], v["old_names"], v["errors"], v["dets"]))
        print("         결과 − 원본 시각(ms): %s" % v["lag"])


def print_compare(stream, new_name, N, old_name, O):
    rows = [
        ("기록 줄", N["frames"], O["frames"]),
        ("사진 장", N["images"], O["images"]),
        ("받은 시간(초)", fmt(N["duration_s"]), fmt(O["duration_s"])),
        ("장/초", fmt(N["fps"], 2), fmt(O["fps"], 2)),
        ("시간 기준", N["time_from"], O["time_from"]),
        ("v2 프레임 헤더 줄", N["sender_v2"], O["sender_v2"]),
        ("상태 줄", N["states"], O["states"]),
        ("상태 enc", json.dumps(N["enc"]), json.dumps(O["enc"])),
    ]
    models = sorted(set(N["vision"]) | set(O["vision"]))
    for m in models:
        rows.append(("비전 %s 줄" % m, (N["vision"].get(m) or {}).get("lines", 0), (O["vision"].get(m) or {}).get("lines", 0)))
    print("── %s · 새 %s ↔ 옛 %s" % (stream, new_name, old_name))
    for k, a, b in rows:
        print("  %-18s %-28s %s" % (k, a, b))


def main():
    ap = argparse.ArgumentParser(description="DS 저장 폴더의 실행 하나를 소스마다 요약 (통합데이터규약 v2 시험용)")
    ap.add_argument("save", help="저장 폴더 (server_stream_multi_source.py 의 --save, 기본 그 파일 옆 stream_data)")
    ap.add_argument("--run", help="실행 이름 (YYYYMMDD_HHMMSS). 안 주면 소스마다 가장 최근")
    ap.add_argument("--compare", help="나란히 볼 옛 실행 이름 (예: 20261008_161514)")
    ap.add_argument("--streams", nargs="+", default=list(STREAMS), help="볼 소스 (기본 넷 다)")
    a = ap.parse_args()
    save = Path(os.path.expanduser(a.save)).resolve()
    if not save.is_dir():
        print("저장 폴더가 없다: %s" % save)
        return 2
    found = 0
    viol = 0
    for stream in a.streams:
        if stream not in ALIASES:
            print("모르는 소스: %s (cam360 · drone · robot1 · robot2)" % stream)
            return 2
        runs = dict(run_dirs(save, stream))
        if not runs:
            continue
        name = a.run or sorted(runs)[-1]
        if name not in runs:
            print("── %s · 실행 %s 없음 (있는 것: %s)" % (stream, name, ", ".join(sorted(runs)[-5:])))
            continue
        found += 1
        N = summarize(runs[name], save)
        print_one(stream, name, N)
        viol += N["ds_violations"]
        if a.compare:
            if a.compare in runs:
                print_compare(stream, name, N, a.compare, summarize(runs[a.compare], save))
            else:
                print("  (옛 실행 %s 은 %s 에 없다)" % (a.compare, stream))
        print()
    if not found:
        print("읽을 실행이 없다: %s" % save)
        return 2
    print("DS 몫 위반 합계 %d — %s" % (viol, "v2 DS 기록 모양 정상" if viol == 0 else "v2 DS 가 아닌 기록이 섞였다 (옛 판 DS 로 쌓은 실행이거나 파일이 안 바뀜)"))
    return 0 if viol == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
