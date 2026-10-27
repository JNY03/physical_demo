#!/usr/bin/env python3
"""통합데이터규약 v2 점검 — DS 기록(jsonl)이 v2 모양인지 본다. 표준 라이브러리만 쓴다.

쓰는 법
  python3 check_v2.py <실행 폴더 또는 jsonl 파일> [...]
    폴더를 주면 그 아래 frames.jsonl · states.jsonl · vision/*/results.jsonl 을 모두 본다.
    파일 이름에 frames / states / results 가 들어 있으면 그 종류로 읽는다.
  --max N   파일마다 앞에서 N줄만 본다 (기본 0 = 전부)

끝 코드  0 = 위반 없음, 1 = 위반 있음, 2 = 읽을 파일 없음
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ISO = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+09:00$")
VER = re.compile(r"^[0-9]+\.[0-9]+$")
HEAD = ("schema_version", "source_id", "node_id", "zone_id", "timestamp")
# v2 에서 없어진 이름 (어디에 있어도 위반)
OLD_FRAME = {"ts", "seq", "frame_seq", "captured_at", "observed_at", "frame_wh",
             "capture_timestamp", "captured_at_unix"}
OLD_RESULT = {"xyxy", "conf", "cls", "name", "image_wh", "recv_ms", "done_ms", "lag_ms", "header",
              "frame_ts", "weight", "checkpoint", "bbox_xyxy", "box", "boxes", "bbox_space", "source"}
OLD_RECORD = {"host", "file", "source"}


class Report:
    def __init__(self, path):
        self.path, self.n = path, 0
        self.bad = collections.Counter()
        self.example = {}
        self.note = collections.Counter()

    def fail(self, key, line_no, detail=""):
        self.bad[key] += 1
        self.example.setdefault(key, "%d줄 %s" % (line_no, detail[:120]))

    def info(self, key):
        self.note[key] += 1

    def print(self):
        print("\n== %s (%d줄)" % (self.path, self.n))
        if not self.bad:
            print("   위반 없음")
        for k, c in self.bad.most_common():
            print("   [위반] %-46s %6d건   예: %s" % (k, c, self.example[k]))
        for k, c in self.note.most_common():
            print("   [참고] %-46s %6d건" % (k, c))


def check_head(rep, i, obj, where, need_seq):
    for k in HEAD:
        if k not in obj:
            rep.fail("%s: 공통 머리 %s 없음" % (where, k), i)
    v = obj.get("schema_version")
    if v is not None and not (isinstance(v, str) and VER.match(v)):
        rep.fail("%s: schema_version 형식" % where, i, repr(v))
    elif v is not None and not v.startswith("2."):
        rep.fail("%s: schema_version 이 2.x 가 아님" % where, i, repr(v))
    t = obj.get("timestamp")
    if t is not None and not (isinstance(t, str) and ISO.match(t)):
        rep.fail("%s: timestamp 형식 (ISO 밀리초 +09:00)" % where, i, repr(t))
    if need_seq and not isinstance(obj.get("sequence_id"), int):
        rep.fail("%s: sequence_id 가 정수가 아님/없음" % where, i, repr(obj.get("sequence_id")))
    if "sent_at" in obj and not ISO.match(str(obj["sent_at"])):
        rep.fail("%s: sent_at 형식" % where, i, repr(obj["sent_at"]))


def check_frame_ref(rep, i, obj, where):
    fr = obj.get("frame_ref")
    if fr is None:
        return
    if not isinstance(fr, dict) or set(fr) != {"camera_id", "sequence_id", "timestamp"}:
        rep.fail("%s: frame_ref 모양 (camera_id·sequence_id·timestamp 셋만)" % where, i, repr(fr))
        return
    if not ISO.match(str(fr["timestamp"])):
        rep.fail("%s: frame_ref.timestamp 형식" % where, i, repr(fr["timestamp"]))


def check_frames(path, limit):
    rep = Report(path)
    seqs = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            rep.n += 1
            try:
                rec = json.loads(line)
            except ValueError:
                rep.fail("JSON 아님", i)
                continue
            for k in OLD_RECORD & set(rec):
                rep.fail("기록: 옛 이름 %s" % k, i)
            for k in ("received_at", "received_by", "frame_file", "stream", "sender"):
                if k not in rec:
                    rep.fail("기록: %s 없음" % k, i)
            if "received_at" in rec and not ISO.match(str(rec["received_at"])):
                rep.fail("기록: received_at 형식", i, repr(rec["received_at"]))
            s = rec.get("sender") or {}
            if any(str(k).startswith("X-") for k in s):
                rep.fail("프레임: X-* 헤더가 남음 (Frame-Meta 미적용)", i)
            for k in OLD_FRAME & set(s):
                rep.fail("프레임: 옛 이름 %s" % k, i)
            if "_file" in s:
                rep.fail("프레임: DS 가 붙인 _file 이 남음 (P2)", i)
            check_head(rep, i, s, "프레임", need_seq=True)
            for k in ("camera_id", "encoding", "width", "height", "session_id"):
                if k not in s:
                    rep.fail("프레임: %s 없음" % k, i)
            for k in ("width", "height"):
                if k in s and not isinstance(s[k], int):
                    rep.fail("프레임: %s 가 정수가 아님" % k, i, repr(s[k]))
            if s.get("encoding") == "h264":
                for k in ("keyframe", "codec"):
                    if k not in s:
                        rep.fail("프레임(h264): %s 없음" % k, i)
            key = (s.get("camera_id"), s.get("session_id"))
            q = s.get("sequence_id")
            if isinstance(q, int):
                last = seqs.get(key)
                if last is not None:
                    if q <= last:
                        rep.info("순번 역전·중복 (같은 camera_id·session_id)")
                    elif q > last + 1:
                        rep.info("순번 건너뜀 (받는 쪽에서 빠진 장이 있음)")
                seqs[key] = q
    return rep


def check_states(path, limit):
    rep = Report(path)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            rep.n += 1
            try:
                rec = json.loads(line)
            except ValueError:
                rep.fail("JSON 아님", i)
                continue
            for k in OLD_RECORD & set(rec):
                rep.fail("기록: 옛 이름 %s" % k, i)
            for k in ("received_at", "received_by", "stream", "topic", "retain"):
                if k not in rec:
                    rep.fail("기록: %s 없음" % k, i)
            p = rec.get("payload")
            if isinstance(p, str):
                if rec.get("enc") == "json":
                    rep.fail("기록: enc=json 인데 payload 가 글자", i)
                try:
                    p = json.loads(p)
                    rep.fail("기록: JSON payload 를 글자로 저장 (객체로 저장해야 함)", i)
                except ValueError:
                    continue
            if not isinstance(p, dict):
                continue
            topic = str(rec.get("topic", ""))
            channel = topic.rsplit("/", 1)[-1]
            if channel not in ("state", "status", "heartbeat") or "/sim/" in topic:
                rep.info("세 채널 밖 토픽 (%s)" % channel)
                continue
            check_head(rep, i, p, channel, need_seq=channel in ("state", "heartbeat"))
            if "ts" in p:
                rep.fail("%s: 옛 칸 ts (frame_ref 로)" % channel, i)
            check_frame_ref(rep, i, p, channel)
            fr = p.get("frame_ref")
            if isinstance(fr, dict) and fr.get("timestamp") != p.get("timestamp"):
                rep.fail("%s: timestamp ≠ frame_ref.timestamp (D13 같은 값 규칙)" % channel, i)
            if p.get("reason") == "frame" and "frame_ref" not in p:
                rep.fail("%s: reason=frame 인데 frame_ref 없음" % channel, i)
            for path_ in (("battery", "pct"), ("velocity", "speed_mps"), ("pose", "yaw_deg"),
                          ("freshness", "bms_age_s"), ("freshness", "sampled_at")):
                d = p.get(path_[0])
                if isinstance(d, dict) and path_[1] in d:
                    rep.fail("%s: 지운 중복 칸 %s.%s" % (channel, path_[0], path_[1]), i)
            if "origin" in p:
                rep.fail("%s: origin 이 남음 (status 의 device_info 로)" % channel, i)
            fl = p.get("flight")
            if isinstance(fl, dict) and "system_status" in fl:
                rep.fail("%s: flight.system_status 가 남음 (P3)" % channel, i)
    return rep


def check_results(path, limit):
    rep = Report(path)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            rep.n += 1
            try:
                r = json.loads(line)
            except ValueError:
                rep.fail("JSON 아님", i)
                continue
            check_head(rep, i, r, "결과", need_seq=False)
            for k in OLD_RESULT & set(r):
                rep.fail("결과: 옛 이름 %s" % k, i)
            for k in ("frame_ref", "emitted_at", "width", "height", "detections", "model"):
                if k not in r:
                    rep.fail("결과: %s 없음" % k, i)
            check_frame_ref(rep, i, r, "결과")
            fr = r.get("frame_ref")
            if isinstance(fr, dict) and fr.get("timestamp") != r.get("timestamp"):
                rep.fail("결과: timestamp ≠ frame_ref.timestamp (D5·D13)", i)
            if "emitted_at" in r and not ISO.match(str(r["emitted_at"])):
                rep.fail("결과: emitted_at 형식", i, repr(r["emitted_at"]))
            for d in r.get("detections") or []:
                for k in OLD_RESULT & set(d):
                    rep.fail("검출: 옛 이름 %s" % k, i)
                b = d.get("bbox")
                if not (isinstance(b, list) and len(b) == 4):
                    rep.fail("검출: bbox [x1,y1,x2,y2] 아님", i, repr(b)[:60])
                elif not (b[0] <= b[2] and b[1] <= b[3]):
                    rep.fail("검출: bbox 순서 (x1<=x2, y1<=y2)", i, repr(b))
                if "label" not in d:
                    rep.fail("검출: label 없음", i)
    return rep


def find(paths):
    out = []
    for a in paths:
        p = Path(a)
        if p.is_dir():
            out += sorted(p.rglob("frames.jsonl")) + sorted(p.rglob("states.jsonl")) \
                + sorted(p.rglob("results.jsonl"))
        elif p.is_file():
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="통합데이터규약 v2 점검")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    files = find(a.paths)
    if not files:
        print("읽을 jsonl 이 없다")
        return 2
    bad = 0
    for p in files:
        name = p.name
        if "results" in name:
            rep = check_results(p, a.max)
        elif "states" in name:
            rep = check_states(p, a.max)
        elif "frames" in name:
            rep = check_frames(p, a.max)
        else:
            print("\n== %s — 종류를 모르겠다 (이름에 frames/states/results 가 없음)" % p)
            continue
        rep.print()
        bad += sum(rep.bad.values())
    print("\n합계 위반 %d건" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
