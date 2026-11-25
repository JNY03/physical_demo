#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""contract_check.py — 가시화가 쓰는 data_stream 계약의 「모양」을 찍고 비교한다 (Docker 계획 D4 · D9, §11).

가시화(다른 저장소, 고치지 않는다)는 data_stream 에 두 길로만 붙는다.
  HTTP   소스 포트 10001~10012 — /frame(?since · &format=json · 머리 X-State · X-Frame-N · X-Frame-Meta) · /stream ·
         /state · /vision · /vision/frame · /health · /control/roi · /control/reid · /scan/*
  Redis  :6380 — camera_stream:<소스> · state_stream:<소스> · vision_stream:<소스>:<모델> · vision_depth:<소스>:<모델> ·
         orchestration_stream · node_resource_stream
컨테이너로 바꿔도 이 둘이 같아야 한다. 이 도구는 **값이 아니라 모양만** 남긴다 — 포트가 답하는지 · 상태 코드 ·
머리 이름 · Content-Type · JSON 키 구조(키와 타입, 재귀) · Redis 키 이름(소스 자리는 *) · 스트림 칸 이름과 칸의 타입.
그림 · 값 · IP · 기기 이름은 남기지 않는다(마지막에 한 번 더 걸러 내고, 남아 있으면 쓰지 않고 멈춘다).

명령
  capture --out 파일.json   지금 떠 있는 data_stream 을 찍는다 (읽기만 — 부작용 없는 GET 과 틀린 몸의 POST 하나)
  compare 기준.json        다시 찍어 기준과 비교한다. 규칙은 「칸 추가만」:
                            새 키 · 새 칸 · 새 머리 · 새 포트 = 경고(통과), 없어진 것 · 타입이 바뀐 것 · 이름이 바뀐 것 = 실패
                            끝 코드: 0 통과 · 1 실패 · 2 찍지 못함
  feed                     시험용 가짜 프레임 · 상태를 포트마다 넣는다 (/upload · /upload_pano · POST /state).
                            **쓰기다** — 파이가 꺼진 시험 서버(--no-pull, 버리는 Redis)에만 쓴다. --yes 가 있어야 돈다.
  seed --to URL            --from(기본 6380, 읽기만)의 vision_stream · vision_depth · orchestration_stream ·
                            node_resource_stream 마지막 몇 건을 --to(버리는 Redis)로 복사한다 — GPU 없이 비전 칸을 시험한다.

기준(tools/contract/baseline_current.json)을 만든 순서 — 컨테이너 판도 똑같이 해서 compare 한다
  redis-server --port 6390 --bind 127.0.0.1 --save '' --appendonly no --daemonize yes   # 버리는 Redis
  python tools/contract_check.py seed --to redis://127.0.0.1:6390/0 --yes                # 6380 은 읽기만
  python server_stream_multi_source.py --no-pull --redis redis://127.0.0.1:6390/0 --save <임시 폴더> --redis-maxmemory off
  python tools/contract_check.py feed --yes
  python tools/contract_check.py capture --redis redis://127.0.0.1:6390/0 --out tools/contract/baseline_current.json
  (컨테이너 판) 같은 Redis 조건 · 같은 seed · feed 뒤에
  python tools/contract_check.py compare tools/contract/baseline_current.json --redis redis://127.0.0.1:6390/0
운영 중인 기기에서 바로 비교할 때는 같은 소스 · 같은 비전 모델이 돌 때 찍은 기준끼리 비교한다 —
표본에 없던 소스 · 모델은 「키가 없음」으로 나온다(소스 인스턴스는 경고, 스트림 무늬 자체가 없으면 실패).

필요한 것: 표준 라이브러리 + redis 패키지(없으면 Redis 쪽만 건너뛴다).
"""
from __future__ import annotations

import argparse
import fnmatch
import http.client
import json
import re
import socket
import sys
import time
import uuid

try:
    import redis
except ImportError:  # pragma: no cover — core 의존성이지만 없으면 HTTP 만 찍는다
    redis = None

FORMAT = "contract_check/1"
DEFAULT_PORTS = (10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10011, 10012)
DEFAULT_REDIS = "redis://127.0.0.1:6380/0"
STREAM_SAMPLE = 5                  # 키마다 읽는 마지막 항목 수
WIDE_SAMPLE_FACTOR = 20            # orchestration_stream · node_resource_stream 은 이만큼 곱해 본다
HTTP_TIMEOUT = 5.0

#: 키가 값(토픽 · 모델 · 상태 이름 · 노드 이름)인 사전 — 키 집합이 아니라 「값의 모양」만 비교한다.
#: 경로는 문서 이름 + ".키" + "[]"(목록 원소). fnmatch 로 맞춘다.
MAP_PATHS = (
    "/health.status",              # video · upload · state … 일어난 일에 따라 생긴다
    "/health.vision",              # 돌고 있는 모델 이름
    "/state.by_topic",             # MQTT 토픽
    "*.by_topic",
    "*.counts",                    # 클래스 이름별 개수
    "*.cues",                      # 대상 클래스별 근거
    "*.verify_by",                 # 클래스별 확인 결과
    "*.cards",                     # 클래스별 카드
    "*.votes",                     # 라벨별 표
    "*.fps",                       # 소스별 처리 속도 (orchestration detail · plan)
    "*.fps_cap",
)

IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
NUMERIC = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


# ---------------------------------------------------------------- 모양(스키마)
# 마디 = {"types": [...], "keys": {이름: 마디}, "items": 마디, "values": 마디, "optional": true}
#   optional: 표본 가운데 일부에만 있던 키 · 칸 — 다음에 안 보여도 실패가 아니라 경고다
#             (늘 있던 키가 없어지면 실패). 표본 수가 적으면 늘 있던 것으로 보일 수 있다.
#   types: object · map · list · str · num · bool · null · bytes · text
#   null 은 「값이 없어 모름」이다 — 비교할 때 어느 타입과도 맞는 것으로 친다.

def _is_map(path, d):
    if any(fnmatch.fnmatchcase(path, p) for p in MAP_PATHS):
        return True
    keys = list(d)
    # 계약 키에는 / · 빈칸 · IP · 숫자만인 이름이 없다 — 그런 키면 값(토픽 · 라벨)이 키 자리에 온 사전이다
    return bool(keys) and all(("/" in k) or (" " in k) or IPV4.search(k) or k.isdigit() for k in keys)


def schema_of(v, path="$"):
    if v is None:
        return {"types": ["null"]}
    if isinstance(v, bool):
        return {"types": ["bool"]}
    if isinstance(v, (int, float)):
        return {"types": ["num"]}
    if isinstance(v, str):
        return {"types": ["str"]}
    if isinstance(v, (bytes, bytearray)):
        return {"types": ["bytes"]}
    if isinstance(v, dict):
        if _is_map(path, v):
            vals = None
            for k, x in v.items():
                vals = merge(vals, schema_of(x, path + ".*"))
            return {"types": ["map"], "values": vals}
        return {"types": ["object"], "keys": {str(k): schema_of(x, "%s.%s" % (path, k)) for k, x in v.items()}}
    if isinstance(v, (list, tuple)):
        items = None
        for x in v:
            items = merge(items, schema_of(x, path + "[]"))
        return {"types": ["list"], "items": items}
    return {"types": [type(v).__name__]}


def merge(a, b):
    """두 모양의 합(여러 표본 · 여러 인스턴스를 하나로)."""
    if a is None:
        return b
    if b is None:
        return a
    out = {"types": sorted(set(a["types"]) | set(b["types"]))}
    if a.get("optional") or b.get("optional"):
        out["optional"] = True
    if "keys" in a or "keys" in b:
        ka, kb = a.get("keys", {}), b.get("keys", {})
        both = "keys" in a and "keys" in b         # 둘 다 객체인데 한쪽에만 있는 키 = 늘 오지는 않는 키
        out["keys"] = {}
        for k in sorted(set(ka) | set(kb)):
            m = merge(ka.get(k), kb.get(k))
            if both and (k not in ka or k not in kb):
                m = dict(m, optional=True)
            out["keys"][k] = m
    for f in ("items", "values"):
        if f in a or f in b:
            out[f] = merge(a.get(f), b.get(f))
    return out


def text_schema(raw, path):
    """Redis 칸 · 머리 값(바이트 또는 글자) → 모양. JSON 이면 안까지 본다."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            return {"types": ["bytes"]}
    s = raw.strip()
    if s == "":
        return {"types": ["null"]}
    if NUMERIC.match(s):
        return {"types": ["num"]}
    if s[0] in "{[":
        try:
            return schema_of(json.loads(s), path)
        except ValueError:
            pass
    if s in ("null",):
        return {"types": ["null"]}
    if s in ("true", "false"):
        return {"types": ["bool"]}
    return {"types": ["str"]}


# ---------------------------------------------------------------- HTTP 찍기
def _norm_ctype(v):
    return ";".join(p.strip() for p in (v or "").lower().split(";")) if v else None


def _http(host, port, method, path, body=None, headers=None, timeout=HTTP_TIMEOUT):
    """→ (상태, [(머리, 값)], 몸). multipart(MJPEG)는 끝이 없으니 첫 칸 머리까지만 읽는다."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        hdrs = list(resp.getheaders())
        if not (resp.getheader("Content-Type") or "").lower().startswith("multipart/"):
            return resp.status, hdrs, resp.read()
        part = []
        try:                                   # 연결 timeout 이 읽기에도 걸린다
            line = resp.fp.readline(4096)
            while line and len(part) < 12:
                part.append(line)
                if line.strip() == b"" and len(part) > 1:
                    break
                line = resp.fp.readline(4096)
        except (socket.timeout, OSError):
            pass
        return resp.status, hdrs, b"".join(part)
    finally:
        conn.close()


def _endpoint_record(status, hdrs, data, doc):
    """응답 하나의 모양 — 몸은 Content-Type 으로 읽는다(JSON → 키 구조, multipart → 첫 칸 머리 이름)."""
    hmap = {k.lower(): v for k, v in hdrs}
    ctype = _norm_ctype(hmap.get("content-type"))
    rec = {"status": status, "content_type": ctype, "headers": sorted(hmap)}
    if ctype and ctype.startswith("application/json"):
        try:
            rec["json"] = schema_of(json.loads(data.decode("utf-8")), doc)
        except (ValueError, UnicodeDecodeError):
            rec["json"] = {"types": ["text"]}
    elif ctype and ctype.startswith("multipart/"):
        part = [ln.split(b":", 1)[0].strip().decode("latin-1").lower()
                for ln in data.splitlines() if b":" in ln and not ln.startswith(b"--")]
        rec["first_part_headers"] = sorted(set(part)) if part else None
    for h, label in (("x-state", "X-State"), ("x-frame-meta", "X-Frame-Meta"), ("x-frame-n", "X-Frame-N")):
        if h in hmap:
            rec.setdefault("header_json", {})[h] = text_schema(hmap[h], label)
    return rec


#: (기록 이름, 메서드, 경로, 몸, 문서 이름) — 부작용 없는 요청만. POST 는 검사에서 걸러지는 틀린 몸이다.
PROBES = (
    ("GET /state", "GET", "/state", None, "/state"),
    ("GET /frame?format=json", "GET", "/frame?wait=0&format=json", None, "/frame?format=json"),
    ("OPTIONS /frame", "OPTIONS", "/frame", None, "/frame"),
    ("GET /stream", "GET", "/stream", None, "/stream"),
    ("GET /vision", "GET", "/vision", None, "/vision"),
    ("GET /vision/frame", "GET", "/vision/frame?image=0&wait=0", None, "/vision/frame"),
    ("GET /vision/frame?image=1", "GET", "/vision/frame?wait=0", None, "/vision/frame"),
    ("GET /vision/view", "GET", "/vision/view", None, "/vision/view"),
    ("GET /", "GET", "/", None, "/"),
    ("GET /control/roi", "GET", "/control/roi", None, "/control/roi"),
    ("GET /control/reid", "GET", "/control/reid", None, "/control/reid"),
    ("GET /scan/status", "GET", "/scan/status", None, "/scan/*"),
    ("GET /scan/result", "GET", "/scan/result?scan_id=contract-check-none", None, "/scan/*"),
    ("GET /scan/frame", "GET", "/scan/frame?scan_id=contract-check-none&rotation_deg=0", None, "/scan/*"),
    ("POST /scan/prepare (bad body)", "POST", "/scan/prepare", b"{}", "/scan/*"),
    ("GET /no-such-path", "GET", "/contract-check-no-such-path", None, "/404"),
)


def probe_port(host, port):
    """한 포트의 계약 → ({기록 이름: 모양}, 소스 이름). 답이 없으면 (None, None)."""
    out = {}

    def get(key, method, path, body=None, doc=None):
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))} if body is not None else None
        try:
            r = _http(host, port, method, path, body=body, headers=headers)
        except (OSError, http.client.HTTPException) as e:
            out[key] = {"error": type(e).__name__}
            return None
        out[key] = _endpoint_record(*r, doc or path.split("?")[0])
        return r

    r = get("GET /health", "GET", "/health", doc="/health")
    if r is None:
        return None, None
    try:
        src = json.loads(r[2].decode("utf-8")).get("source")      # 포트 = 소스 — 이 값만은 계약이다
    except (ValueError, UnicodeDecodeError, AttributeError):
        src = None
    r = get("GET /frame", "GET", "/frame?wait=0", doc="/frame")
    n = dict((k.lower(), v) for k, v in r[1]).get("x-frame-n") if r and r[0] == 200 else None
    if n is not None:                          # 받은 번호를 since 로 — 새 장이 없으면 204
        get("GET /frame?since=<n>", "GET", "/frame?since=%s&wait=0" % n, doc="/frame")
    for key, method, path, body, doc in PROBES:
        get(key, method, path, body, doc)
    return out, src


def capture_http(host, ports):
    res = {}
    for p in ports:
        rec, src = probe_port(host, p)
        res[str(p)] = {"answers": False} if rec is None else {"answers": True, "source": src, "endpoints": rec}
    return res


# ---------------------------------------------------------------- Redis 찍기
STREAM_FAMILIES_FAIL = ("camera_stream", "state_stream", "vision_stream", "vision_depth",
                        "orchestration_stream", "node_resource_stream")


def key_pattern(key):
    """camera_stream:robot1 → camera_stream:* · vision_stream:robot1:yoloe → vision_stream:*:yoloe
    (둘째 마디 = 소스 · scan_id 자리. 모델 이름은 모양을 가르므로 남긴다)"""
    seg = key.split(":")
    if len(seg) >= 2:
        seg[1] = "*"
    return ":".join(seg)


def capture_redis(client, sample=STREAM_SAMPLE):
    pats = {}
    for k in client.scan_iter(count=500):
        key = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
        pat = key_pattern(key)
        t = client.type(k)
        t = t.decode() if isinstance(t, bytes) else str(t)
        p = pats.setdefault(pat, {"type": t, "instances": []})
        if p["type"] != t:
            p["type"] = "mixed"
        if len(seg := key.split(":")) >= 2:
            p["instances"].append(seg[1])
        doc = "redis:" + pat
        if t == "stream":
            fields = p.setdefault("fields", {})
            seen = p.setdefault("_seen", {})
            # 소스 자리가 없는 스트림(orchestration · node)은 한 건마다 event · change 가 달라 더 많이 본다
            count = sample if ":" in key else sample * WIDE_SAMPLE_FACTOR
            for _eid, f in client.xrevrange(k, "+", "-", count=count):
                p["_entries"] = p.get("_entries", 0) + 1
                for fk, fv in f.items():
                    name = fk.decode("utf-8", "replace") if isinstance(fk, bytes) else str(fk)
                    fields[name] = merge(fields.get(name), text_schema(fv, doc + "#" + name))
                    seen[name] = seen.get(name, 0) + 1
        elif t == "string":
            p["value"] = merge(p.get("value"), text_schema(client.get(k) or b"", doc))
    for p in pats.values():
        p["instances"] = sorted(set(p["instances"]))
        seen, entries = p.pop("_seen", {}), p.pop("_entries", 0)
        if "fields" in p:
            p["fields"] = {k: (dict(v, optional=True) if seen.get(k, 0) < entries else v)
                           for k, v in sorted(p["fields"].items())}
    return dict(sorted(pats.items()))


# ---------------------------------------------------------------- 걸러 내기 (IP · 기기 이름)
def _hostnames():
    names = set()
    for n in (socket.gethostname(), socket.getfqdn()):
        if n and n not in ("localhost",) and len(n) >= 3:
            names.add(n.lower())
            names.add(n.split(".")[0].lower())
    return names


def scrub(obj, extra_names=()):
    """키 · 값 글자에서 IPv4 와 이 기기 이름을 <ip> · <host> 로 바꾼다."""
    names = sorted(_hostnames() | {x.lower() for x in extra_names if x}, key=len, reverse=True)

    def s(x):
        x = IPV4.sub("<ip>", x)
        low = x.lower()
        for n in names:
            i = low.find(n)
            while i >= 0:
                x = x[:i] + "<host>" + x[i + len(n):]
                low = x.lower()
                i = low.find(n)
        return x

    def walk(v):
        if isinstance(v, dict):
            return {s(str(k)): walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            return s(v)
        return v
    return walk(obj)


def leaks(obj, extra_names=()):
    text = json.dumps(obj, ensure_ascii=False)
    found = set(IPV4.findall(text))
    low = text.lower()
    found |= {n for n in (_hostnames() | {x.lower() for x in extra_names if x}) if n in low}
    return sorted(found)


def capture(host="127.0.0.1", ports=DEFAULT_PORTS, redis_url=DEFAULT_REDIS, sample=STREAM_SAMPLE,
            no_redis=False, redis_client=None):
    snap = {"format": FORMAT, "ports": [int(p) for p in ports],
            "http": capture_http(host, ports), "redis": None, "redis_error": None}
    if not no_redis:
        if redis_client is None and redis is None:
            snap["redis_error"] = "redis 패키지 없음"
        else:
            try:
                cli = redis_client or redis.Redis.from_url(redis_url, socket_connect_timeout=3, socket_timeout=10)
                snap["redis"] = capture_redis(cli, sample)
            except Exception as e:  # noqa: BLE001
                snap["redis_error"] = type(e).__name__
    extra = [host] if host not in ("127.0.0.1", "localhost", "::1") else []
    snap = scrub(snap, extra)
    left = leaks(snap, extra)
    if left:
        raise RuntimeError("걸러 내지 못한 주소 · 이름이 남았다: %s" % ", ".join(left))
    return snap


# ---------------------------------------------------------------- 비교 (칸 추가만)
class Report:
    def __init__(self):
        self.fails, self.warns = [], []

    def fail(self, where, msg):
        self.fails.append("%s: %s" % (where, msg))

    def warn(self, where, msg):
        self.warns.append("%s: %s" % (where, msg))


def compare_schema(base, new, where, rep):
    if base is None:
        return
    if new is None:
        rep.fail(where, "없어짐")
        return
    bt, nt = set(base["types"]) - {"null"}, set(new["types"]) - {"null"}
    if bt and not nt:
        rep.warn(where, "값이 비어 모양을 확인 못함 (기준 %s)" % "|".join(sorted(bt)))
        return
    if bt and nt:
        if not (bt & nt):
            rep.fail(where, "타입 바뀜 %s → %s" % ("|".join(sorted(bt)), "|".join(sorted(nt))))
            return
        if bt - nt:
            rep.warn(where, "이번 표본에 없는 타입 %s" % "|".join(sorted(bt - nt)))
        if nt - bt:
            rep.warn(where, "새 타입 %s" % "|".join(sorted(nt - bt)))
    if "keys" in base and "object" in nt:
        nk = new.get("keys", {})
        for k, sub in base["keys"].items():
            if k not in nk:
                if sub and sub.get("optional"):
                    rep.warn("%s.%s" % (where, k), "늘 오지는 않던 키 — 이번 표본에 없음")
                else:
                    rep.fail("%s.%s" % (where, k), "키 없어짐")
            else:
                compare_schema(sub, nk[k], "%s.%s" % (where, k), rep)
        for k in nk:
            if k not in base["keys"]:
                rep.warn("%s.%s" % (where, k), "새 키 (추가)")
    for f, mark in (("items", "[]"), ("values", ".*")):
        if base.get(f) is not None and new.get(f) is not None:
            compare_schema(base[f], new[f], where + mark, rep)


def compare(base, new):
    rep = Report()
    if base.get("format") != FORMAT:
        rep.fail("format", "기준 파일 형식이 다름 (%s)" % base.get("format"))
    bh, nh = base.get("http") or {}, new.get("http") or {}
    for port, b in bh.items():
        n = nh.get(port)
        w = ":" + port
        if n is None:
            rep.fail(w, "이번에 찍지 않은 포트 (--ports 확인)")
            continue
        if b.get("answers") and not n.get("answers"):
            rep.fail(w, "포트가 답하지 않음")
            continue
        if not b.get("answers"):
            if n.get("answers"):
                rep.warn(w, "새로 답하는 포트 (추가)")
            continue
        if b.get("source") != n.get("source"):
            rep.fail(w, "소스가 바뀜 %s → %s" % (b.get("source"), n.get("source")))
        be, ne = b.get("endpoints", {}), n.get("endpoints", {})
        for ep, br in be.items():
            ww = "%s %s" % (w, ep)
            nr = ne.get(ep)
            if nr is None:
                if ep == "GET /frame?since=<n>" and ne.get("GET /frame", {}).get("status") != 200:
                    rep.warn(ww, "프레임이 없어 since 를 시험 못함")
                else:
                    rep.fail(ww, "없어짐")
                continue
            if "error" in nr and "error" not in br:
                rep.fail(ww, "요청 실패 (%s)" % nr["error"])
                continue
            if br.get("status") != nr.get("status"):
                rep.fail(ww, "상태 코드 %s → %s" % (br.get("status"), nr.get("status")))
                continue
            if br.get("content_type") != nr.get("content_type"):
                rep.fail(ww, "Content-Type %s → %s" % (br.get("content_type"), nr.get("content_type")))
            bhd, nhd = set(br.get("headers") or []), set(nr.get("headers") or [])
            for h in sorted(bhd - nhd):
                rep.fail(ww, "머리 없어짐: %s" % h)
            for h in sorted(nhd - bhd):
                rep.warn(ww, "새 머리 (추가): %s" % h)
            if br.get("first_part_headers") and nr.get("first_part_headers") is not None:
                for h in sorted(set(br["first_part_headers"]) - set(nr["first_part_headers"])):
                    rep.fail(ww, "MJPEG 칸 머리 없어짐: %s" % h)
            if br.get("json") is not None:
                compare_schema(br["json"], nr.get("json"), ww + " json", rep)
            for h, s in (br.get("header_json") or {}).items():
                compare_schema(s, (nr.get("header_json") or {}).get(h), "%s %s" % (ww, h), rep)
        for ep in ne:
            if ep not in be:
                rep.warn("%s %s" % (w, ep), "새 주소 (추가)")
    for port in nh:
        if port not in bh and nh[port].get("answers"):
            rep.warn(":" + port, "새 포트 (추가)")

    br_, nr_ = base.get("redis"), new.get("redis")
    if br_ is not None:
        if nr_ is None:
            rep.fail("redis", "못 읽음 (%s)" % new.get("redis_error"))
        else:
            for pat, b in br_.items():
                fam = pat.split(":")[0]
                strict = b.get("type") == "stream" and fam in STREAM_FAMILIES_FAIL
                w = "redis " + pat
                n = nr_.get(pat)
                if n is None:
                    (rep.fail if strict else rep.warn)(w, "키가 없음")
                    continue
                if b["type"] != n["type"]:
                    rep.fail(w, "키 타입 %s → %s" % (b["type"], n["type"]))
                    continue
                for inst in sorted(set(b.get("instances", [])) - set(n.get("instances", []))):
                    rep.warn(w, "이번에 없는 인스턴스 %s" % inst)
                bf, nf = b.get("fields") or {}, n.get("fields") or {}
                for f, s in bf.items():
                    if f not in nf:
                        if s and s.get("optional"):
                            rep.warn("%s #%s" % (w, f), "늘 오지는 않던 칸 — 이번 표본에 없음")
                        else:
                            (rep.fail if strict else rep.warn)("%s #%s" % (w, f), "칸 없어짐")
                    else:
                        compare_schema(s, nf[f], "%s #%s" % (w, f), rep)
                for f in nf:
                    if f not in bf:
                        rep.warn("%s #%s" % (w, f), "새 칸 (추가)")
                if "value" in b:
                    compare_schema(b["value"], n.get("value"), w + " 값", rep)
            for pat in nr_:
                if pat not in br_:
                    rep.warn("redis " + pat, "새 키 (추가)")
    return rep


# ---------------------------------------------------------------- 시험용 넣기 (쓰기)
#: 8x8 검정 JPEG — 서버는 풀지 않고 그대로 나른다
TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsK"
    "DAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAIAAgD"
    "ASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKB"
    "kaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZ"
    "mqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQF"
    "BgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX"
    "2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigD/2Q==")

#: 포트 → (올리는 길, 상태 payload). 상태 payload 모양은 기기 쪽 것을 흉내 낸 고정값이다(비교가 재현되게).
CAM360_STATE = {"gps": {"lat": 37.5, "lon": 127.0, "alt_m": 40.0, "fix_type": 3, "satellites": 9},
                "battery": {"level_pct": 80, "charging": False},
                "temperature_c": {"battery": 30.0, "board": 40.0, "module": 45.0}}
ROBOT_STATE = {"device_id": "contract-check", "battery": {"level_pct": 70}, "position": {"x": 1.0, "y": 2.0},
               "ts": "2026-01-01T00:00:00.000+09:00"}


def _multipart(fields, image):
    b = "cc" + uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode())
    out.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f.jpg\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n" % b).encode() + image + b"\r\n")
    out.append(("--%s--\r\n" % b).encode())
    return "multipart/form-data; boundary=" + b, b"".join(out)


def feed(host, ports, frames=2):
    import base64
    img = base64.b64decode(TINY_JPEG_B64)
    done = {}
    for p in ports:
        path = "/upload_pano" if p == 10001 else "/upload"
        ok = 0
        try:
            for i in range(frames):
                ctype, body = _multipart({"camera_id": "contract-check-%d" % p, "sequence_id": str(i + 1),
                                          "timestamp": "2026-01-01T00:00:0%d.000+09:00" % i}, img)
                st, _, _ = _http(host, p, "POST", path, body=body,
                                 headers={"Content-Type": ctype, "Content-Length": str(len(body))})
                ok += st == 200
            payload = CAM360_STATE if p == 10001 else ROBOT_STATE
            body = json.dumps({"topic": "contract-check/state", "payload": payload}).encode()
            st, _, _ = _http(host, p, "POST", "/state", body=body,
                             headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            done[p] = "프레임 %d/%d · 상태 %s" % (ok, frames, st)
        except OSError as e:
            done[p] = "못 넣음 (%s)" % type(e).__name__
    return done


SEED_PREFIXES = ("vision_stream:", "vision_depth:")
SEED_KEYS = ("orchestration_stream", "node_resource_stream")


def seed(src, dst, per_key=3, wide=30):
    """src(읽기만)의 비전 결과 · 오케스트레이션 · 노드 스트림 마지막 몇 건을 dst(버리는 Redis)로 옮겨 적는다.
    GPU · torch 없이 /vision · /vision/frame · /health 의 비전 칸과 Redis 비전 스트림 모양을 시험하려고 쓴다."""
    copied = {}
    for k in src.scan_iter(count=500):
        key = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
        if not (key.startswith(SEED_PREFIXES) or key in SEED_KEYS):
            continue
        t = src.type(k)
        if (t.decode() if isinstance(t, bytes) else str(t)) != "stream":
            continue
        entries = src.xrevrange(k, "+", "-", count=wide if key in SEED_KEYS else per_key)
        for _eid, f in reversed(entries):
            dst.xadd(key, f)
        copied[key] = len(entries)
    return copied


# ---------------------------------------------------------------- 명령행
def _ports(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += [p for p in range(int(a), int(b) + 1) if p != 10010]   # 10010 은 비워 둔 자리
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="가시화 ↔ data_stream 계약(모양) 찍기 · 비교 (칸 추가만)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--target", default="127.0.0.1", help="data_stream 이 도는 기기 (기본 127.0.0.1)")
        p.add_argument("--ports", type=_ports, default=list(DEFAULT_PORTS),
                       help="소스 포트 (예: 10001-10012 또는 10002,10003). 기본: 10001~10012 (10010 빼고)")
        p.add_argument("--redis", default=None, help="Redis 주소 (기본 redis://<target>:6380/0)")
        p.add_argument("--no-redis", action="store_true", help="Redis 는 찍지 않는다")
        p.add_argument("--sample", type=int, default=STREAM_SAMPLE, help="스트림마다 읽는 마지막 항목 수")

    c = sub.add_parser("capture", help="모양을 찍어 파일로")
    common(c)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare", help="다시 찍어 기준과 비교")
    m.add_argument("baseline")
    common(m)
    m.add_argument("--save", metavar="파일", help="이번에 찍은 것도 남긴다")
    m.add_argument("--quiet-warn", action="store_true", help="경고는 개수만")
    f = sub.add_parser("feed", help="시험용 가짜 프레임·상태 넣기 (쓰기 — 시험 서버에만)")
    f.add_argument("--target", default="127.0.0.1")
    f.add_argument("--ports", type=_ports, default=list(DEFAULT_PORTS))
    f.add_argument("--frames", type=int, default=2)
    f.add_argument("--yes", action="store_true", help="쓰는 것을 안다")
    sd = sub.add_parser("seed", help="다른 Redis(읽기만)의 비전 · 오케스트레이션 · 노드 스트림 몇 건을 시험 Redis 로 복사")
    sd.add_argument("--from", dest="src", default=DEFAULT_REDIS, help="읽을 Redis (기본 %s, 읽기만)" % DEFAULT_REDIS)
    sd.add_argument("--to", dest="dst", required=True, help="쓸 시험 Redis (예: redis://127.0.0.1:6390/0)")
    sd.add_argument("--yes", action="store_true", help="--to 에 쓰는 것을 안다")
    a = ap.parse_args(argv)

    if a.cmd == "seed":
        if not a.yes or a.src == a.dst:
            print("seed 는 --to 에 쓴다 — 버리는 시험 Redis 에만, --from 과 달라야 한다. --yes 를 붙인다.", file=sys.stderr)
            return 2
        src = redis.Redis.from_url(a.src, socket_connect_timeout=3, socket_timeout=10)
        dst = redis.Redis.from_url(a.dst, socket_connect_timeout=3, socket_timeout=10)
        for k, n in sorted(seed(src, dst).items()):
            print("%-40s %d건" % (k, n))
        return 0

    if a.cmd == "feed":
        if not a.yes:
            print("feed 는 서버에 가짜 프레임을 넣는다 — 시험 서버(--no-pull, 버리는 Redis)에만. --yes 를 붙인다.",
                  file=sys.stderr)
            return 2
        for p, r in feed(a.target, a.ports, a.frames).items():
            print(":%d  %s" % (p, r))
        time.sleep(0.5)
        return 0

    url = a.redis or "redis://%s:6380/0" % a.target
    try:
        snap = capture(a.target, a.ports, url, a.sample, a.no_redis)
    except RuntimeError as e:
        print("찍지 못함: %s" % e, file=sys.stderr)
        return 2
    answering = [p for p, r in snap["http"].items() if r.get("answers")]
    if a.cmd == "capture":
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.write("\n")
        print("찍음 → %s  (답한 포트 %d/%d · Redis 키 무늬 %s)" % (
            a.out, len(answering), len(snap["http"]),
            len(snap["redis"]) if snap["redis"] is not None else "없음 (%s)" % snap["redis_error"]))
        return 0 if answering else 2

    with open(a.baseline, encoding="utf-8") as fh:
        base = json.load(fh)
    if a.save:
        with open(a.save, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, ensure_ascii=False, indent=1, sort_keys=True)
    if not answering:
        print("찍지 못함: 답하는 포트가 없다 (%s)" % a.target, file=sys.stderr)
        return 2
    if a.no_redis:
        base = dict(base, redis=None)
    rep = compare(base, snap)
    if a.quiet_warn:
        print("경고 %d건" % len(rep.warns))
    else:
        for w in rep.warns:
            print("경고  " + w)
    for x in rep.fails:
        print("실패  " + x)
    print("=" * 60)
    print("%s — 실패 %d · 경고 %d (칸 추가만: 추가는 경고, 없어짐·타입 바뀜은 실패)"
          % ("통과" if not rep.fails else "실패", len(rep.fails), len(rep.warns)))
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
