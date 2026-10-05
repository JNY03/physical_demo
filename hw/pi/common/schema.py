"""
피지컬팀 mk2 — 메시지 스키마 어댑터 (HW-C-02 / BE-C-01·BE-C-02 정렬)
=====================================================================
백엔드 공통 규약이 확정되기 전에 필드명이 코드 곳곳에 흩어지면, 확정 후 전 파일을
고쳐야 한다. 그래서 "봉투(envelope) 만드는 곳"을 여기 한 군데로 모았다.
필드명이 바뀌면 `envelope()` 한 함수만 고치면 된다.

BE-C-01 공통 필드: source_id, node_id, zone_id, timestamp, schema_version, correlation_id
BE-C-02 식별자 계층: Entity(개체) / Node(물리 노드) / Zone(구역) — IP·MAC 같은
        가변값이 아니라 논리 식별자로 참조한다. MAC은 등록 메시지의 참고 필드이자
        BE-T-05(사설 IP 라우팅)의 매핑 근거로만 싣는다.

## 결측 표현 규칙 — **모르는 값을 0 으로 채우지 않는다**

디지털 트윈은 데이터 소스를 몰라야 하고(SAR·드론·CCTV·로봇 무관), 그래서 서버가
스키마를 소유하며 **채워지지 않은 필드는 비워서** 내려보낸다. 말단도 같은 규칙을 진다.

| 상태 | 표현 | 금지 |
|---|---|---|
| 값이 있다 | 그 값 | — |
| 아직 없다 / 센서가 안 준다 | `null` (파이썬 `None`) | `0`, `-1`, `""` 로 대체 |
| 값이 오래됐다 | 값 + 나이(`*_age_s`) | 최신값인 척 |

규약 `CommandResult.result` 처럼 **null 을 실을 수 없는 자리**(map<string,double>)에서는
**키를 아예 넣지 않는다.** 0 을 넣으면 "측정했더니 0" 과 구별되지 않는다.

이 규칙은 실패에서 나왔다(2026-09-10). Go1 이 전부 0 인 텔레메트리 프레임을 발행했는데
파서가 그대로 읽어 "배터리 0%·자세 0°·위치 원점" 이라는 그럴듯한 거짓값을 올렸고,
배터리 경보가 오발동했으며 임무가 `battery_too_low` 로 거부될 뻔했다. 값이 없는 것과
값이 0 인 것은 다른 사실이다. 자세한 배경은 `docs/ARCHITECTURE_ALIGNMENT.md` §1-3.
"""
import hashlib
import os
import re
import socket
import time
import uuid
from datetime import datetime, timezone

from common import config

SCHEMA_VERSION = "1.1"

# 프로세스 1회 기동을 가리키는 값. 순번 리셋의 경계를 정확히 가른다(백엔드 회신 §6-2).
# 파일에 저장하지 않는다 — 저장하면 재시작해도 같은 값이 되어 의미가 사라진다.
SESSION_ID = uuid.uuid4().hex[:12]

# 백엔드가 source_id 단일 필드로 확정(2026-09-07 회신 §1-1 ③) — 별칭 발행 중단.
# 소비자(monitor.py:56·analyzer.py:92)는 source_id 우선이라 안전하다.
LEGACY_DEVICE_ID = False

# BE-T-04 / [G3]: 장치 자기보고 상태. 서버 판정 가용성(availability)과는 다른 층이다.
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_FAULT = "fault"


def iso_now():
    """HW-S-08: 모든 메시지에 타임스탬프. chrony로 엣지와 시각을 맞춘 뒤라야 의미가 있다.

    콜론 있는 오프셋(+09:00) + 밀리초. RFC3339 / 계약 date-time 정합(백엔드 회신 §1-1 ② · §6-9).
    strftime("%z")가 내는 +0900 은 계약이 거부한다 — 전량 격리된다."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


# ---------------------------------------------------------------- 산출물 작명
# `_` 는 칸 구분자로 쓰므로 안전 목록에서 뺀다. 안 빼면 토큰 안의 `_` 가
# 칸 경계처럼 보여 받는 쪽이 이름을 되쪼갤 수 없다.
_UNSAFE = re.compile(r"[^0-9A-Za-z.-]+")


def file_stamp(unix_ts=None):
    """파일명에 넣는 시각. **UTC 기본형식 + `Z`** 다.

    `iso_now()` 와 형식이 **일부러** 다르다. 파일명에는 콜론을 쓸 수 없고(윈도우·
    SMB 에서 못 만든다), `+09:00` 의 `+` 는 URL·S3 도구가 공백으로 잘못 되돌리는
    일이 흔하다. `Z` 를 쓰면 콜론도 플러스도 없고, **문자열 정렬이 곧 시간 정렬**이다.

    로컬 시각(20:33)과 달라 보이는 것도 의도다. 오프셋 없는 로컬 시각을 파일명에
    박으면 파일이 저장소로 옮겨진 뒤 그게 어느 시간대였는지 아무도 알 수 없다 —
    모르는 값을 0 으로 채우지 않는다는 위 규칙과 같은 종류의 문제다.
    본문(JSON)에는 로컬 오프셋이 붙은 `iso_now()` 값이 그대로 들어간다.

        20260930T113301.123Z
    """
    dt = datetime.fromtimestamp(time.time() if unix_ts is None else unix_ts,
                                timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S.") + f"{dt.microsecond // 1000:03d}Z"


def _safe(token):
    """파일명에 넣어도 되는 형태로 다듬는다.

    한글 라벨처럼 통째로 걸러지는 값이 들어오면 **조용히 사라지게 두지 않는다.**
    빈 칸으로 만들면 서로 다른 두 사진이 같은 이름이 되어 하나가 다른 하나를
    덮어쓴다 — 그런 종류의 유실은 아무 로그도 남기지 않는다. 걸러졌을 때는
    원문의 짧은 해시를 대신 넣어, 이름이 겹치지 않으면서 변환됐음이 드러나게 한다.
    """
    if token in (None, ""):
        return ""
    out = _UNSAFE.sub("-", str(token)).strip("-.")
    if out:
        return out
    return "x" + hashlib.sha1(str(token).encode("utf-8")).hexdigest()[:6]


def artifact_name(node_id, entity_id=None, parts=(), unix_ts=None, ext="jpg"):
    """산출물 파일 하나의 이름. **기기명과 시각을 맨 앞에 둔다.**

    예전 이름은 `frame_000001.jpg`·`go1-001_03.jpg` 처럼 사실상 순번뿐이었다.
    파일이 세션 디렉터리를 떠나 객체 저장소에 쌓이면 그게 언제 어느 기기에서
    나온 것인지 **파일명만으로는 알 수 없다.** 게다가 순번은 프로세스가 다시 뜰
    때마다 1 부터 시작하므로 서로 덮어쓴다(frame_ring 에서 실제로 겪었다).

        pi7_go1-001_20260930T113301.123Z_cam1_rot135_seq03.jpg
        └노드┘ └개체──┘ └───시각(UTC)────┘ └──── 무엇인지 ────┘

    노드를 앞에 두는 이유: 한 기기가 만든 것이 한 덩어리로 묶이고, 그 안에서는
    시각순으로 정렬된다. 시각이 UTC 기본형식이라 문자열 정렬이 그대로 시간순이다.
    """
    seg = [_safe(node_id), _safe(entity_id), file_stamp(unix_ts)]
    seg += [_safe(p) for p in parts]
    name = "_".join(s for s in seg if s)
    return f"{name}.{ext.lstrip('.')}" if ext else name


def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _mac(ip=""):
    """이 노드의 MAC. **BE-T-05 가 MAC↔구역 매핑을 라우팅 근거로 쓰므로 값이
    흔들리면 안 된다.**

    원래는 `uuid.getnode()` 하나였는데, 컨테이너 안에서 실제 인터페이스를 못 찾으면
    **난수를 만들어 돌려준다**(locally-administered 비트가 선 값). 실측(pi1, 2026-09-14):
    같은 파이에서 센서 노드는 2c:cf:67:9d:10:4c, 컨테이너 로봇 노드는 46:fa:28:cd:e2:c0
    를 보고했고 후자는 프로세스마다 바뀐다. 대장이 한 노드를 여럿으로 본다.

    그래서 **우리가 실제로 쓰는 인터페이스**의 MAC 을 찾는다 — `_ip()` 가 고른,
    브로커로 나가는 그 인터페이스다. 못 찾으면 단계적으로 물러난다.
    """
    forced = os.environ.get("HW_MAC", "").strip()
    if forced:
        return forced

    if ip:
        try:
            import psutil
            for name, addrs in psutil.net_if_addrs().items():
                if name == "lo":
                    continue
                if not any(getattr(a, "address", "") == ip for a in addrs):
                    continue
                for a in addrs:
                    if getattr(a, "family", None) != psutil.AF_LINK:
                        continue
                    addr = (a.address or "").lower()
                    # 루프백·가상 인터페이스는 전부 0 을 돌려준다. 그건 주소가 아니다.
                    if len(addr) == 17 and addr != "00:00:00:00:00:00":
                        return addr
        except Exception:
            pass                      # psutil 이 없거나 형태가 다르면 아래로 내려간다

    # 물리 인터페이스를 직접 뒤진다. 가상(docker0·veth·usb0)과 난수 MAC 은 뺀다.
    try:
        for name in sorted(os.listdir("/sys/class/net")):
            if name == "lo" or name.startswith(("docker", "veth", "br-", "usb")):
                continue
            addr = _read(f"/sys/class/net/{name}/address").lower()
            if len(addr) == 17 and addr != "00:00:00:00:00:00" \
                    and not int(addr[:2], 16) & 0x02:      # locally-administered 제외
                return addr
    except OSError:
        pass

    # 마지막 수단. 여기까지 오면 값이 난수일 수 있다 — 그래서 순서가 마지막이다.
    return ":".join(f"{(uuid.getnode() >> i) & 0xff:02x}" for i in range(40, -1, -8))


def _default_gateway():
    """기본 경로의 게이트웨이 IPv4. 없으면 빈 문자열.

    폐쇄망에서도 통한다 — 외부 주소로 물어보는 방식은 경로가 없으면 그냥 실패한다.
    `/proc/net/route` 의 주소는 리틀엔디안 hex 다."""
    try:
        with open("/proc/net/route") as f:
            next(f)                                   # 머리글
            for line in f:
                c = line.split()
                if len(c) > 2 and c[1] == "00000000" and c[2] != "00000000":
                    raw = int(c[2], 16)
                    return ".".join(str((raw >> i) & 0xFF) for i in (0, 8, 16, 24))
    except (OSError, StopIteration, ValueError):
        pass
    return ""


def _ip():
    """이 노드의 주소. **다른 기기가 실제로 찾아올 수 있는 값이어야 한다.**

    브로커로 나가는 인터페이스를 골랐는데, 브로커가 이 파이 자신이면(현장 구성이
    `HW_BROKER_HOST=127.0.0.1` 이다) **루프백이 잡힌다.** 실측(pi7, 2026-09-30):
    등록 메시지의 ip 가 계속 `127.0.0.1` 로 나가고 있었다. 가시화와 서버가 파이가
    띄운 주소로 데이터를 가져오는 구조에서 이 값은 **틀린 것보다 나쁘다** — 형식이
    멀쩡해서 아무도 의심하지 않는다. _mac() 이 컨테이너 난수 MAC 때문에 똑같은
    함정에 빠졌던 것과 같은 종류의 실패다.

    그래서 루프백이 잡히면 기본 경로 게이트웨이 쪽으로 다시 물어 본다. 어느 경우에도
    실제 패킷은 나가지 않는다 — UDP connect 는 경로만 고른다.
    """
    fallback = ""
    for target in ((config.BROKER_HOST, config.BROKER_PORT),
                   (_default_gateway(), 9)):
        if not target[0]:
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(target)
            ip = s.getsockname()[0]
        except OSError:
            ip = ""
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
        fallback = fallback or ip
    return fallback


class Identity:
    """Entity/Node/Zone 3계층 + 물리 주소. HW-C-07의 '변경 시 갱신'을 위해
    현재 값을 다시 읽어 이전과 비교하는 책임까지 여기서 진다."""

    def __init__(self, entity_id, node_id, zone_id, mac, ip, entity_type="node"):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.node_id = node_id
        self.zone_id = zone_id
        self.mac = mac
        self.ip = ip

    @classmethod
    def resolve(cls, entity_type="node"):
        # 우선순위: 명시적 환경변수 > 설정 파일 > 기본값.
        # 운용에서는 /etc/device_id 가 정본이지만, 한 대의 파이에서 센서 노드와 로봇
        # 노드를 함께 검증할 때처럼 명시적으로 지정한 값이 있으면 그쪽이 더 구체적이다.
        entity_id = os.environ.get("HW_ENTITY_ID", "") or _read(config.ENTITY_ID_FILE)
        if not entity_id:
            raise SystemExit(
                f"device_id를 찾을 수 없다: {config.ENTITY_ID_FILE} (HW-C-07 채번 필요). "
                "HW_ENTITY_ID 환경변수로도 지정할 수 있다."
            )
        node_id = (os.environ.get("HW_NODE_ID", "") or _read(config.NODE_ID_FILE)
                   or socket.gethostname())
        zone_id = (os.environ.get("HW_ZONE_ID", "") or _read(config.ZONE_ID_FILE)
                   or config.ZONE_ID)
        # 노드 클래스가 자기 타입을 안다. 환경변수는 명시적 덮어쓰기로만 이긴다.
        etype = config.ENTITY_TYPE or entity_type
        ip = _ip()
        return cls(entity_id, node_id, zone_id, _mac(ip), ip, etype)

    @property
    def topic_base(self):
        """통합정립본 v3의 {domain}/{type}/{id}/{channel} 체계.
        구조 자체를 config.TOPIC_TEMPLATE 로 뺐다 — 도메인 체계(AGENDA #2)가
        어떻게 확정되든 설정 한 줄로 전환되고 코드는 바뀌지 않는다."""
        return config.TOPIC_TEMPLATE.format(
            zone=self.zone_id, etype=self.entity_type, eid=self.entity_id)

    def fingerprint(self):
        """이 값이 달라지면 재등록 대상(HW-C-07: 네트워크 또는 구역 변경 시 갱신)."""
        return (self.zone_id, self.mac, self.ip)

    def registration(self):
        """등록(Birth)에 싣는 장치 정보. BE-T-04가 구역 단위 장치 목록으로 보관하고
        BE-T-05가 MAC↔구역 매핑을 라우팅 근거로 쓴다."""
        return {
            "entity_id": self.entity_id,
            "node_id": self.node_id,
            "zone_id": self.zone_id,
            "entity_type": self.entity_type,
            "device_type": config.DEVICE_TYPE,
            "fw_version": config.FW_VERSION,
            "mac": self.mac,
            "ip": self.ip,
        }


def envelope(identity, seq=None, correlation_id=None):
    """모든 발행 메시지의 공통 머리. 채널별 본문은 호출부에서 합친다."""
    env = {
        "schema_version": SCHEMA_VERSION,
        "source_id": identity.entity_id,   # BE-C-01
        "node_id": identity.node_id,       # BE-C-02
        "zone_id": identity.zone_id,
        "timestamp": iso_now(),            # HW-S-08
        "session_id": SESSION_ID,          # 백엔드 회신 §6-2: 순번 리셋 경계
    }
    if seq is not None:
        env["sequence_id"] = seq   # 계약 필드명(백엔드 회신 §1-1 ①). 인자 이름은 그대로.
    if correlation_id is not None:
        env["correlation_id"] = correlation_id   # BE-X-01: 백엔드 발급 command_id를 에코
    if LEGACY_DEVICE_ID:
        env["device_id"] = identity.entity_id
    return env
