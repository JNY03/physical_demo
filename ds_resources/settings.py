"""경로 · 설정을 푸는 한 곳 (Docker 계획 D1, 2026-11-25). 표준 라이브러리만 쓴다 — core 어디서든 import 된다.

우선순위: 저장소 기본값 < config/ 파일 < 환경변수 DS_*.
경로는 모두 DS_HOME 기준 상대 경로다. DS_HOME 이 없으면 저장소 루트(지금 기기에서 하던 그대로).

  DS_HOME        데이터 · 가중치 · 상태가 놓인 뿌리 (models/ · external/ · state/ · stream_data/ · config/)
  DS_CONFIG_DIR  config/ 대신 쓸 폴더 (기본 DS_HOME/config)
  DS_PYTHON      비전 등 자식 프로세스를 띄울 Python (기본 sys.executable — venv 로 띄웠으면 그 venv)
  DS_NODE_NAME   이 기기 이름 (기억 파일의 열쇠 · k3s 노드 이름과 맞춤). 없으면 호스트 이름

옛 이름도 그대로 읽는다: config/hosts.json 이 없으면 저장소 루트의 hosts.local.json.
"""
import json
import os
import socket
import sys
from pathlib import Path

#: 코드가 놓인 저장소 루트 (vision/ · ds_resources/ 가 있는 곳)
REPO = Path(__file__).resolve().parent.parent

#: 설정 파일 안의 상대 경로 중 DS_HOME 기준으로 푸는 첫 폴더 이름. 그 밖의 문자열은 건드리지 않는다
#: (vision_config 의 source: box_height 같은 값이 경로로 오인되지 않게)
PATH_ROOTS = ("models", "external")


def ds_home():
    """DS_HOME (없으면 저장소 루트)."""
    v = os.environ.get("DS_HOME")
    return Path(v).expanduser().resolve() if v else REPO


def resolve(path, base=None):
    """상대 경로 → base(기본 DS_HOME) 기준 절대 경로. ~ 를 푼다. 절대 경로는 그대로."""
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else (Path(base) if base is not None else ds_home()) / p


def config_dir():
    """설정 폴더 — DS_CONFIG_DIR > DS_HOME/config."""
    v = os.environ.get("DS_CONFIG_DIR")
    return Path(v).expanduser().resolve() if v else ds_home() / "config"


def config_file(name, *legacy):
    """config/<name> 이 있으면 그것, 없으면 DS_HOME 기준 옛 이름(legacy)들 중 처음 있는 것, 다 없으면 None."""
    for p in [config_dir() / name] + [resolve(x) for x in legacy]:
        if p.is_file():
            return p
    return None


def hosts_file():
    """기기 주소 파일 — config/hosts.json > 옛 hosts.local.json (DS_HOME 루트)."""
    return config_file("hosts.json", "hosts.local.json")


def load_hosts():
    """기기 주소 사전 {소스: 주소}. 파일이 없으면 {}. 틀린 JSON 이면 바로 멈춘다(조용히 빈 주소로 돌지 않게)."""
    path = hosts_file()
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SystemExit("%s 을 못 읽었습니다: %s" % (path, e))
    return data if isinstance(data, dict) else {}


def python():
    """자식 프로세스용 Python — DS_PYTHON > 지금 Python."""
    return os.environ.get("DS_PYTHON") or sys.executable


def node_name():
    """노드 이름(기억 · 배치의 열쇠) — DS_NODE_NAME > 호스트 이름(소문자)."""
    return os.environ.get("DS_NODE_NAME") or socket.gethostname().lower()


def host_label():
    """데이터 칸(received_by · node_id · 텔레메트리 키)에 적는 이 기기 이름 — DS_NODE_NAME > 호스트 이름 그대로.
    DS_NODE_NAME 이 없으면 지금까지와 같은 값(대소문자 그대로)이라 읽는 쪽이 바뀐 것을 모른다."""
    return os.environ.get("DS_NODE_NAME") or socket.gethostname()


# ---------------------------------------------------------------- 설정 사전
def _is_root_path(s):
    if not isinstance(s, str) or not s or s.startswith(("~", "/")) or "://" in s:
        return False
    head = s.replace("\\", "/").split("/", 1)[0]
    return head in PATH_ROOTS


def resolve_tree(obj, base=None):
    """설정 사전을 훑어 models/… · external/… 상대 경로를 base(기본 DS_HOME) 기준 절대 경로로 바꾼 사본."""
    if isinstance(obj, dict):
        return {k: resolve_tree(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_tree(v, base) for v in obj]
    if _is_root_path(obj):
        return str(resolve(obj, base))
    return obj


def relative_tree(obj, base=None):
    """resolve_tree 의 반대 — base(기본 DS_HOME) 밑 models/… · external/… 절대 경로를 상대 경로로 되돌린 사본.
    설정 해시가 저장소 위치(DS_HOME)에 따라 달라지지 않게 쓴다."""
    if isinstance(obj, dict):
        return {k: relative_tree(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relative_tree(v, base) for v in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        root = str(Path(base) if base is not None else ds_home()).rstrip("/") + "/"
        if obj.startswith(root) and _is_root_path(obj[len(root):]):
            return obj[len(root):]
    return obj


def deep_merge(base, over):
    """사전 덮어쓰기 — over 의 사전은 안으로 합치고, 그 밖의 값(목록 포함)은 통째로 바꾼다. base 는 안 바꾼다."""
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def load_yaml(path, overlay=None):
    """YAML 한 파일 + (있으면) config/<overlay> 덮어쓰기. PyYAML 은 부를 때만 import 한다."""
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overlay:
        o = config_dir() / overlay
        if o.is_file():
            with open(o, encoding="utf-8") as f:
                data = deep_merge(data, yaml.safe_load(f) or {})
    return data


#: vision_config.yaml 의 덮어쓰기 파일 이름 (config/ 안)
VISION_OVERLAY = "vision_config.local.yaml"


def load_vision_config(path=None):
    """vision_config.yaml (+ config/vision_config.local.yaml). 경로는 풀지 않는다 — 설정 해시가
    기록하는 쪽(vision_infer)과 배치하는 쪽(오케스트레이터)에서 같게 나오도록 둘 다 이 함수로 읽는다."""
    return load_yaml(path or REPO / "vision" / "vision_config.yaml", VISION_OVERLAY)
