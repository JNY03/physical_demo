"""선택 백엔드 찾기 — k3s 는 있으면 쓰고 없으면 안 쓴다 (core, 2026-11-24).

core 는 orchestrator/ 를 import 문으로 모른다. 여기 경로 문자열로만 가리키고, kubectl 과 kubeconfig 가 있을 때만
불러온다. 오픈소스 사용자가 k3s 없이 쓰면 auto 는 아무것도 고르지 않는다.
"""
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
K3S_SOURCE = "orchestrator.k3s_source:K3sNodeSource"
K3S_CPU = "orchestrator.k3s_executor:K3sCpuExecutor"


def k3s_available():
    cfg = os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config")
    return bool(shutil.which("kubectl")) and os.path.exists(cfg) and (REPO / "orchestrator" / "k3s_source.py").exists()


def resolve_cluster(choice):
    """auto | none | 모듈:클래스 → (노드 원천 경로, CPU 실행기 경로) 또는 (None, None)"""
    if choice in (None, "", "none") or (choice == "auto" and not k3s_available()):
        return None, None
    if choice == "auto":
        return K3S_SOURCE, K3S_CPU
    return choice, K3S_CPU
