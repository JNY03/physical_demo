"""층 표시 (Docker 계획 D2, 2026-11-25) — 폴더로 vision · k3s 표시를 붙인다. 표시 이름은 pytest.ini.

vision/ 안이라도 core 가 쓰는 모듈(cue_catalog · model_registry)만 시험하는 파일은 core 로 둔다.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
#: vision/ 안의 core 시험 (core venv 에서도 돈다 — 웹 런처가 ds-core 안에서 카탈로그를 만든다)
CORE_IN_VISION = {"test_model_registry.py"}
LAYERS = {"vision": "vision", "orchestrator": "k3s"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            rel = Path(str(item.fspath)).resolve().relative_to(ROOT)
        except ValueError:
            continue
        mark = LAYERS.get(rel.parts[0]) if len(rel.parts) > 1 else None
        if mark and rel.name not in CORE_IN_VISION:
            item.add_marker(getattr(pytest.mark, mark))
