import pytest


@pytest.fixture(autouse=True)
def _no_stop_wait(monkeypatch):
    """가짜 프로세스는 끝나지 않는다 — 멈춤 대기를 0 으로 (실제 대기는 gpu_executor.STOP_WAITS)"""
    from ds_resources import gpu_executor
    monkeypatch.setattr(gpu_executor, "STOP_WAITS", (0.0, 0.0, 0.0))
