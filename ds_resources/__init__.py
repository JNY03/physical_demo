"""ds_resources — 자원 기록 · 기억 (core, k3s 없이 돈다).

streams  가시화가 읽는 두 스트림의 계약 (orchestration_stream · node_resource_stream) 과 기록 억제 규칙
memory   노드 기본 자원 · 모델 조합 사용량을 재고 기억하고 갱신한다 (M3)

k3s 를 쓰는 쪽(orchestrator/)은 이 패키지를 쓰지만, 이 패키지는 orchestrator/ 를 모른다.
"""
