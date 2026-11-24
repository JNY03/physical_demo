#!/usr/bin/env bash
# smoke_test.sh — 클러스터 점검: 노드 Ready · 역할 라벨/taint · metrics · 노드마다 시험 파드(DNS · 노드 넘는 통신 · MTU).
#   bash orchestrator/cluster/smoke_test.sh        (실패하면 0 이 아닌 코드로 끝난다)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NS=ds-smoke
fail=0
ok()  { printf '  ok   %s\n' "$*"; }
bad() { printf '  FAIL %s\n' "$*"; fail=1; }

mapfile -t NODES < <(python3 -c "import json;d=json.load(open('$HERE/nodes.json'));print(d['server']['name']);[print(a['name']) for a in d['agents']]")
mapfile -t TAINTED < <(python3 -c "import json;d=json.load(open('$HERE/nodes.json'));[print(a['name']) for a in d['agents'] if a['role'] in d['taint_roles']]")

echo "[1] 노드 Ready · 라벨"
for n in "${NODES[@]}"; do
    r="$(kubectl get node "$n" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status} {.metadata.labels.ds\.role}' 2>/dev/null)"
    [[ "$r" == True\ * ]] && ok "$n Ready role=${r#True }" || bad "$n: '$r'"
done
echo "[2] taint"
for n in "${TAINTED[@]}"; do
    kubectl get node "$n" -o jsonpath='{.spec.taints[*].key}' | grep -qw ds.role && ok "$n taint" || bad "$n taint 없음"
done
echo "[3] metrics"
for i in $(seq 24); do   # 막 붙은 노드는 첫 수집 전까지 <unknown> — 최대 2분 기다린다
    out="$(kubectl top nodes --no-headers 2>/dev/null)"
    [ -n "$out" ] && ! grep -q unknown <<<"$out" && break; sleep 5
done
while read -r n c _ m _; do
    [ -z "$n" ] && continue
    [[ "$c" == *unknown* ]] && bad "$n metrics 없음" || ok "$n cpu=$c mem=$m"
done <<<"$out"

echo "[4] 노드마다 시험 파드 (DNS · 서버 노드 파드와 통신 · 1200바이트 ping)"
kubectl delete ns "$NS" --ignore-not-found --wait=true --timeout=120s >/dev/null 2>&1   # 앞 실행이 지우는 중이면 기다린다
kubectl create ns "$NS" >/dev/null
pod() {  # pod <이름> <노드>
    cat <<EOF | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata: {name: $1, namespace: $NS, labels: {app: ds-smoke}}
spec:
  nodeName: $2
  tolerations: [{key: ds.role, operator: Exists, effect: NoSchedule}]
  restartPolicy: Never
  containers: [{name: c, image: busybox:1.36, command: ["sleep", "600"],
                resources: {requests: {cpu: 10m, memory: 8Mi}, limits: {memory: 32Mi}}}]
EOF
}
for n in "${NODES[@]}"; do pod "p-$n" "$n"; done
kubectl -n "$NS" wait --for=condition=Ready pod -l app=ds-smoke --timeout=300s >/dev/null 2>&1
SRV_IP="$(kubectl -n "$NS" get pod "p-${NODES[0]}" -o jsonpath='{.status.podIP}')"
for n in "${NODES[@]}"; do
    p="p-$n"
    if [ "$(kubectl -n "$NS" get pod "$p" -o jsonpath='{.status.phase}')" != Running ]; then bad "$n 파드가 안 뜸"; continue; fi
    dns="$(kubectl -n "$NS" exec "$p" -- nslookup kubernetes.default.svc.cluster.local 2>&1 | grep -c 'Address:.*10\.43\.0\.1')"
    big="$(kubectl -n "$NS" exec "$p" -- ping -c 2 -W 3 -s 1172 "$SRV_IP" 2>&1 | grep -c 'bytes from')"
    [ "$dns" -ge 1 ] && [ "$big" -ge 1 ] && ok "$n DNS · 서버 파드 큰 ping" || bad "$n DNS=$dns big_ping=$big"
done
kubectl delete ns "$NS" --wait=true --timeout=120s >/dev/null 2>&1
[ $fail = 0 ] && echo "통과" || echo "실패 있음"
exit $fail
