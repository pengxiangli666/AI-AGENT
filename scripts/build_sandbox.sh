#!/usr/bin/env bash
# Build the sandbox image and report cold/warm build + container start times.
# These are the numbers to quote for "environment boot time" - measure, don't guess.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${SANDBOX_IMAGE:-coding-agent-sandbox:latest}"

now() { date +%s.%N; }

echo "== build (uses layer cache; run twice to see cold vs warm) =="
t0=$(now)
DOCKER_BUILDKIT=1 docker build -f sandbox/Dockerfile -t "$IMAGE" .
t1=$(now)
printf 'build time: %.1fs\n' "$(echo "$t1 - $t0" | bc -l)"

echo "== container start-to-exit, hardened flags, 10 runs =="
total=0
for i in $(seq 1 10); do
  s=$(now)
  docker run --rm --network none --cap-drop ALL \
    --security-opt no-new-privileges --security-opt "seccomp=sandbox/seccomp.json" \
    --read-only --tmpfs /tmp --user 10001:10001 "$IMAGE" \
    sh -c 'python -c "print(1)" && node -e "console.log(1)"' >/dev/null
  e=$(now)
  d=$(echo "$e - $s" | bc -l)
  total=$(echo "$total + $d" | bc -l)
  printf 'run %2d: %.2fs\n' "$i" "$d"
done
printf 'mean: %.2fs\n' "$(echo "$total / 10" | bc -l)"
