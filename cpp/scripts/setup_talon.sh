#!/usr/bin/env bash
# Host-side orchestrator for a local, containerized Talon stack — one coordinator
# and one S3/MinIO-backed worker — so the Talon integration test
# (test/filesystem/talon) and benchmark (benchmark_talon_reader) have a real
# deployment to run against.
#
# It mirrors talon's own e2e harness (talon's test/stack/deploy.sh): one script
# with up/down/status/test subcommands, and the coordinator and worker IMAGES are
# built locally from the SAME pinned talon source milvus-storage fetched for the C
# SDK (build/<type>/_deps/talon-src) — so the servers never drift from the linked
# client and nothing is pulled from a registry. MinIO is brought up in a container
# the same way scripts/setup_minio.sh does for the object-store tests.
#
# talon reaches its in-cluster C SDK test via `kubectl exec`; with a plain docker
# network we reach ours the analogous way. The test/benchmark client ALWAYS runs
# inside the stack network — in a milvus builder container joined to $NET — and
# reaches services by container name. This is not a convenience: the worker
# advertises its own container name and the coordinator relays every object
# request (StatObject/read) to the worker at that advertised address, so a client
# off the network (e.g. on the host via a published port) is never served —
# the on-network coordinator cannot dial a host-only address. Running the client
# on $NET makes the single advertised address reachable by both coordinator and
# client. We run the PRE-BUILT binary (no rebuild) in the builder image, which is
# ABI-compatible with it: locally the binary was built in this image; in CI it was
# built on the ubuntu-22.04 runner, whose libstdc++ the (also ubuntu-22.04-based)
# builder image is backward-compatible with.
#
# Subcommands:
#   up      build images (if missing) + bring the stack up, seed the origin
#           object, wait for the worker to register, then leave it running
#   test    run the Talon gtest suite against the running stack
#   bench   run the Talon benchmark against the running stack
#   down    stop and remove the stack (containers + network)
#   e2e     up + test + down as one self-cleaning shot (local convenience)
#   status  show stack state
#
# Requires a WITH_TALON build first (make build WITH_TALON=True) so both talon-src
# (the image build context) and the milvus_test/benchmark binaries exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CPP_DIR/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# talon source: the copy milvus-storage fetched for WITH_TALON (same pinned commit
# as the linked C SDK). The images are built from here, so they can never drift
# from the client. Override with TALON_SRC for an out-of-tree checkout.
TALON_SRC="${TALON_SRC:-$CPP_DIR/build/$BUILD_TYPE/_deps/talon-src}"

# --- topology -----------------------------------------------------------------
NET="${TALON_NET:-talon-net}"
MINIO_CT="${TALON_MINIO_CONTAINER:-talon-minio}"
COORD_CT="${TALON_COORDINATOR_CONTAINER:-talon-coordinator}"
WORKER_CT="${TALON_WORKER_CONTAINER:-talon-worker}"
COORD_PORT=7000
COORD_ADMIN=8000
WORKER_PORT=7001
WORKER_ADMIN=8001

# --- images -------------------------------------------------------------------
# The coordinator/worker images are tagged with the pinned talon short SHA for
# traceability; the no-drift guarantee comes from the build context (TALON_SRC),
# not the tag. Infra images match the object-store tests' choices where they overlap.
IMG_TAG="$(git -C "$TALON_SRC" rev-parse --short HEAD 2>/dev/null || echo pinned)"
COORD_IMG="${TALON_COORDINATOR_IMAGE:-talon-local/coordinator:$IMG_TAG}"
WORKER_IMG="${TALON_WORKER_IMAGE:-talon-local/worker:$IMG_TAG}"
MINIO_IMG="${TALON_MINIO_IMAGE:-quay.io/minio/minio}"
SEED_IMG="${TALON_SEED_IMAGE:-python:3-slim}"
BUILDER_IMG="${TALON_BUILDER_IMAGE:-milvusdb/milvus-env:ubuntu22.04-20260714-c135601}"

# Own back any files the container run touches (e.g. coverage .gcda written next
# to the instrumented binary) to the invoking user after the root-in-container run.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# --- origin object store (MinIO / S3-compatible) ------------------------------
MINIO_USER="${TALON_MINIO_ACCESS_KEY:-minioadmin}"
MINIO_PASS="${TALON_MINIO_SECRET_KEY:-minioadmin}"
BUCKET="${TALON_TEST_BUCKET:-test-bucket}"
OBJECT_KEY="${TALON_TEST_OBJECT_KEY:-talon-test/obj}"
OBJECT_MB="${TALON_TEST_OBJECT_MB:-64}"
REGION="${TALON_TEST_REGION:-us-east-1}"
BLOCK_SIZE="${TALON_WORKER_BLOCK_SIZE:-8388608}"
CLUSTER_ID="${TALON_CLUSTER_ID:-milvus-storage-dev}"
TEST_URI="s3://$BUCKET/$OBJECT_KEY"

log() { printf '\033[1;34m[talon]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[talon] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

require_talon_src() {
  [[ -f "$TALON_SRC/deploy/docker/coordinator.Dockerfile" ]] || fail \
    "talon source not found at $TALON_SRC. Build once with 'make build WITH_TALON=True' (in the builder container) so talon-src and the test binaries exist, or set TALON_SRC to a talon checkout."
}

build_images() {
  # Build from the pinned Dockerfiles with talon-src as context. Cached across
  # runs (cargo-chef layers survive source-only changes); force a rebuild with
  # TALON_REBUILD_IMAGES=1. The first build is expensive (compiles talon's Rust
  # deps) — that is the cost of the local-build approach talon itself uses.
  local img df
  for pair in "$COORD_IMG:coordinator" "$WORKER_IMG:worker"; do
    img="${pair%:*}"; df="${pair##*:}"
    if [[ "${TALON_REBUILD_IMAGES:-0}" != "1" ]] && docker image inspect "$img" >/dev/null 2>&1; then
      log "image $img present (set TALON_REBUILD_IMAGES=1 to rebuild)"
      continue
    fi
    log "building $img from $TALON_SRC/deploy/docker/$df.Dockerfile"
    docker build -f "$TALON_SRC/deploy/docker/$df.Dockerfile" -t "$img" "$TALON_SRC"
  done
}

remove_container() { docker rm -f "$1" >/dev/null 2>&1 || true; }

cmd_down() {
  log "tearing down the talon stack"
  remove_container "$WORKER_CT"
  remove_container "$COORD_CT"
  remove_container "$MINIO_CT"
  docker network rm "$NET" >/dev/null 2>&1 || true
}

seed_object() {
  # The milvus builder image ships no S3 CLI (s3cmd/mc/aws) and no boto3, so seed
  # through the stdlib-only SigV4 helper, running it in a throwaway python
  # container joined to the stack network — MinIO is never published to the host.
  # Idempotent (HEAD then PUT) and retried while MinIO finishes starting.
  local i
  for i in $(seq 1 15); do
    if docker run --rm --network "$NET" -v "$SCRIPT_DIR/s3_seed.py:/s3_seed.py:ro" "$SEED_IMG" \
        python3 /s3_seed.py --endpoint "http://$MINIO_CT:9000" \
        --bucket "$BUCKET" --key "$OBJECT_KEY" \
        --access "$MINIO_USER" --secret "$MINIO_PASS" \
        --region "$REGION" --size-mb "$OBJECT_MB" 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  fail "could not seed s3://$BUCKET/$OBJECT_KEY (MinIO not reachable at $MINIO_CT:9000?)"
}

wait_registered() {
  # Gate on the worker actually joining placement. The coordinator's admin API
  # (/api/v1/cluster) reports healthy_worker_count and is unauthenticated by
  # default; curl ships in the coordinator image, so query it via `docker exec`
  # (no admin port need be published, and it works identically in both modes).
  local i n
  printf '[talon] waiting for the worker to register'
  for i in $(seq 1 60); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$COORD_CT"; then
      printf '\n'; docker logs --tail 40 "$COORD_CT" 2>&1 || true
      fail "coordinator container exited"
    fi
    n="$(docker exec "$COORD_CT" curl -fsS "http://127.0.0.1:$COORD_ADMIN/api/v1/cluster" 2>/dev/null \
         | grep -oE '"healthy_worker_count"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$' || true)"
    if [[ -n "$n" && "$n" -ge 1 ]]; then printf ' ready\n'; return 0; fi
    printf '.'; sleep 1
  done
  printf ' FAILED\n' >&2
  docker logs --tail 40 "$WORKER_CT" 2>&1 || true
  fail "worker did not register within 60s"
}

cmd_up() {
  require_talon_src
  build_images
  cmd_down # clean slate: remove any stale stack from a previous run
  log "creating network $NET"
  docker network create "$NET" >/dev/null

  log "starting MinIO ($MINIO_CT) on $NET"
  docker run -d --name "$MINIO_CT" --network "$NET" \
    -e "MINIO_ROOT_USER=$MINIO_USER" -e "MINIO_ROOT_PASSWORD=$MINIO_PASS" \
    "$MINIO_IMG" server /data --console-address ":9001" >/dev/null
  seed_object
  log "seeded $TEST_URI ($OBJECT_MB MiB)"

  # No host port publishing: every client runs on $NET (see the header), so the
  # worker advertises its own container name for the coordinator to relay to.
  local worker_advertise="$WORKER_CT:$WORKER_PORT"

  log "starting coordinator ($COORD_CT)"
  docker run -d --name "$COORD_CT" --network "$NET" \
    "$COORD_IMG" \
    --listen "0.0.0.0:$COORD_PORT" --admin-listen "0.0.0.0:$COORD_ADMIN" \
    --cluster-id "$CLUSTER_ID" --node-id coord-0 >/dev/null

  # Backend/S3/cache settings are env-only in talon (the secret key intentionally
  # so); only topology goes on the CLI. The worker reaches MinIO by name on $NET.
  log "starting worker ($WORKER_CT), advertising $worker_advertise"
  docker run -d --name "$WORKER_CT" --network "$NET" \
    -e TALON_WORKER_BACKEND=s3 \
    -e "TALON_WORKER_S3_REGION=$REGION" \
    -e "TALON_WORKER_S3_ENDPOINT=http://$MINIO_CT:9000" \
    -e "TALON_WORKER_S3_ACCESS_KEY_ID=$MINIO_USER" \
    -e "TALON_WORKER_S3_SECRET_ACCESS_KEY=$MINIO_PASS" \
    -e TALON_WORKER_S3_PATH_STYLE=true \
    -e TALON_WORKER_FORCE_TOKIO_DATA_PLANE=1 \
    "$WORKER_IMG" \
    --coordinator "$COORD_CT:$COORD_PORT" \
    --listen "0.0.0.0:$WORKER_PORT" --advertise-addr "$worker_advertise" \
    --admin-listen "0.0.0.0:$WORKER_ADMIN" \
    --cluster-id "$CLUSTER_ID" --node-id worker-0 \
    --block-size "$BLOCK_SIZE" >/dev/null

  wait_registered
  cat <<EOF

[talon] stack is up
  coordinator : $COORD_CT:$COORD_PORT   (image $COORD_IMG)
  worker      : advertises $worker_advertise   (image $WORKER_IMG)
  origin      : $TEST_URI   (MinIO $MINIO_CT, in-network only)

  run:   $0 test        # Talon gtest suite
         $0 bench       # Talon benchmark
  stop:  $0 down
EOF
}

# Run the PRE-BUILT Talon client (test or benchmark) against the running stack,
# in a builder container joined to $NET (see the header for why on-network is
# mandatory). No rebuild: the binary from `make build WITH_TALON=True` is executed
# as-is, so this needs no conan cache and never drifts from the CI build. The tree
# is chown'd back afterwards in case the instrumented binary drops root-owned
# coverage files. LD_LIBRARY_PATH is set to the container-side build paths so the
# loader finds the bundled deps regardless of the binary's baked rpath.
run_client() {
  local kind="$1" relbin filter
  case "$kind" in
    test) relbin="test/milvus_test"; filter="--gtest_filter=Talon*" ;;
    bench) relbin="benchmark/benchmark"; filter="--benchmark_filter=TalonReaderBenchmark" ;;
    *) fail "unknown client kind: $kind" ;;
  esac
  [[ -x "$CPP_DIR/build/$BUILD_TYPE/$relbin" ]] || fail \
    "build/$BUILD_TYPE/$relbin not found — build first with 'make build WITH_TALON=True'."

  log "running the Talon $kind client on $NET ($BUILDER_IMG)"
  docker run --rm --network "$NET" \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$REPO_ROOT:/workspace/milvus-storage" \
    -e "TALON_COORDINATOR_ADDR=$COORD_CT:$COORD_PORT" \
    -e "TALON_TEST_URI=$TEST_URI" \
    -e "TALON_BENCH_BLOCK_SIZE=$BLOCK_SIZE" \
    -w /workspace/milvus-storage/cpp \
    "$BUILDER_IMG" \
    bash -lc "export LD_LIBRARY_PATH=/workspace/milvus-storage/cpp/build/$BUILD_TYPE:/workspace/milvus-storage/cpp/build/$BUILD_TYPE/libs:\${LD_LIBRARY_PATH:-}; ./build/$BUILD_TYPE/$relbin $filter; rc=\$?; chown -R $HOST_UID:$HOST_GID build 2>/dev/null || true; exit \$rc"
}

cmd_status() {
  echo "network: $NET"
  docker ps -a --filter "name=$MINIO_CT" --filter "name=$COORD_CT" --filter "name=$WORKER_CT" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || echo "(docker unavailable)"
}

case "${1:-status}" in
  up) trap 'cmd_down' EXIT; cmd_up; trap - EXIT ;;
  test) run_client test ;;
  bench) run_client bench ;;
  down | stop) cmd_down ;;
  status) cmd_status ;;
  e2e)
    trap 'cmd_down' EXIT
    cmd_up
    rc=0; run_client test || rc=$?
    cmd_down; trap - EXIT
    exit $rc
    ;;
  *) echo "usage: $0 [up|test|bench|down|e2e|status]" >&2; exit 2 ;;
esac
