#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo "usage: $0 MANIFEST SHARD_INDEX SHARD_COUNT WORKERS" >&2
    exit 2
fi
manifest=$1
shard_index=$2
shard_count=$3
workers=$4
if (( shard_count < 1 || shard_index < 0 || shard_index >= shard_count || workers < 1 )); then
    echo "invalid shard or worker count" >&2
    exit 2
fi
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python="$repo_root/.venv/bin/python"
if [[ "$manifest" != /* ]]; then manifest="$repo_root/$manifest"; fi
if [[ ! -x "$python" || ! -f "$manifest" ]]; then
    echo "repository environment or manifest is missing" >&2
    exit 2
fi
if ! "$python" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("enabled") and not p.get("production") else 1)' "$manifest"; then
    echo "manifest is disabled or is a production proposal" >&2
    exit 2
fi
split=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["split"])' "$manifest")
if [[ "$split" == audit ]]; then
    if ! "$python" -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); m=json.load(p.open()); g=pathlib.Path(m.get("frozen_gates_path") or "frozen_gates.json"); g=g if g.is_absolute() else (p.parent/g).resolve(); sys.exit(0 if g.is_file() and hashlib.sha256(g.read_bytes()).hexdigest()==m.get("frozen_gates_sha256") else 1)' "$manifest"; then
        echo "audit gate file is absent or does not match the manifest" >&2
        exit 2
    fi
fi
manifest_hash=$("$python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:16])' "$manifest")
result_root="$repo_root/results/scalar_gaussian_translation"
log_root="$result_root/logs/$manifest_hash"
cache_root="$result_root/.jax-cache/$manifest_hash"
mkdir -p "$log_root" "$cache_root"
cpu_count=$(getconf _NPROCESSORS_ONLN)
if (( workers > cpu_count )); then
    echo "${workers} workers exceed the host's ${cpu_count} CPUs" >&2
    exit 2
fi
threads_per_worker=$(((cpu_count + workers - 1) / workers))
export OMP_NUM_THREADS="$threads_per_worker"
export OPENBLAS_NUM_THREADS="$threads_per_worker"
export JAX_COMPILATION_CACHE_DIR="$cache_root"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
run_one() {
    local job_id=$1
    /usr/bin/time -v "$python" -m experiments.scalar_gaussian_translation.campaign \
        --manifest "$manifest" run-job "$job_id" >"$log_root/${job_id}.log" 2>&1
}
export -f run_one
export python manifest log_root
"$python" -m experiments.scalar_gaussian_translation.campaign --manifest "$manifest" \
    list-shard "$shard_index" "$shard_count" | xargs -r -n 1 -P "$workers" bash -c 'run_one "$1"' _
