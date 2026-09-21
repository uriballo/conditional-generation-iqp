#!/usr/bin/env bash
set -euo pipefail

if (( $# != 6 )); then
    echo "usage: $0 MANIFEST AUTHORIZATION MODE HOST_INDEX HOST_COUNT WORKERS" >&2
    exit 2
fi
manifest=$1
authorization=$2
mode=$3
host_index=$4
host_count=$5
workers=$6
if [[ "$mode" != "canary" && "$mode" != "full" ]]; then
    echo "MODE must be canary or full" >&2
    exit 2
fi
if (( host_count < 1 || host_index < 0 || host_index >= host_count || workers < 1 )); then
    echo "invalid host or worker count" >&2
    exit 2
fi
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python="$repo_root/.venv/bin/python"
if [[ "$manifest" != /* ]]; then manifest="$repo_root/$manifest"; fi
if [[ "$authorization" != /* ]]; then authorization="$repo_root/$authorization"; fi
if [[ ! -x "$python" || ! -f "$manifest" || ! -f "$authorization" ]]; then
    echo "repository environment, manifest, or authorization is missing" >&2
    exit 2
fi
"$python" -m experiments.scalar_gaussian_translation.production \
    --manifest "$manifest" --authorization "$authorization" verify >/dev/null
manifest_hash=$("$python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:16])' "$manifest")
result_root="$repo_root/results/scalar_gaussian_translation"
log_root="$result_root/logs/$manifest_hash/$mode-host${host_index}"
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
    local output_args=()
    if [[ "$mode" == "canary" ]]; then
        output_args=(--output-root "$repo_root/results/scalar_gaussian_translation_canary/host${host_index}")
    fi
    /usr/bin/time -v "$python" -m experiments.scalar_gaussian_translation.production \
        --manifest "$manifest" --authorization "$authorization" \
        run-job "$job_id" "${output_args[@]}" >"$log_root/${job_id}.log" 2>&1
}
export -f run_one
export python manifest authorization log_root repo_root mode host_index
if [[ "$mode" == "canary" ]]; then
    "$python" -m experiments.scalar_gaussian_translation.production \
        --manifest "$manifest" --authorization "$authorization" list-canary
else
    "$python" -m experiments.scalar_gaussian_translation.production \
        --manifest "$manifest" --authorization "$authorization" \
        list-shard "$host_index" "$host_count"
fi | xargs -r -n 1 -P "$workers" bash -c 'run_one "$1"' _
touch "$log_root/DONE"
