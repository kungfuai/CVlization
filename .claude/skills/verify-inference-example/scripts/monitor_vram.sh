#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
    echo "usage: $0 LOG.csv GPU_INDEX -- COMMAND [ARG ...]" >&2
    exit 2
fi

log_file=$1
gpu_index=$2
shift 3
interval=${CVL_VRAM_INTERVAL:-0.2}
gpu_uuid=$(
    nvidia-smi --query-gpu=uuid --format=csv,noheader \
        -i "$gpu_index" | tr -d ' '
)
gpu_info=$(
    nvidia-smi --query-gpu=index,name,uuid,memory.total \
        --format=csv,noheader -i "$gpu_index"
)
read -r baseline_used baseline_free < <(
    nvidia-smi --query-gpu=memory.used,memory.free \
        --format=csv,noheader,nounits -i "$gpu_index" |
        tr -d ' ' | tr ',' ' '
)

printf '%s\n' \
    'timestamp,device_used_mib,device_free_mib,process_used_mib' \
    > "$log_file"

monitor() {
    while true; do
        timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
        read -r device_used device_free < <(
            nvidia-smi --query-gpu=memory.used,memory.free \
                --format=csv,noheader,nounits -i "$gpu_index" |
                tr -d ' ' | tr ',' ' '
        )
        process_used=$(
            nvidia-smi \
                --query-compute-apps=gpu_uuid,used_gpu_memory \
                --format=csv,noheader,nounits 2>/dev/null |
                awk -F, -v target="$gpu_uuid" '
                    {
                        gsub(/[[:space:]]/, "", $1)
                        gsub(/[[:space:]]/, "", $2)
                        if ($1 == target && $2 ~ /^[0-9]+$/) {
                            sum += $2
                        }
                    }
                    END { print sum + 0 }
                '
        )
        printf '%s,%s,%s,%s\n' \
            "$timestamp" "$device_used" "$device_free" "$process_used" \
            >> "$log_file"
        sleep "$interval"
    done
}

monitor_pid=
stop_monitor() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=
    fi
}
trap stop_monitor EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

monitor &
monitor_pid=$!
sleep "$interval"

set +e
"$@"
command_status=$?
set -e

# Capture release after a short-running container exits.
sleep 1
stop_monitor
trap - EXIT INT TERM

read -r final_used final_free < <(
    nvidia-smi --query-gpu=memory.used,memory.free \
        --format=csv,noheader,nounits -i "$gpu_index" |
        tr -d ' ' | tr ',' ' '
)

echo "GPU: $gpu_info" >&2
echo "Baseline device memory: ${baseline_used} MiB used, ${baseline_free} MiB free" >&2
awk -F, '
    NR > 1 {
        if ($2 > device_peak) device_peak = $2
        if ($4 > process_peak) process_peak = $4
        samples++
    }
    END {
        printf "Observed device peak: %d MiB\n", device_peak
        printf "Observed process peak: %d MiB\n", process_peak
        printf "Samples: %d\n", samples
    }
' "$log_file" >&2
echo "Final device memory: ${final_used} MiB used, ${final_free} MiB free" >&2
echo "Polling interval: ${interval}s" >&2
echo "CSV evidence: $log_file" >&2

sample_count=$(awk 'END { print NR - 1 }' "$log_file")
if (( sample_count < 2 )); then
    echo "VRAM monitoring failed: fewer than two samples recorded" >&2
    if (( command_status == 0 )); then
        command_status=3
    fi
fi

exit "$command_status"
