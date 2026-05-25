#!/usr/bin/env bash
# Run pi against every task in examples/agentic/code/_tasks/ and report.
# Contract documented in ../_tasks/README.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="$(cd "${SCRIPT_DIR}/../_tasks" && pwd)"
SCRATCH="${SCRATCH:-$(mktemp -d -t cvl-pi-tasks-XXXXXX)}"
echo "Adapter: pi  Scratch: ${SCRATCH}"
echo "Tasks dir: ${TASKS_DIR}"
echo

declare -a results=()
overall_start=$(date +%s)

for task_dir in "${TASKS_DIR}"/*/; do
  name=$(basename "${task_dir%/}")
  prompt_file="${task_dir}PROMPT.md"
  input_dir="${task_dir}input"
  verify_script="${task_dir}verify.sh"
  [ -f "${prompt_file}" ] || continue
  [ -f "${verify_script}" ] || continue

  work="${SCRATCH}/${name}"
  rm -rf "${work}"
  mkdir -p "${work}"
  if [ -d "${input_dir}" ]; then
    # Copy hidden files too; tar handles empty input/ gracefully.
    ( cd "${input_dir}" && tar cf - . ) | ( cd "${work}" && tar xf - )
  fi

  log="${SCRATCH}/${name}.agent.log"
  vlog="${SCRATCH}/${name}.verify.log"
  prompt="$(cat "${prompt_file}")"

  echo "[${name}] running pi ..."
  start=$(date +%s)
  if ( cd "${work}" && bash "${SCRIPT_DIR}/run.sh" -- -p "${prompt}" ) >"${log}" 2>&1; then
    agent_exit=0
  else
    agent_exit=$?
  fi
  end=$(date +%s)
  wall=$((end - start))

  if [ "${agent_exit}" -ne 0 ]; then
    results+=("[${name}] FAIL agent  wall=${wall}s  exit=${agent_exit}  log=${log}")
    continue
  fi

  if bash "${verify_script}" "${work}" >"${vlog}" 2>&1; then
    results+=("[${name}] PASS         wall=${wall}s")
  else
    results+=("[${name}] FAIL verify  wall=${wall}s  vlog=${vlog}")
  fi
done

overall_end=$(date +%s)
overall_wall=$((overall_end - overall_start))

echo
echo "=== summary ==="
printf '  %s\n' "${results[@]}"
pass=$(printf '%s\n' "${results[@]}" | grep -c PASS || true)
total=${#results[@]}
echo "  ${pass}/${total} passed in ${overall_wall}s"
echo "  scratch dir kept at: ${SCRATCH}"

[ "${pass}" = "${total}" ]
