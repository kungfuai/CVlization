#!/usr/bin/env bash
# Run browser_use against every task in examples/agentic/web/_tasks/ and
# report. Contract documented in ../_tasks/README.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="$(cd "${SCRIPT_DIR}/../_tasks" && pwd)"
SCRATCH="${SCRATCH:-$(mktemp -d -t cvl-browser-use-tasks-XXXXXX)}"
echo "Adapter: browser_use  Scratch: ${SCRATCH}"
echo "Tasks dir: ${TASKS_DIR}"
echo

declare -a results=()
overall_start=$(date +%s)

for task_dir in "${TASKS_DIR}"/*/; do
  name=$(basename "${task_dir%/}")
  prompt_file="${task_dir}PROMPT.md"
  verify_script="${task_dir}verify.sh"
  [ -f "${prompt_file}" ] || continue
  [ -f "${verify_script}" ] || continue

  out="${SCRATCH}/${name}"
  mkdir -p "${out}"
  prompt="$(cat "${prompt_file}")"

  echo "[${name}] running browser_use ..."
  start=$(date +%s)
  # BROWSER_USE_OUTPUTS overrides run.sh's default per-cwd outputs path,
  # so artifacts land in our per-task scratch dir alongside the agent log.
  if BROWSER_USE_OUTPUTS="${out}" \
       bash "${SCRIPT_DIR}/run.sh" -- "${prompt}" \
       >"${out}/agent.log" 2>&1; then
    agent_exit=0
  else
    agent_exit=$?
  fi
  end=$(date +%s)
  wall=$((end - start))

  if [ "${agent_exit}" -ne 0 ]; then
    results+=("[${name}] FAIL agent  wall=${wall}s  exit=${agent_exit}  log=${out}/agent.log")
    continue
  fi

  if bash "${verify_script}" "${out}" >"${out}/verify.log" 2>&1; then
    results+=("[${name}] PASS         wall=${wall}s")
  else
    results+=("[${name}] FAIL verify  wall=${wall}s  vlog=${out}/verify.log")
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
