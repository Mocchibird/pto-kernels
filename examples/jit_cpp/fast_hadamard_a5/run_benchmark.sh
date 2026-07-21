#!/usr/bin/env bash
# One-command on-device benchmark for fast_hadamard_a5 on an Ascend 950 (A5).
# Requires: a real 950 device, torch + torch_npu, and bisheng (CANN toolkit).
# This is the REAL-DEVICE path (no camodel) — for the simulator, use sim_test/run.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_HOME_PATH
if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
fi

# Pass through any args, e.g.:  ./run_benchmark.sh --npu 0 --block-dim 24 --csv bw.csv
cd "${SCRIPT_DIR}"
exec python3 benchmark.py "$@"
