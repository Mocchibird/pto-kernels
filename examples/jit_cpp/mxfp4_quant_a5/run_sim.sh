#!/usr/bin/env bash
# Run the mxfp4_quant_a5 simulator smoke on the A5 (Ascend950) CA model.
#   ./run_sim.sh                     # default cases, k=64, 2 cores
#   ./run_sim.sh --k 128 --block-dim 4
# Any argument is forwarded to run_sim_mxfp4_a5.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export ASCEND_HOME_PATH

SOC="${MSPROF_SOC_VERSION:-Ascend950PR_9599}"
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC}/lib"
if [[ ! -d "${SIM_LIB}" ]]; then
  echo "no simulator package for ${SOC} at ${SIM_LIB}" >&2
  echo "installed: $(ls "${ASCEND_HOME_PATH}/tools/simulator" | tr '\n' ' ')" >&2
  exit 2
fi
export LD_LIBRARY_PATH="${SIM_LIB}:${LD_LIBRARY_PATH:-}"
# Read by pto_demo_utils: stretches the process timeout and drops repeat counts.
export PTO_SIMULATOR=1

# One run writes ~7000 dump files. Point MSPROF_OUT_DIR at a container-local
# filesystem when the checkout is a bind mount: msprof's parse step re-reads the
# CA model's dump, and on a virtiofs/"fakeowner" mount aicore_binary.o comes back
# unreadable, which fails the report (but not the numerics).
OUT_DIR="${MSPROF_OUT_DIR:-${SCRIPT_DIR}/outputs/msprof_mxfp4}"
RESULT_JSON="${OUT_DIR}/result.json"
mkdir -p "${OUT_DIR}"
rm -f "${RESULT_JSON}"

# The numerics survive a bad output filesystem; only the profiling report dies, so
# say so up front rather than leaving a confusing parse error to interpret.
FSTYPE="$(findmnt -T "${OUT_DIR}" -no FSTYPE 2>/dev/null || true)"
case "${FSTYPE}" in
  fakeowner | virtiofs | 9p | fuse* | nfs*)
    echo "[warn] ${OUT_DIR} is on ${FSTYPE}: the profiling report will likely fail" >&2
    echo "[warn] to parse (aicore_binary.o unreadable). Correctness still verifies." >&2
    echo "[warn] For the report: MSPROF_OUT_DIR=/tmp/sim_out $0 $*" >&2
    ;;
esac

# Compile OUTSIDE msprof: bisheng under the injected simulator runtime is only
# slower, and a compile error is easier to read without the profiler's wrapping.
python3 -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from jit_util_mxfp4_a5 import compile_kernel
print('[compile]', compile_kernel(verbose=False))
"

msprof op simulator \
  --soc-version="${SOC}" \
  --timeout="${MSPROF_TIMEOUT:-120}" \
  --output="${OUT_DIR}" \
  python3 "${SCRIPT_DIR}/run_sim_mxfp4_a5.py" --output-json "${RESULT_JSON}" "$@"

# msprof reports its own exit status, not the kernel's verdict, so the JSON is
# the gate: no file means the run died before it could compare anything.
python3 - "${RESULT_JSON}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    sys.exit(f"no {path}: the run did not reach the comparison")
data = json.loads(path.read_text())
print(f"[verdict] {data['result']}")
sys.exit(0 if data.get("result") == "PASS" else 1)
PY
