#!/usr/bin/env bash
# Build + run fused_hadamard_mxfp4_a5 under the Ascend 950 camodel simulator.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD="${SCRIPT_DIR}/build"
SOC="${SOC_VERSION:-Ascend950PR_9599}"
: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
export ASCEND_HOME_PATH
DRV="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"
SIMLIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC}/lib"
INC="${ASCEND_HOME_PATH}/aarch64-linux/include"
BF="${IN_BF16:-0}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/runtime/lib64' | paste -sd: -)"
fi
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH:-}"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export LD_LIBRARY_PATH="${SIMLIB}:${LD_LIBRARY_PATH}"

mkdir -p "${BUILD}" && cd "${BUILD}"
echo "==> Compile kernel (IN_BF16=${BF})"
"${BISHENG}" -xcce --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE -DHAD_IN_BF16="${BF}" -DROWS_PER_TILE=64 \
  -O2 -std=c++17 -Wno-ignored-attributes -Wno-macro-redefined -fPIC \
  -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false -Xhost-start -Xhost-end \
  -I"${INC}" -I"${ASCEND_HOME_PATH}/include" -I"${DRV}/kernel/inc" \
  -c "${KDIR}/fused_hadamard_mxfp4_a5.cpp" -o k.o
"${BISHENG}" -fPIC -shared --cce-fatobj-link -Wl,-soname,libk.so k.o -o libk.so

echo "==> Compile host"
"${BISHENG}" -xc++ -include stdint.h -include stddef.h -std=c++17 -O2 \
  -Wno-ignored-attributes -Wno-macro-redefined -DBATCH_DIM=64 -DIN_BF16="${BF}" \
  -I"${INC}" -I"${ASCEND_HOME_PATH}/include" -c "${SCRIPT_DIR}/main.cpp" -o main.o
"${BISHENG}" main.o -o fused_test -L. -lk -L"${ASCEND_HOME_PATH}/lib64" -L"${SIMLIB}" \
  -lruntime_camodel -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${SIMLIB}:${BUILD}"

mkdir -p camodel_log; export CAMODEL_LOG_PATH="${BUILD}/camodel_log"
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "==> fd limit: $(ulimit -n)"
echo "==> Run under sim (${SOC})"
./fused_test
