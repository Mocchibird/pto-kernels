#!/usr/bin/env bash
# Build + run the register-resident fast_hadamard_a5 on the Ascend 950 sim.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"

: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
export ASCEND_HOME_PATH
ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"
SIMULATOR_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
PTO_INC="${ASCEND_HOME_PATH}/aarch64-linux/include"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/runtime/lib64' | paste -sd: -)"
fi
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH:-}"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export LD_LIBRARY_PATH="${SIMULATOR_LIB}:${LD_LIBRARY_PATH}"

: "${N_DIM:=128}"
: "${BATCH_DIM:=256}"
: "${ROWS:=16}"      # 256/16 = 16 tiles -> fills all 16 AIV
LOG2N=0; t=$N_DIM; while [ $t -gt 1 ]; do LOG2N=$((LOG2N+1)); t=$((t/2)); done
INVSQRT=$(python3 -c "import math;print(repr(1.0/math.sqrt($N_DIM)))")

mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
echo "==> Compiling fast_hadamard_a5 (N=${N_DIM} log2=${LOG2N} batch=${BATCH_DIM} rows/tile=${ROWS})"
"${BISHENG}" -xcce --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE \
  -DHAD_N="${N_DIM}" -DHAD_LOG2N="${LOG2N}" -DHAD_INV_SQRT="${INVSQRT}f" -DROWS_PER_TILE="${ROWS}" \
  -O2 -std=c++17 -Wno-ignored-attributes -Wno-macro-redefined -fPIC \
  -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false -Xhost-start -Xhost-end \
  -I"${PTO_INC}" -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_DRIVER_PATH}/kernel/inc" \
  -c "${KERNEL_DIR}/fast_hadamard_a5.cpp" -o k.o
"${BISHENG}" -fPIC -shared --cce-fatobj-link -Wl,-soname,libk.so k.o -o libk.so

echo "==> Compiling host main.cpp"
"${BISHENG}" -xc++ -include stdint.h -include stddef.h -std=c++17 -O2 \
  -Wno-ignored-attributes -Wno-macro-redefined \
  -DHAD_N="${N_DIM}" -DHAD_LOG2N="${LOG2N}" -DBATCH_DIM="${BATCH_DIM}" \
  -I"${PTO_INC}" -I"${ASCEND_HOME_PATH}/include" -c "${SCRIPT_DIR}/main.cpp" -o main.o
"${BISHENG}" main.o -o fht_a5_test -L. -lk -L"${ASCEND_HOME_PATH}/lib64" -L"${SIMULATOR_LIB}" \
  -lruntime_camodel -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${SIMULATOR_LIB}:${BUILD_DIR}"

mkdir -p camodel_log; export CAMODEL_LOG_PATH="${BUILD_DIR}/camodel_log"
echo "==> Running fht_a5_test under sim (${SOC_VERSION})"
./fht_a5_test