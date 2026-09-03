#!/usr/bin/env bash
# One-command on-device benchmark for fused_hadamard_quant_b32_a5 on an Ascend 950 (A5).
# Requires a real A5 device, torch + torch_npu, and bisheng (CANN toolkit).
#
# Needs a CANN whose PTO carries MXFP4 (Exp2DStrided in pto/npu/a5/TQuant.hpp):
# 9.1.0 and 9.2.0 both do, 9.0.0 does not.
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" && -z "${ASCEND_HOME_PATH:-}" ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_HOME_PATH
cd "${SCRIPT_DIR}"
exec python3 benchmark.py "$@"
