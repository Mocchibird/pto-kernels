#!/usr/bin/env bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_HOME_PATH
cd "$SD"; python3 check256_grid.py 64
