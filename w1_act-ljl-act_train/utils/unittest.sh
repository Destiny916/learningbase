#!/bin/bash
REPO_ROOT=$(cd $(dirname $(readlink -f $0))/../ && pwd)

TARGET_SCRIPT="${REPO_ROOT}/act_async_infer_distributed_demo/scripts/run_unittest.sh"

if [ ! -f "${TARGET_SCRIPT}" ]; then
    echo "test script is not exist, path：${TARGET_SCRIPT}"
    exit 1
fi

bash "${TARGET_SCRIPT}"

if [ $? != 0 ]; then
    echo "unit test failed!"
    exit 1
fi