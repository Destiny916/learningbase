#!/bin/bash

source ~/.dex_env.sh
DEPLOY_DIR="/home/dexforce/w1"
W1_ACT_HOME="/home/dexforce/w1/w1_act"

if [ -d "${DEPLOY_DIR}/w1_act/checkpoints" ]; then
    export W1_CHECKPOINTS_HOST_HOME="${DEPLOY_DIR}/w1_act/checkpoints"
fi
export W1_ACT_HOME

logger -i -t "dexe_launcher" "enter act mode"

echo "等待所有EtherCAT从站进入OP状态..."
if ! ${DEPLOY_DIR}/dexe_mobile_application/startup/detect_op.sh; then
    echo "错误：EtherCAT从站未就绪"
    exit 1
fi

# 清一下旧容器（没的话也不报错）
docker rm -f act_ros2 >/dev/null 2>&1 || true

# 执行Python脚本
echo "执行Python脚本..."
source ${DEPLOY_DIR}/install/setup.bash
cd ${DEPLOY_DIR}/dexe_mobile_application/script/
python3 slowly_move_to.py records/hello.json 2>&1 | logger -i -t "hello" -p daemon.info &

# 进到 w1_act deb 包安装目录
cd "${W1_ACT_HOME}"

# 启动 Docker
./start_docker.sh

# 进容器里跑策略脚本
docker exec act_ros2 bash -ic 'cd ~/w1_act && ./policy_run_social_new.sh' 2>&1 | logger -i -t "act_ros2" -p daemon.info &

echo "Docker 容器内的 policy_run_social_new.sh 已在 lerobot conda 环境的后台启动。"
echo "所有操作已触发完成！"

wait
