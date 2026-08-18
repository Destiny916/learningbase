W1_ACT_CONTAINER_HOME="${W1_ACT_CONTAINER_HOME:-/root/w1_act}"
W1_SHAKE_POLICY_PATH="${W1_SHAKE_POLICY_PATH:-${W1_ACT_CONTAINER_HOME}/checkpoints/social/shake_0309_054/pretrained_model}"

export PYTHONPATH="${W1_ACT_CONTAINER_HOME}:${PYTHONPATH}"

python3 "${W1_ACT_CONTAINER_HOME}/inference_codes/remote_select_action_server_new_act.py"   \
    --host 0.0.0.0 \
    --port 1111   \
    --policy_path "${W1_SHAKE_POLICY_PATH}"  \
    --device cuda
