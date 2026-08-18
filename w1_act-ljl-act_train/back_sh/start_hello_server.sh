W1_ACT_CONTAINER_HOME="${W1_ACT_CONTAINER_HOME:-/root/w1_act}"
W1_HELLO_POLICY_PATH="${W1_HELLO_POLICY_PATH:-${W1_ACT_CONTAINER_HOME}/checkpoints/social/hello_0309_082/pretrained_model}"

export PYTHONPATH="${W1_ACT_CONTAINER_HOME}:${PYTHONPATH}"

python3 "${W1_ACT_CONTAINER_HOME}/inference_codes/remote_select_action_server_new_act.py"   \
    --host 0.0.0.0 \
    --port 2222   \
    --policy_path "${W1_HELLO_POLICY_PATH}"  \
    --device cuda
