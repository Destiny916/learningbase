W1_ACT_CONTAINER_HOME="${W1_ACT_CONTAINER_HOME:-/root/w1_act}"
W1_PI05_POLICY_PATH="${W1_PI05_POLICY_PATH:-${W1_ACT_CONTAINER_HOME}/checkpoints/new_year/purple_pi05_006/pretrained_model}"

python3 "${W1_ACT_CONTAINER_HOME}/inference_codes/remote_select_action_server_new_pi05.py"   \
    --host 0.0.0.0 \
    --port 8899   \
    --policy_path "${W1_PI05_POLICY_PATH}"  \
    --device cuda \
    --task "purple candy" --robot_type w1
