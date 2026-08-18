#!/usr/bin/env bash
set -euo pipefail

W1_ACT_HOME="${W1_ACT_HOME:-/home/dexforce/w1/w1_act}"
W1_CLIENT_CONFIG="${W1_CLIENT_CONFIG:-${W1_ACT_HOME}/act_async_infer_distributed_demo/scripts/client/client_config.json}"

python -m act_async_infer_distributed_demo.scripts.client.run_robot_client \
--server_host 192.168.20.99 \
--server_port 8889 \
--model_config "${W1_CLIENT_CONFIG}" \
--control_frequency 8 \
--collect_frequency 10 \
--chunk_size_threshold 0.5 \
--use_lipo \
--max_steps 500 \
--time_infer 0.5 \
--save_actionchunk \
--sample_factor 2.0 \
--is_go_home \
--home_position handbookv2_0409 \
--mode 2
