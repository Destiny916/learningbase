
# W1-ACT分离机器异步推理demo启动方式

## 1. 推理前的必须启动项
### 1.1 KFC相机
```bash
source  ~/w1/install/setup.bash
#查看KFC相机进程并杀死以避免调用冲突
ps  -ef | grep  camera
kill -9 pid
```
### 1.2 腕部相机
```bash
#在PC2上，打开终端
python ~/dual_d405_capture.py
```

## 2. server启动

### 下载对应的ACT pretrained model，配置server_start.sh
```bash
cd w1_act
sh  act_async_infer_distributed_demo/scripts/server/server_start.sh
```
或者
```bash
cd w1_act
python -m act_async_infer_distributed_demo.scripts.server.run_policy_server \
--host 0.0.0.0 \
--port 8889 \
--model_config /home/workspace/Documents/w1_act/act_async_infer_distributed_demo/scripts/server/server_config.json \
--inference_latency 0.3 \
--use_smooth
```
## 3. client启动
```bash
cd w1_act
sh  act_async_infer_distributed_demo/scripts/client/client_start.sh
```
或者
```bash
cd w1_act
python -m act_async_infer_distributed_demo.scripts.client.run_robot_client \
--server_host 192.168.20.21 \
--server_port 8891 \
--model_config /home/dexforce/Documents/w1_act/act_async_infer_distributed_demo/scripts/client/client_config.json \
--control_frequency 8 \
--collect_frequency 5 \
--chunk_size_threshold 0.5 \
--max_steps 150 \
--use_lipo \
--time_infer 1.0 \
--save_actionchunk \
--is_go_home \
--home_position handbookv2_0409 \
--sample_factor 2.0 \
--mode 2
```

## 4. 语音启动
```bash
cd ~/Documenst
# 首次部署语音
python3 -m pip install virtualenv
python3 -m virtualenv venv
source ~/Documents/venv/bin/activate
pip install opencv-python
pip install cvxpy
pip uninstall numpy && pip install numpy==1.26.4
# 非首次部署
source ~/Documents/venv/bin/activate

cd w1_act/
python -m act_async_infer_distributed_demo.vla_control --prompt act_async_infer_distributed_demo/vla_control_instruction.json
```
  

参数说明：
- model_config：模型的推理配置文件。
- max_steps：推理的最大步数。
- control_frequency：控制频率，单位是Hz。
- collect_frequency：观测收集频率，单位是Hz。
- chunk_size_threshold：异步推理触发阈值，当推理时候的剩余要执行的动作数占模型推理所预测的一批动作的数量的比例小于这个值时，会触发模型进行推理下一批新的动作。如果该值为0.0那么就等价于同步推理，如果执行完一批动作的时间远小于模型推理延迟，那么也等价于同步推理，也会出现模型停滞在原地等待的现象，默认为0.5。
- use_lipo：是否对两段轨迹做平滑。
- time_infer：标准的推理延迟。可以通过跑一次推理观察客户端结尾inference delay zones打印的值粗略得到。
- sample_factor：对动作队列进行插值的因子，2.0即将动作数量乘2，可以使得机器人运动更平稳些。
- mode：值为1时客户端进入仿真模式不会直接控制机器人，值为2时开启真机推理。
- is_go_home：开始推理时是否回原点。
- home_position：填入PC1的/home/dexforce/w1/dexe_mobile_application/script/records路径下的home点位置json文件的文件名。
- save_input：服务器开启save_input，可以保存推理数据
- save_vis_images：服务器开启save_vis_images，可以保存推理的图像可视化数据
- save_joint_change：服务器开启save_joint_change，可以保存查看推理的joint的变化图