# 如何使用 Stereo ACT 训练双臂数据

本文说明当前项目中基于 LeRobot ACT 的顶视双目（Stereo）训练方式。Stereo 模式保留 LeRobot 原有的 ACT Transformer、VAE、chunk action head、归一化和相对关节动作流程；仅把原来的多相机 ResNet 特征提取替换为“顶视双目融合 + 腕部 RGB/RGBD 特征”的视觉前端。

当前代码位于：

```text
/home/wengyikun/workplace/joint_songling/lerobot
```

当前六路 RGBD Stereo ACT 启动器：

```text
run_scripts/launch_act_730_endpoint20_relative_stereo_rgbd_gpu7.sh
```

## 1. 适用的数据格式

当前启动器对应的数据集：

```text
/data/wengyikun/datasets/joint_songling/730_subtask_doubletop_rgbd_endpoint20
```

数据集必须是 LeRobot v3.0，且当前这份数据满足：

```text
fps: 25
episodes: 39
frames: 4422
state: 20D
action: 14D
```

Stereo RGBD 模式要求下面六个图像键。键名、通道数、尺寸必须完全一致；启动器的 preflight 会在训练前检查它们。

```text
observation.images.top_left             RGB,   [3, 405, 720]
observation.images.top_right            RGB,   [3, 405, 720]
observation.images.gripper_left         RGB,   [3, 480, 640]
observation.images.gripper_right        RGB,   [3, 480, 640]
observation.images.gripper_left_depth   depth, [1, 480, 640]
observation.images.gripper_right_depth  depth, [1, 480, 640]
```

这里的 `top_left/top_right` 是一对固定外部顶视双目相机。两路腕部 RGB 分别对应左、右夹爪相机；两路 depth 分别对应相同侧的腕部深度相机。

## 2. 图像实际如何进入模型

图像先经过 ACT 的确定性预处理，然后才进入 Stereo 视觉编码器。实现位于：

```text
src/lerobot/policies/image_preprocessing.py
src/lerobot/policies/act/modeling_act.py
```

### 顶视双目 RGB

每一张 `405x720` 顶视图按下面的规则处理：

```text
405x720
-> 上方补 157 像素黑边、下方补 158 像素黑边
-> 720x720
-> bilinear resize
-> 224x224 RGB
```

因此顶视图不是中心裁剪成 `405x405`，而是保持完整的 720 像素水平视野，通过上下黑边补成正方形后再缩放。这样左右双目图保留相同的水平几何范围。

### 腕部 RGB 和 depth

每张 `480x640` 腕部图（RGB 与 depth 均相同）按下面规则处理：

```text
480x640
-> 水平中心裁剪 [80:560]
-> 480x480
-> bilinear resize
-> 224x224
```

所以深度图也会以 `224x224` 进入深度编码器，并不会保持原始 `480x640` 分辨率。

### 不做随机图像增强

启动参数：

```text
--dataset.image_transforms.enable=false
```

它关闭随机颜色、几何等数据增强，但不会关闭上面固定的补黑边、中心裁剪和 resize。固定预处理由模型内部的 `camera_crop_resize_torch(...)` 完成，训练和推理都应走同一套规则。

## 3. Stereo 视觉融合结构

开启方式：

```text
--policy.stereo_visual_mode=stereo_top_rgbd
```

实现文件：

```text
src/lerobot/policies/act/stereo_visual.py
```

顶视双目融合流程如下：

```text
top_left  224x224 RGB -> 共享、可训练的 ResNet18 -> left CNN feature [B,512,7,7]
top_right 224x224 RGB -> 共享、可训练的 ResNet18 -> right CNN feature [B,512,7,7]

top_left/top_right -> 冻结 DINOv2 ViT-S/14 -> patch feature
                        -> 可训练投影、下采样并对齐到 7x7

每一眼：ResNet feature + DINO feature -> 可训练 1x1 projection -> 256D stereo token

两层 Stereo Transformer：
1. 左、右各自 self-attention
2. 左查询右、右查询左的双向 cross-attention
3. 对 query/key 加入 2D RoPE 位置编码
4. 残差 MLP

left token + right token -> 拼接 -> 1x1 projection
                         -> 融合顶视 feature [B,512,7,7]
```

`top_left` 与 `top_right` 会在 Stereo Transformer 中显式交互，因此不是将两张图简单拼接后独立处理。融合完成后，左右图不再以两个独立 token map 交给 ACT，而是作为一个融合后的顶视特征图输入 ACT。

腕部部分流程：

```text
gripper_left RGB  -> 可训练 ResNet18 -> left RGB feature [B,512,7,7]
gripper_right RGB -> 可训练 ResNet18 -> right RGB feature [B,512,7,7]

left/right depth [1,224,224]
-> 按有效距离范围 clip、映射到 [0,1]、复制为 3 通道
-> 独立且可训练的 depth ResNet18
-> 与同侧 RGB feature 拼接
-> 可训练 1x1 fusion
-> left/right wrist feature [B,512,7,7]
```

深度有效范围固定为：

```text
left wrist depth:  [0.07 m, 0.90 m]
right wrist depth: [0.07 m, 0.60 m]
```

最后只有三张特征图进入 LeRobot ACT 主干：

```text
fused top stereo feature
left wrist RGBD feature
right wrist RGBD feature
-> 原 ACT encoder / VAE / decoder / 14D action chunk head
```

### DINOv2 的训练状态

顶视图额外使用 `dinov2_vits14`。它遵循 StereoPolicy 的外部视角先验思路：DINOv2 始终为 `eval()`、`requires_grad=False` 并在 `torch.no_grad()` 中执行，因此不反向更新；ResNet18、DINO 特征投影、Stereo Transformer、depth ResNet18、RGBD 融合层以及 ACT 本体均参与训练。

本地 DINO 权重目录由环境变量指定，默认是：

```text
DINO_V2_REPO=/data/wengyikun/models/dinov2
```

## 4. state 与 action 的相对位姿语义

当前 `state=20D`：

```text
[0:6]    left joints: q_left(t) - q_left(t-1)       相对关节增量
[6]      left gripper: g_left(t)                    绝对开度
[7:13]   right joints: q_right(t) - q_right(t-1)    相对关节增量
[13]     right gripper: g_right(t)                  绝对开度
[14:17]  right endpoint xyz: p_right(t)             绝对位置
[17:20]  left endpoint xyz: p_left(t)               绝对位置
```

当前 `action=14D`，训练目标是 16 帧 action chunk：

```text
[0:6]    left joint action at horizon k: q_left(t+k) - q_left(t)
[6]      left gripper action at horizon k: g_left(t+k)
[7:13]   right joint action at horizon k: q_right(t+k) - q_right(t)
[13]     right gripper action at horizon k: g_right(t+k)
```

换言之，12 个关节维度是相对当前时刻 `t` 的 joint delta；两个夹爪始终是绝对开度。末端 xyz 仅作为 state 条件输入，不出现在当前 14D action 内，且保持绝对坐标。

对应的关键参数为：

```text
joint_representation=relative
condition_on_state=true
chunk_size=16
n_action_steps=16
gripper_indices=[6,13]
state_gripper_indices=[6,13]
state_absolute_indices=[14,15,16,17,18,19]
state_position_indices=[14,15,16,17,18,19]
```

## 5. q01/q99 归一化和噪声

本数据集只可使用自己计算的统计量：

```text
/data/wengyikun/datasets/joint_songling/730_subtask_doubletop_rgbd_endpoint20/
  normalization_relative_state20_action14_xyz_absolute_chunk16_v2/
    relative_state_q01_q99.json
    relative_action_chunk16_q01_q99.json
    relative_stats_manifest.json
```

其中：

```text
relative_state_q01_q99.json
  对 20D state 的实际表示统计：12D joint delta、2D absolute gripper、6D absolute xyz。

relative_action_chunk16_q01_q99.json
  对每个 horizon 的 14D action chunk 实际表示统计：12D relative joint delta、2D absolute gripper。
```

`relative_stats_manifest.json` 必须满足：

```text
format_version=3
state_gripper_indices=[6,13]
state_absolute_indices=[14,15,16,17,18,19]
```

训练时，state/action 先按上述 q01/q99 映射到归一化空间，训练 loss 在归一化后的 action 上计算。`clip_quantiles=true` 会在归一化前将超出 q01/q99 的值裁剪到边界。

当前训练 state 噪声只作用于训练 batch，不改 action 标签：

```text
关节 state delta 噪声: 0.003 rad
末端 xyz state 噪声:  0.003 m
夹爪 state 噪声:      0.001 m
```

如果配置验证集并记录真实量纲 MSE/RMSE，预测与真实 action 都必须先使用同一份 `relative_action_chunk16_q01_q99.json` 反归一化；不能只反归一化预测或只反归一化标签。

## 6. 启动参数

启动器默认参数：

```text
batch_size=32
gradient_accumulation_steps=1
steps=100000
save_freq=10000
num_workers=8
eval_steps=0
mixed_precision=bf16
```

`eval_steps=0` 表示当前数据集没有 train/test 划分，因此本次训练不会生成 validation loss 或每关节 validation MSE；只记录训练 loss。若要比较模型，必须先固定 episode 划分并配置验证集，再启用 validation。

在远端以单张 GPU（例如 GPU7）启动：

```bash
ssh -p 50210 wengyikun@183.230.224.121

docker run -d \
  --name act_730_endpoint20_relative_stereo_rgbd_gpu7 \
  --gpus "device=7" \
  --ipc=host --shm-size=32g \
  -e HOME=/home/wengyikun \
  -e PYTHONPATH=/workspace/lerobot/src \
  -v /data/wengyikun:/data/wengyikun \
  -v /home/wengyikun/lerobot_endpoint20_rgbd:/workspace/lerobot \
  -w /workspace/lerobot \
  lerobot-pi05-train:20260706 \
  bash run_scripts/launch_act_730_endpoint20_relative_stereo_rgbd_gpu7.sh
```

容器挂载源必须包含 `src/lerobot/policies/act/stereo_visual.py` 以及该 launcher。训练前不要把 `/home/wengyikun/lerobot`、`/data/wengyikun/lerobot_endpoint20_rgbd` 和本地工作区的不同版本混用。

## 7. 训练前检查

进入容器或在同一挂载源执行：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path('/data/wengyikun/datasets/joint_songling/730_subtask_doubletop_rgbd_endpoint20')
stats = root / 'normalization_relative_state20_action14_xyz_absolute_chunk16_v2'
info = json.loads((root / 'meta/info.json').read_text())
manifest = json.loads((stats / 'relative_stats_manifest.json').read_text())
print('LeRobot:', info['format_version'], 'fps:', info['fps'])
print('state:', info['features']['observation.state']['shape'])
print('action:', info['features']['action']['shape'])
for key in sorted(k for k in info['features'] if k.startswith('observation.images.')):
    print(key, info['features'][key]['shape'])
print('manifest:', manifest)
PY
```

预期至少确认：

```text
format_version=3.0
fps=25
state=[20]
action=[14]
六个图像键及其形状正确
state_gripper_indices=[6,13]
state_absolute_indices=[14,15,16,17,18,19]
```

启动器本身也会在 `accelerate launch` 前执行同等检查，检查失败会直接退出，不会启动错误训练。

## 8. 监控、停止和恢复

检查容器、日志、GPU 和 checkpoint：

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
docker logs --tail 150 act_730_endpoint20_relative_stereo_rgbd_gpu7
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
find /data/wengyikun/outputs/act_730_endpoint20_relative_stereo_rgbd_chunk16_b32_100k_gpu7 \
  -maxdepth 4 -type f | sort
```

正常启动日志应依次出现：

```text
dataset preflight: v3, fps=25, state=20D, action=14D, images=6, stats=relative_joint_v3
Effective batch size: 32 x 1 x 1 = 32
Start offline training
step=10 train/loss=...
```

checkpoint 预期为：

```text
10000, 20000, 30000, ..., 100000
```

停止指定训练：

```bash
docker stop -t 30 act_730_endpoint20_relative_stereo_rgbd_gpu7
```

恢复时必须确保 checkpoint、数据集、统计量目录、`stereo_visual_mode=stereo_top_rgbd`、state/action 维度与原训练完全一致；使用新的输出目录时则是从基础 ACT 权重重新训练，而不是恢复旧权重。

## 9. 常见错误

### 顶视图没有黑边

训练或推理时若直接自行 resize、或误用普通 `center_crop_resize_torch(...)`，就会看不到上下黑边，且几何关系与训练不一致。必须保留图像键 `top_left/top_right`，使其命中 `camera_crop_resize_torch(...)` 的顶视补边规则。

### depth 图像被当作 RGB

depth 原始输入必须是 `[1,H,W]`、单位为米。Stereo RGBD 分支会在模型中做有效距离 clip、归一化和三通道复制；不要在数据集中将 depth 伪彩色化为 RGB，也不要将它错误声明为 `[3,H,W]`。

### six inputs 不是 six ACT tokens

六路原始图像不会直接以六个独立特征图送给 ACT。`stereo_top_rgbd` 最终只输出三张 512x7x7 feature map：融合顶视、左腕 RGBD、右腕 RGBD。这是设计行为，不是丢失任一路相机。

### q01/q99 混用

不能使用其它数据集、绝对关节动作或不同 chunk size 计算出的统计量。统计量与“state/action 的表示方式、维度、gripper 索引、absolute state 索引、chunk_size”绑定；任意一项改变都必须重新计算。

## 10. 关键源码索引

```text
src/lerobot/policies/act/stereo_visual.py
  StereoACTVisual：顶视双目、DINOv2、深度融合。

src/lerobot/policies/act/modeling_act.py
  ACTPolicy：固定图像预处理；把视觉特征送进原 ACT encoder/VAE/action head。

src/lerobot/policies/image_preprocessing.py
  camera_crop_resize_torch：顶部补黑边、腕部中心裁剪、resize。

run_scripts/launch_act_730_endpoint20_relative_stereo_rgbd_gpu7.sh
  数据集 preflight、统计量路径和全部训练参数。
```
