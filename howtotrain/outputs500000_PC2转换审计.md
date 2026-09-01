# outputs/500000 → PC2 ACT 直连转换审计

## 结论

`outputs/500000/pretrained_model` 是 `type=act`、`chunk_size=16` 的 LeRobot ACT
checkpoint。它保持原始 16 步权重不变，由 PC2 runtime 保持为 16 个动作点，PC1
直接执行这 16 帧；不做插值，也不把模型权重伪造为 100 步。

转换产物由以下命令生成：

```bash
PYTHONPATH=w1_act-ljl-act_train python3 -m xwiz_act_server.checkpoint_converter \
  outputs/500000 \
  outputs/500000_pc2 \
  --target-horizon 100
```

源目录不覆盖。产物包含原 checkpoint 文件和 `xwiz_conversion.json`。

## 动作合同

训练配置和数据转换脚本的 19D 顺序为：

```text
WAIST,
LEFT_J1..LEFT_J7,
NECK1, NECK2,
RIGHT_J1..RIGHT_J7,
LEFT_GRIPPER, RIGHT_GRIPPER
```

`outputs/500000` 没有 `relative_joint_processor`，其 `action` 与 `observation.state`
按绝对值处理，不做“预测增量 + 当前状态”的相对关节还原。左右末端仍是 `0..100`
开合度，由 PC1 的已验证 Linker L6 映射转换。PC1 六关节 POSITION 顺序必须是：

```text
T_MCP, T_CMC_YAW, IF_MCP_PITCH, MF_MCP_PITCH, RF_MCP_PITCH, LF_MCP_PITCH
```

ACT 默认手姿态为 `[0,70,0,0,0,0]`，其中第二项 `T_CMC_YAW=70` 是拇指根部旋转；
左侧 scalar 端点为 `0:[100,0,35,45,47,37]`、`100:[0,70,0,0,0,0]`，
右侧为 `0:[100,65,70,75,100,100]`、`100:[0,70,0,0,0,0]`。

源模型动作块实际是 `16×19`。转换器不修改 `config.json` 的 `chunk_size=16`，runtime
按源长度反归一化后直接把有限的 `16×19` 轨迹交给 PC1。

## 图像合同

训练转换脚本 `scripts/convert_popcorn_0827_to_lerobot.py` 的确定性处理为：

```text
物理右头目：原图 → 上下黑边补成正方形 → 224×224
左腕图：    原图 → 360×360 → 224×224
右腕图：    原图 → 360×360 → 224×224
```

PC1 发送 `cam_high`（左头目）和 `cam_high_r`（右头目）两路；转换协议优先采用
`cam_high_r`，映射为 `observation.images.cam_high_right`。腕图分别映射到
`cam_hand_left` 和 `cam_hand_right`。PC2 再执行 BGR→RGB、确定性图像处理和
`HWC→CHW`。

## 归一化合同

`outputs/500000` 的 preprocessor/postprocessor 均声明：

```text
VISUAL = MEAN_STD
STATE  = MEAN_STD
ACTION = MEAN_STD
```

统计文件含有 `mean/std`，且视觉 mean/std 为 ImageNet 值：
`[0.485, 0.456, 0.406]` 和 `[0.229, 0.224, 0.225]`。文件也保存了 `q01/q99`，但当前
训练合同没有启用 `QUANTILES`；runtime 使用 `mean/std`，不使用 q01/q99，这与训练一致。

若将来切换到声明 `QUANTILES` 的 checkpoint，不能复用本转换器，必须实现并验证
`2*(x-q01)/(q99-q01)-1` 及其逆变换。

## PC2 部署

独立产物已复制为 PC2 上的目录：

```text
/home/dexforce/workspace/outputs/500000_pc2
```

PC2 `start_pc2_direct_server.sh` 通过环境变量支持该路径，当前服务实际监听 `8889`：

```bash
export XWIZ_ACT_POLICY_PATH=/home/dexforce/workspace/outputs/500000_pc2
```

只允许先做严格 checkpoint load 和离线 finite smoke test；未完成前不要向 PC1 的
`8890` 发送 `start`。
