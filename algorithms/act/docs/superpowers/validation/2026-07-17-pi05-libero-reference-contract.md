# PI05 LIBERO 官方 v044 参考契约

日期：2026-07-17

## 不可变参考

- 模型：`lerobot/pi05_libero_finetuned_v044`
- Hugging Face revision：`dbf8a3f794a9c4297b44f40b752712f50073d945`
- 模型卡标注数据集：`HuggingFaceVLA/libero`
- API 最后修改时间：`2026-01-22T08:48:43Z`

配置文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `2f6d4b96b032593e1de65b391ee7252a70227bff398ef1b94658fea17818142f` |
| `policy_preprocessor.json` | `3d669ffc18d0364536735d8203076dcc69d3a6523aa5956ccaada3f9e1e6b748` |
| `policy_postprocessor.json` | `37d719a6584600988e6c343306d2eaee0575109f235ad500313577f73e47e8a1` |

## 数据与动作契约

- 两路视觉输入：`observation.images.image`、`observation.images.image2`，原始配置
  特征尺寸为 `3x256x256`，PI05 内部图像分辨率为 `224x224`。
- proprioception：`observation.state`，8D。
- 动作：`action`，7D 连续相对末端动作。
- 模型 chunk：50；官方 OpenPI/LeRobot 评估时执行 horizon：10。
- 评估控制模式：`relative`。
- 官方 checkpoint 使用 `ACTION=MEAN_STD`、`STATE=MEAN_STD`、`VISUAL=IDENTITY`。

## 官方规模参考

LeRobot LIBERO 文档的 PI05 复现设置为：从 `pi05_libero` 出发，使用
`HuggingFaceVLA/libero` 额外训练 6000 step，BF16，全局 batch size 256（8 张 H100）。
评估覆盖 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，每个 task
10 episode，共 400 episode，`n_action_steps=10`。

本项目的参考 launcher 固定上述评估协议。只有在保持 checkpoint、数据 revision、
相对控制、执行 horizon 和 episode 数一致时，成功率才允许与 v044 比较。
