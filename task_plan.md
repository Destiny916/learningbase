# Popcorn 0827 转换为 LeRobot v3.0

- [completed] 审计源数据、LeRobot 版本和已有转换脚本
- [completed] 明确三路视频与 19D 状态/动作时间对齐规则
- [completed] 编写只读源、独立输出的转换脚本
- [completed] 先跑单 episode smoke test，检查视频、parquet、19D 字段
- [completed] 执行全量转换并验证 episode/frame/video/统计结果

## Smoke test 结果

- `/data/popcorn/0827_lerobot_v30_smoke`：1 episode / 1,944 frames。
- `meta/info.json` 为 v3.0，19D state/action 顺序正确，三路视频均成功编码并可用 PyAV 解码。
- 发现 LeRobot 保存 episode 时会为视频统计加载整段视频，内存峰值约 13 GB；已改为逐帧源读取并串行 episode 编码，避免源图像全部驻留内存。
- 由于用户补充了 224x224 预处理，旧 smoke/full partial 输出已停止并清理；重新输出将使用三路 224x224 视频。

## 固定契约

- 源目录：`/data/popcorn/0827/episode0`–`episode49`
- 输出目录：新目录，不覆盖源数据
- 视频：`hand/left`、`hand/right`、`head/right`
- 19D 顺序：WAIST、LEFT_J1–J7、NECK1、NECK2、RIGHT_J1–J7、LEFT_GRIPPER、RIGHT_GRIPPER
- `action`：先按当前帧姿态写入同一 19D 值；如源仅有 state/pose，则不伪造未来动作

## 全量结果

- 正式输出：`/data/popcorn/0827_lerobot_v30`
- LeRobot v3.0，50 episodes，95,725 frames，30 FPS。
- 三路 H.264 视频均为 224x224；每路视频帧数均为 95,725。
- 全量 19D state/action 有限且形状正确；当前 `action == observation.state`。
- 完整验收已通过，包括逐 episode 帧数、时间戳姿态插值、视频解码抽样、图像预处理 MAE、头部黑边和临时图片清理。
