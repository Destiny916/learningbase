# 转换发现

- `/data/popcorn/0827` 当前保留 50 个完整 episode（0–49），每个有四路 JPEG、metadata、pose、VR JSONL。
- 四路相机均约 30 FPS；目标应从 metadata 按时间戳对齐到 `head/right` 时间轴，再取最近/线性插值姿态。
- 源 pose 每帧包含身体、双手 6D 和两个 gripper；目标 19D 只取 17 个身体关节加两个 gripper。
- 需要确认本地 LeRobot `LeRobotDataset.create(..., use_videos=True)` 的 v3.1 API 及视频编码依赖。
- 用户确认可继续使用 LeRobot v3.0；本地 `lerobot` 0.6.1 的 `CODEBASE_VERSION` 为 `v3.0`。
- 输出时间轴采用每个 episode 的 `head/right` metadata；左右腕部图像按最近时间戳匹配，姿态按时间戳逐维线性插值，超出 pose 边界时端点夹紧。
- 三路视频 feature 为 `observation.images.cam_high_right`、`observation.images.cam_hand_left`、`observation.images.cam_hand_right`；对应源图分别为 1920x1080、640x360、640x360 RGB，输出均为 224x224。
- 用户补充预处理：`head/right` 居中上下补黑边到 1920x1920 后 resize 224x224；两路 wrist 先 resize 360x360，再 resize 224x224。
- 正式转换使用 H.264 `ultrafast`、CRF 23，避免默认 SVT-AV1 编码过慢；数据语义和视频尺寸不变。
- `observation.state` 与 `action` 都保存 19D 绝对值；源数据没有独立 action 轨迹，不能凭空制造未来 action。
- 正式输出 `/data/popcorn/0827_lerobot_v30` 已完成：50 episodes / 95,725 frames / LeRobot v3.0。
- 三路视频均为 H.264、224x224、30 FPS，每路视频头信息帧数合计均为 95,725。
- 全量 parquet state/action 均为 `(95725, 19)`、全部 finite、逐元素完全相等；episode/global/frame index 连续正确。
- episode 0、7、25、49 重新按源 pose 时间戳插值后与输出最大误差均为 0.0。
- 开头、中间、末尾三组视频抽样与源预处理图的归一化 MAE 为 0.00477–0.01452；头部上下 40 行黑边均小于 0.00005。
- LeRobot 会保留空的 `images/<feature>` 目录；正确验收条件是目录中没有临时图片文件，而不是要求 `images/` 目录不存在。
