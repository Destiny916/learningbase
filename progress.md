# 进度

## 2026-08-27

- 已按用户要求删除不完整 episode50–75，仅保留 episode0–49。
- 已开始审计 `/home/wengyikun/workplace/joint_songling` 中 LeRobot 转换实现和本地环境。
- 已确认使用本地 LeRobot 0.6.1 / 数据版本 v3.0；完成视频和 19D 对齐方案设计。
- 已创建 `scripts/convert_popcorn_0827_to_lerobot.py`。
- 单 episode smoke test 通过：三路视频、parquet、19D state/action 和 PyAV 读取均正常；输出在 `/data/popcorn/0827_lerobot_v30_smoke`。
- 初版全量策略曾因整集图像列表占用约 15 GB 内存中断；已修复为逐帧读取，并删除了不完整 smoke 输出后重新验证。
- 用户补充 224x224 图像预处理后，已停止并清理不符合要求的正式输出；脚本已改为三路 224x224、H.264 fast 编码。
- 已使用 8 个 JPEG 解码/resize worker、8 个异步图像写入线程和 64 帧有界 batch 完成正式转换；转换支持 `--resume`。
- 正式输出 `/data/popcorn/0827_lerobot_v30`：50 episodes、95,725 frames、三路 224x224 H.264 视频、19D state/action。
- 完整验收通过：源目录仅 episode0–49；逐 episode 长度一致；全量 parquet、索引、finite、state/action 相等、四个 episode 姿态插值及三组视频解码抽样均通过。
- 最后一个旧验收断言的根因是误把 LeRobot 保留的空 `images/` 目录当作残留；实测临时图片文件数为 0，正式输出无需修改。
