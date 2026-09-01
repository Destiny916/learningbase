# W1 Pro 腕部相机 USB 延长线连接补充说明

本说明记录使用 USB 延长线连接左右腕部 RealSense 时的启动与验收方法。

## 固定映射

当前已验证的物理映射不是按 `camera_l`/`camera_r` 字面判断：

| 物理位置 | 序列号 | ROS 彩色 resize 话题 |
| --- | --- | --- |
| 左腕部 | `412622271335` | `/camera_r/color/image_rect_raw_resized` |
| 右腕部 | `412622273406` | `/camera_l/color/image_rect_raw_resized` |

延长线重新插拔后 Bus/Device 编号可能变化，必须按序列号确认相机身份。

本机实际映射为：`412622271335` 是物理左腕部，使用 `/camera_r`；
`412622273406` 是物理右腕部，使用 `/camera_l`。因此不能根据节点名中的
`l/r` 直接推断机器人左右，必须以序列号和现场画面复核。

## 本次代码修改

PC2 修改：

```text
/home/dexforce/w1/dexe_mobile_application/startup/detect_realsense.py
/home/dexforce/w1/dexe_mobile_application/startup/run_app.sh
```

修改后按序列号启动固定的 `camera_l`、`camera_r`，只有两只相机同时检测到
才生成启动脚本，避免延长线重新插拔后误生成 `/camera2`。两只相机只启用
color，关闭 depth、infra、align_depth 和 pointcloud。

PC1 `hand_camera_node` 的彩色订阅 QoS 已改成 `BEST_EFFORT`，与 PC2
`wrist_resize_node.py` 的发布 QoS 一致；否则 ROS 会显示订阅者存在，实际却
收不到图像，导致 `hand/left`、`hand/right` 目录为空。

## 启动顺序

1. 延长线接入 PC2，确认两个序列号都能被识别。
2. 保持 PC2 `dexe-app.service` 运行；它负责发布腕部相机 ROS 图像。
3. PC2 仅发布彩色 resize 图像，关闭 depth、infra 和 pointcloud。
4. PC1 停止 Auto/ACT/Map/XWiz 真机客户端，执行 `./change_mode.sh tele`。
5. 验收 PC1 的 `8443/8764/8765/8013`，Quest 通过 ADB reverse 打开
   `https://localhost:8443/`。

## 只保留什么、可以停什么

必须保留：

```text
PC1: dexe-system.service
PC1: dexe-devices.service
PC1: dexe-tele.service
PC1: xwiz-kfc-v1.service（顶部相机源）
PC2: dexe-app.service
PC2: dexe-wrist-resize.service
PC2: camera_l、camera_r、wrist_resize_node.py
```

遥操采集前应停止：

```text
PC1: xwiz-inference-manager.service、ACT 真机客户端
PC2: xwiz-real-client.service、optimized_robot_client
PC2: xwiz-act-server.service、w1-act-server-direct.service
PC2: xwiz_calibration、dexe-xwiz-backend.service
PC2: dexe_application-controller.service（不用 W1 Controller 时）
```

这些进程可能重复订阅相机、占用 CPU/内存/网络或形成第二个动作控制源；只
停止具体 unit，不要用 `pkill -f python`、`pkill -f camera` 或 `pkill -f xwiz`。

## 延长线后验收

PC2 设置 `ROS_DOMAIN_ID=20` 后，使用 `rs-enumerate-devices` 核对序列号，
再测量 `/camera_l/color/image_rect_raw_resized` 和
`/camera_r/color/image_rect_raw_resized`。两路应约 `30 FPS`、`640×360`。
PC1 `hand_camera_node` 的订阅 QoS 必须为 `BEST_EFFORT`，与 PC2 resize
发布端一致；否则会出现有话题但收不到帧。

## 采集完成验收

检查最新 session 的 `hand/left`、`hand/right` 目录均有数量接近的 JPEG，
尺寸为 `640×360`，并在 `metadata.jsonl` 中出现 `hand_left`、`hand_right`；
日志还应出现 `All cameras saved successfully`。
