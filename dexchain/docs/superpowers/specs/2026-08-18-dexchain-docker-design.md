# DexChain Docker 设计

## 目标

在当前单卡 RTX 5060 主机上提供可重复执行的 DexEChain/EmbodiChain Docker 环境，不在文件中保存 GitLab 凭据，并保留 `install.md` 原始说明。

## 方案

- 基础镜像默认使用文档指定的 `192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3`。
- 通过内部 PyPI 安装 `dexechain==0.1.6` 与 `embodichain==0.2.4`，绕过需要交互登录的 GitLab clone。
- GPU 使用已验证的 Docker CDI 设备 `nvidia.com/gpu=0`，不使用本机失效的 legacy `--gpus` 路径。
- 提供 Dockerfile、Compose、环境变量样例、拉取/启动/验证脚本和静态配置测试。
- 默认使用 host network、host IPC、X11、Vulkan/NVIDIA 只读挂载和 `/dev/dri`，匹配原安装说明。

## 错误处理

- 拉取脚本先验证 Docker、registry 和磁盘，再执行 `docker pull`。
- 启动脚本检查 Compose、CDI GPU、X11 目录和必需配置。
- 验证脚本在容器内检查 GPU、Python，以及 `dexechain`、`embodichain` 导入。
- 大镜像拉取失败时不删除用户现有镜像或容器，只返回非零状态并保留 Docker 自身可恢复缓存。

## 验收

- `docker compose config` 成功。
- 静态测试确认镜像、包版本、CDI GPU、挂载和验证命令均存在。
- 基础镜像拉取成功后，镜像构建成功。
- 容器内能列出 RTX 5060，并成功导入两个 Python 包。
