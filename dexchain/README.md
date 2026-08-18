# DexChain Docker 环境

本目录根据 `install.md` 配置 DexEChain/EmbodiChain。默认使用文档指定的 DexSDK CUDA 12.8 镜像，并从内部 PyPI 安装固定版本：

- `dexechain==0.1.6`
- `embodichain==0.2.4`

## 当前主机配置

- Docker registry `192.168.3.13:5000` 已加入 `insecure-registries`。
- 已安装 Docker Compose v2 和 NVIDIA Container Toolkit 1.20.0。
- Docker 下载并发设为 1，以降低内部 registry 大层并发下载失败的概率。
- GPU 使用 CDI：`nvidia.com/gpu=0`。本机 legacy `--gpus` 路径会导致 NVML `Unknown Error`，不要替换为 `--gpus all`。

## 快速开始

```bash
cd /home/wengyikun/workplace/joint_songling/dexchain

# 1. 拉取约 35 GB 的基础镜像
./scripts/pull_image.sh

# 2. 构建并启动容器
./scripts/start.sh

# 3. 验证 GPU、Python 与两个包
./scripts/verify.sh

# 4. 进入容器
docker compose exec dexchain bash
```

停止容器：

```bash
docker compose down
```

## 配置

运行配置位于 `.env`。模板为 `.env.example`，主要变量如下：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `BASE_IMAGE` | `192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3` | DexSDK 基础镜像 |
| `DEXCHAIN_IMAGE` | `dexchain:0.1.6` | 本机构建镜像名 |
| `NVIDIA_CDI_DEVICE` | `nvidia.com/gpu=0` | 暴露给容器的单张 GPU |
| `DEXECHAIN_VERSION` | `0.1.6` | DexEChain wheel 版本 |
| `EMBODICHAIN_VERSION` | `0.2.4` | EmbodiChain 包版本 |
| `DISPLAY` | `:0` | X11 显示 |

Compose 使用 host network/IPC，并挂载：

- 当前目录到 `/workspace/dexchain`
- `/tmp/.X11-unix`
- `/usr/share/nvidia`（只读）
- `/usr/share/vulkan`（只读）
- `/dev/dri`

## 源码模式

`install.md` 中的 GitLab 仓库 `http://192.168.3.16/Engine/embodichain.git` 需要登录认证。当前配置不保存用户名、密码或 token，而是使用内部 PyPI 包。

如必须开发源码，请先在主机完成 GitLab credential 登录，再在本目录执行：

```bash
mkdir -p src
git clone http://192.168.3.16/Engine/embodichain.git src/embodichain
```

随后可在容器中执行 `pip install -e /workspace/dexchain/src/embodichain`。

## 故障排查

### Registry 超时

确认内网或 VPN 可到达 registry：

```bash
curl --max-time 10 http://192.168.3.13:5000/v2/
```

当前主机的 `Meta` 代理策略可能覆盖 `tunx` 内网路由。`pull_image.sh` 会调用
`scripts/fix_registry_route.sh`，仅添加一条运行时规则：

```text
to 192.168.3.0/24 lookup main
```

该规则不修改默认互联网路由，重启后会失效，脚本会按需重新添加。

返回 `{}` 后重新执行：

```bash
./scripts/pull_image.sh
```

脚本会检查至少 45 GiB 可用空间。当前基础镜像压缩层约 35.1 GB，解压后需要更多空间。

### GPU 验证

```bash
docker run --rm \
  --device nvidia.com/gpu=0 \
  --entrypoint nvidia-smi \
  ubuntu22_ros2_v2:latest -L
```

不要在本机使用 `--gpus all` 作为验证方式。

### X11

启动脚本会在可用时执行：

```bash
xhost +si:localuser:root
```

如果不运行图形界面，可忽略 `DISPLAY` 和 X11 相关挂载。
