# Installation

## System Requirements

The general system requirements can be found in the main [EmbodiChain installation guide](https://dexforce.github.io/EmbodiChain/quick_start/install.html).

---

### Recommended: Install with Docker Setup

We strongly recommend using our pre-configured Docker environment, which contains all necessary dependencies.

1. Pull the Docker image:

```bash
docker pull 192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3
```

2. Run the Docker container (recommended example)

Below is a tested example command that starts the container with GPU access, Vulkan/OpenGL devices and X11 display forwarding. Adjust the container name and volume mounts for your environment.

```bash
docker run -itd \
	--gpus all \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DISABLE_REQUIRE=1 \
	--device /dev/dri \
	--net=host \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	--name "your_container_name" \
	-v /usr/share/nvidia:/usr/share/nvidia \
	-v /usr/share/vulkan:/usr/share/vulkan \
	192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3 \
	/bin/bash
```

More details and info please check: [DexSim](http://192.168.3.120/MixedAI/docs_dev/dexsim/markdown/docker.html#step-5-running-the-docker-container)。

---

### Manual Setup (If Not Using Docker)

PyTorch and its related packages should be installed from the **official PyTorch website**:
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

### Install DexEChain

Clone the EmbodiChain repository:
```bash
git clone http://192.168.3.16/Engine/embodichain.git
```

#### Install the project in development mode:

```bash
```bash
pip install -e .[full] --extra-index-url http://pyp.open3dv.site:2345/simple/ --extra-index-url http://192.168.3.43:8080/simple/ --trusted-host pyp.open3dv.site --trusted-host 192.168.3.43
```
```

This will install all optional dependencies for development, including:
- embodichain
- lerobot

If you want to install `embodichain` in [dev mode](https://dexforce.github.io/EmbodiChain/quick_start/install.html#install-embodichain) as well, please install `embodichain` first then install `dexechain` with:

```bash
# Install embodichain
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site

# Then install dexechain
pip install -e . --extra-index-url http://192.168.3.43:8080/simple/ --trusted-host 192.168.3.43
```

If you want to use some vision toolkits with `glia` dependency, please install the internal Open3D package manually by running the following command:

```bash
pip install http://192.168.3.43:8080/packages/open3d-0.18.0-cp310-cp310-manylinux_2_27_x86_64.whl

# then install glia
pip install glia==0.0.0 --index-url http://192.168.3.43:8080/simple/ --trusted-host 192.168.3.43
```


#### Install the project in deploy mode:

```bash
pip install -e .[deploy]
```


### Verify Installation
To verify that EmbodiChain is installed correctly, run:

```bash
python -c "import dexechain; print(dexechain.__version__)"
```
---
