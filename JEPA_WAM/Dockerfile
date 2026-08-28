FROM lerobot-pi05-train:20260706

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/jepa_wam
COPY requirements.txt pyproject.toml README.md ./
# The LeRobot training image already provides CUDA PyTorch, PyAV and NumPy.
# Install the JEPA-WAM-only runtime without downgrading that validated stack.
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 transformers==4.57.0 peft==0.13.2 \
    diffusers==0.30.3 draccus==0.8.0 accelerate==1.13.0 \
    swanlab==0.6.1

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python3", "-m", "prismatic.training.train"]
