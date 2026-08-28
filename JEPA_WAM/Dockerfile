FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/jepa_wam
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python3", "-m", "prismatic.training.train"]
