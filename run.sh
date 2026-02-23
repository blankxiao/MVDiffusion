#!/bin/bash
# run.sh

# 创建必要的目录
mkdir -p weights outputs cache/huggingface cache/torch

# 运行容器
docker run --gpus all -it --rm \
  -v ${PWD}/weights:/app/weights \
  -v ${PWD}/outputs:/app/outputs \
  -v ${PWD}/cache/huggingface:/root/.cache/huggingface \
  -v ${PWD}/cache/torch:/root/.cache/torch \
  mvdiffusion:latest \
  python3 demo.py --text "$@"