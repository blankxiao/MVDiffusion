# 使用 NVIDIA CUDA 基础镜像（兼容驱动 30.0.14.7256）
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    exiftool \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 配置 pip 镜像源并安装依赖
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.extra-index-url https://pypi.org/simple && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir PyExifTool

# 创建输出目录
RUN mkdir -p /app/outputs /app/weights

# 设置默认命令
CMD ["/bin/bash"]

