#!/bin/bash

echo "================================================"
echo "MVDiffusion 环境设置脚本"
echo "================================================"

# 检查GPU
echo "[1/4] 检查GPU状态..."
if nvidia-smi > /dev/null 2>&1; then
    echo "✓ GPU检测成功!"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "✗ 未检测到GPU，请确保使用GPU实例!"
    exit 1
fi

# 创建weights目录
echo ""
echo "[2/4] 下载模型权重..."
mkdir -p /root/autodl-tmp/MVDiffusion/weights
cd /root/autodl-tmp/MVDiffusion/weights

# 下载pano模型（必需）
if [ ! -f "pano.ckpt" ]; then
    echo "下载 pano.ckpt ..."
    wget --no-check-certificate -c "https://www.dropbox.com/scl/fi/yx9e0lj4fwtm9xh2wlhhg/pano.ckpt?rlkey=kowqygw7vt64r3maijk8klfl0&dl=1" -O pano.ckpt
fi

# 验证下载
if [ -s "pano.ckpt" ]; then
    echo "✓ pano.ckpt 下载成功! ($(ls -lh pano.ckpt | awk '{print $5}'))"
else
    echo "✗ pano.ckpt 下载失败，请手动下载"
    echo "下载链接: https://www.dropbox.com/scl/fi/yx9e0lj4fwtm9xh2wlhhg/pano.ckpt?rlkey=kowqygw7vt64r3maijk8klfl0&dl=0"
fi

# 检查Python环境
echo ""
echo "[3/4] 验证Python环境..."
cd /root/autodl-tmp/MVDiffusion
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 运行demo（使用项目内 cache，避免加载不到 tokenizer）
echo ""
echo "[4/4] 运行Demo..."
echo "================================================"
cd /root/autodl-tmp/MVDiffusion
export HF_HOME="/root/autodl-tmp/MVDiffusion/cache/huggingface"
export OMP_NUM_THREADS=8
python demo.py --text "This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop."

echo ""
echo "================================================"
echo "完成! 生成的图像保存在当前目录"
echo "================================================"
