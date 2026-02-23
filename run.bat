@echo off
REM run.bat - Windows 批处理脚本

REM 创建必要的目录
if not exist "weights" mkdir weights
if not exist "outputs" mkdir outputs
if not exist "cache\huggingface" mkdir cache\huggingface
if not exist "cache\torch" mkdir cache\torch

REM 运行容器
docker run --gpus all -it --rm ^
  -v %cd%\weights:/app/weights ^
  -v %cd%\outputs:/app/outputs ^
  -v %cd%\cache\huggingface:/root/.cache/huggingface ^
  -v %cd%\cache\torch:/root/.cache/torch ^
  mvdiffusion:latest ^
  python3 demo.py --text %*