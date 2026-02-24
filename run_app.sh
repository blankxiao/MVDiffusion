#!/usr/bin/env bash
# 启动 app 服务（FastAPI + Redis 队列 Worker）
# 注意：这是启动 app/main.py，不是根目录的 server.py。
# 根目录 server.py 用法：uvicorn server:app --host 0.0.0.0 --port 9000

set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# 用 exec 让当前 shell 被 Python 进程替换，方便收 SIGTERM 优雅退出。
# 等价于在 Python 里调 uvicorn.run("app.main:app", host=HTTP_HOST, port=HTTP_PORT)，
# 即 uvicorn app.main:app --host "${HTTP_HOST:-0.0.0.0}" --port "${HTTP_PORT:-9000}"
exec python -m app.main
