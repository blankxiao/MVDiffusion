"""
通过 Redis 队列测试「发送任务 -> Worker 消费 -> 返回结果」的完整流程。

用法（在项目根目录 MVDiffusion 下执行，以便正确加载 .env）:
  cd /path/to/MVDiffusion
  python -m app.test.test_redis_flow
  python -m app.test.test_redis_flow --text "a cozy living room"
  python -m app.test.test_redis_flow --task-id my-task-001 --no-wait   # 只发任务，不等待结果
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# 项目根目录（MVDiffusion），并确保 .env 可被 config 找到
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
# 若当前目录不是项目根，切换到项目根以便 pydantic_settings 读取 .env
if Path.cwd() != _root and (_root / ".env").exists():
    os.chdir(_root)

from redis import Redis

from app.config import get_settings
from app.schemas import ResultMessage, TaskMessage


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 Redis 测试任务入队与结果接收")
    parser.add_argument("--task-id", default=None, help="任务 ID，默认自动生成 UUID")
    parser.add_argument("--text", default="a beautiful sunset over the ocean", help="文生图提示文案")
    parser.add_argument("--user-id", default="test-user", help="用户/业务标识")
    parser.add_argument("--mode", choices=["text2pano", "outpaint"], default="text2pano", help="推理模式")
    parser.add_argument("--no-wait", action="store_true", help="只发送任务，不等待结果队列")
    parser.add_argument("--timeout", type=int, default=None, help="等待结果超时秒数，默认使用配置中的 inference_timeout_seconds+30")
    args = parser.parse_args()

    settings = get_settings()
    task_id = args.task_id or str(uuid.uuid4())

    task = TaskMessage(
        task_id=task_id,
        user_id=args.user_id,
        text=args.text,
        mode=args.mode,
        image_path=None,
        text_path=None,
        gen_video=False,
    )
    payload = task.model_dump_json()

    print(f"Redis URL: {settings.redis_url}")
    print(f"任务队列: {settings.task_queue}  结果队列: {settings.result_queue}")
    print(f"发送任务 task_id={task_id}  mode={args.mode}  text={args.text[:50]}...")
    print()

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        redis.lpush(settings.task_queue, payload)
        print("任务已入队。")
    finally:
        redis.close()

    if args.no_wait:
        print("已使用 --no-wait，退出。请在服务端日志或结果队列中查看结果。")
        return

    # 等待结果：从 result 队列 brpop，超时使用配置推理超时 + 缓冲
    wait_timeout = args.timeout
    if wait_timeout is None:
        wait_timeout = settings.inference_timeout_seconds + 30
    print(f"等待结果（超时 {wait_timeout}s）...")

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        raw = redis.brpop(settings.result_queue, timeout=wait_timeout)
        if not raw:
            print("超时：未在结果队列中收到消息。请确认 Worker 已启动且任务已被消费。")
            sys.exit(1)
        _queue_name, result_json = raw
        result = ResultMessage.model_validate(json.loads(result_json))
        if result.task_id != task_id:
            print(f"警告：收到的是其他任务的结果 task_id={result.task_id}，当前期望 {task_id}")
        print("收到结果:")
        print(f"  task_id:   {result.task_id}")
        print(f"  success:   {result.success}")
        print(f"  output_dir: {result.output_dir}")
        print(f"  image_paths: {result.image_paths}")
        print(f"  pano_oss_url: {result.pano_oss_url}")
        print(f"  message:   {result.message}")
    finally:
        redis.close()


if __name__ == "__main__":
    main()
