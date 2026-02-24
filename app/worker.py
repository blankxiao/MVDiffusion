"""
队列 Worker：从 Redis 消费任务、执行推理、将结果写回队列。
在 FastAPI lifespan 中以后台线程启动。
"""
import json
import logging
import threading
import time
from typing import Optional

from redis import Redis

from app.config import Settings, get_settings
from app.core.inference import InferenceResult, InferenceService, PlaceholderInferenceService
from app.schemas import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)

_worker_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None


def _run_worker(
    settings: Settings,
    inference_service: InferenceService,
) -> None:
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    while not (_stop_event and _stop_event.is_set()):
        try:
            # 阻塞等待任务，超时 5 秒以便检查 stop_event
            raw = redis.brpop(settings.task_queue, timeout=5)
            if not raw:
                continue
            _queue_name, payload = raw
            try:
                data = json.loads(payload)
                task = TaskMessage.model_validate(data)
            except Exception as e:
                logger.exception("非法任务载荷: %s", e)
                continue
            logger.info("处理任务 task_id=%s mode=%s", task.task_id, task.mode)
            result: InferenceResult = inference_service.run(
                task.text,
                image_path=task.image_path,
                mode=task.mode,
                gen_video=task.gen_video,
                text_path=task.text_path,
                timeout_seconds=settings.inference_timeout_seconds,
            )
            result_msg = ResultMessage(
                task_id=task.task_id,
                success=result.success,
                output_dir=result.output_dir,
                image_paths=result.image_paths,
                message=result.message,
            )
            redis.lpush(settings.result_queue, result_msg.model_dump_json())
            logger.info("任务 task_id=%s 完成 success=%s", task.task_id, result.success)
        except Exception as e:
            logger.exception("Worker 循环异常: %s", e)
            time.sleep(5)
    redis.close()
    logger.info("Worker 线程退出")


def start_worker(
    settings: Optional[Settings] = None,
    inference_service: Optional[InferenceService] = None,
) -> None:
    """在后台线程中启动队列 Worker。"""
    global _worker_thread, _stop_event
    if _worker_thread is not None and _worker_thread.is_alive():
        logger.warning("Worker 已在运行")
        return
    settings = settings or get_settings()
    inference_service = inference_service or PlaceholderInferenceService()
    _stop_event = threading.Event()
    _worker_thread = threading.Thread(
        target=_run_worker,
        args=(settings, inference_service),
        daemon=True,
        name="queue-worker",
    )
    _worker_thread.start()
    logger.info("Worker 线程已启动 (queue=%s)", settings.task_queue)


def stop_worker() -> None:
    """通知 Worker 线程退出。"""
    global _stop_event
    if _stop_event:
        _stop_event.set()
