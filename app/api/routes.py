"""
HTTP 路由：健康与就绪探针、测试用推理接口。
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from redis import Redis

from app.config import Settings, get_settings
from app.core.demo_inference import DemoInProcessInferenceService
from app.schemas import TestInferenceRequest, TestInferenceResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])
test_router = APIRouter(prefix="/api/test", tags=["test"])


@router.get("/health")
def health():
    """存活探针：进程在运行即可，不检查依赖。"""
    return {"status": "ok"}


@router.get("/ready")
def ready(settings: Settings = Depends(get_settings)):
    """就绪探针：进程与 Redis 均可达，用于 K8s readinessProbe。"""
    try:
        r = Redis.from_url(settings.redis_url)
        r.ping()
        r.close()
        return {"status": "ready"}
    except Exception as e:
        logger.warning("就绪探针失败: %s", e)
        raise HTTPException(status_code=503, detail={"status": "not_ready", "reason": str(e)})


@test_router.post("/inference", response_model=TestInferenceResponse)
def test_inference(
    body: TestInferenceRequest,
    settings: Settings = Depends(get_settings),
) -> TestInferenceResponse:
    """
    测试用 HTTP 接口：仅文生图。提交文案后同步执行 demo.py 并返回结果。
    仅用于联调与验证，生产请走任务队列。
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    service = DemoInProcessInferenceService(settings)
    result = service.run(text, mode="text2pano", timeout_seconds=settings.inference_timeout_seconds)
    return TestInferenceResponse(
        success=result.success,
        output_dir=result.output_dir,
        image_paths=result.image_paths,
        message=result.message,
    )
