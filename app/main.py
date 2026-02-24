"""
FastAPI 应用：健康/就绪探针与后台队列 Worker。
配置通过环境变量读取，便于 Docker/K8s 部署。
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as health_router, test_router
from app.config import get_settings
from app.core.demo_inference import DemoInProcessInferenceService
from app.worker import start_worker, stop_worker


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    settings = get_settings()
    start_worker(inference_service=DemoInProcessInferenceService(settings))
    yield
    stop_worker()


app = FastAPI(
    title="MVDiffusion Inference Service",
    version="0.1.0",
    description="健康探针与任务队列消费者，用于全景图生成。",
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(test_router)


def run_server() -> None:
    """从环境变量读取配置并启动 uvicorn。"""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.http_host,
        port=settings.http_port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
