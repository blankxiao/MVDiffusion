"""
从环境变量读取配置，适用于 Docker 与 K8s。
"""
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # HTTP 服务
    http_host: str = Field(default="0.0.0.0", description="FastAPI 监听地址")
    http_port: int = Field(default=9000, description="FastAPI 监听端口")

    # Redis 任务队列
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis 连接 URL")
    task_queue: str = Field(default="panorama:task", description="待消费任务队列名")
    result_queue: str = Field(default="panorama:result", description="推理结果队列名")

    # 推理
    inference_timeout_seconds: int = Field(default=600, description="单次推理最大耗时（秒）")

    # 路径（Docker 下挂载卷并设置）
    project_root: str = Field(default="/app", description="项目根目录，含 demo.py、configs、weights、outputs")
    weights_dir: str = Field(default="/app/weights", description="模型权重目录")
    outputs_dir: str = Field(default="/app/outputs", description="推理输出目录")
    hf_home: Optional[str] = Field(default="./cache/huggingface", description="HuggingFace 缓存目录，供 CLIPTokenizer 等使用；相对路径时基于 project_root")

    # 日志
    log_level: str = Field(default="INFO", description="日志级别")


def get_settings() -> Settings:
    return Settings()
