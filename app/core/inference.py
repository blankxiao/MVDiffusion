"""
推理接口：当前为占位实现，后续可接 demo.py 子进程或进程内 MVDiffusion。
支持文生图（text2pano）与图+文外扩（outpaint）两种模式。
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """推理结果。"""
    success: bool
    output_dir: Optional[str] = None
    image_paths: Optional[List[str]] = None
    message: Optional[str] = None


class InferenceService(ABC):
    """
    全景推理抽象接口，可用 demo.py 或进程内模型实现。
    支持两种模式：text2pano（仅文案）、outpaint（文案+参考图）。
    """

    @abstractmethod
    def run(
        self,
        text: str,
        *,
        image_path: Optional[str] = None,
        mode: str = "text2pano",
        gen_video: bool = False,
        text_path: Optional[str] = None,
        timeout_seconds: int = 600,
    ) -> InferenceResult:
        """
        执行推理。文生图仅传 text；图+文外扩传 text 与 image_path。
        """
        pass


class PlaceholderInferenceService(InferenceService):
    """占位实现：固定返回成功与假路径，后续替换为真实推理。"""

    def run(
        self,
        text: str,
        *,
        image_path: Optional[str] = None,
        mode: str = "text2pano",
        gen_video: bool = False,
        text_path: Optional[str] = None,
        timeout_seconds: int = 600,
    ) -> InferenceResult:
        logger.info("占位推理 mode=%s text 长度=%d image_path=%s", mode, len(text or ""), image_path)
        return InferenceResult(
            success=True,
            output_dir="/app/outputs/placeholder",
            image_paths=[f"/app/outputs/placeholder/{i}.png" for i in range(8)],
            message="占位实现，请替换为真实 MVDiffusion 推理。",
        )
