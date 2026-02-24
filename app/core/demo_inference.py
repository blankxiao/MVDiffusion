"""
基于 app 内封装 run_inference 的进程内调用：文生图与图+文外扩，不依赖子进程与 demo.py。
"""
import logging
from typing import List, Optional

from app.config import Settings
from app.core.inference import InferenceResult, InferenceService
from app.core.pano_inference_impl import run_inference as run_pano_inference

logger = logging.getLogger(__name__)


def _run_in_process(
    project_root: str,
    text: str,
    image_path: Optional[str] = None,
    gen_video: bool = False,
    text_path: Optional[str] = None,
) -> tuple[bool, Optional[str], Optional[List[str]], str]:
    """进程内调用 app 内封装的 run_inference，返回 (success, output_dir, image_paths, message)。"""
    try:
        output_dir, image_paths = run_pano_inference(
            project_root=project_root,
            text=text,
            image_path=image_path,
            gen_video=gen_video,
            text_path=text_path,
        )
        return True, output_dir, image_paths, ""
    except Exception as e:
        logger.exception("进程内推理异常: %s", e)
        return False, None, None, str(e)


class DemoInProcessInferenceService(InferenceService):
    """使用 app 内封装的 run_inference，支持文生图与图+文外扩，不修改 demo.py。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
        if mode == "outpaint" and not image_path:
            return InferenceResult(success=False, message="outpaint 模式需提供 image_path")
        logger.info("进程内推理 mode=%s text 长度=%d", mode, len(text or ""))
        success, output_dir, image_paths, message = _run_in_process(
            project_root=self.settings.project_root,
            text=text,
            image_path=image_path,
            gen_video=gen_video,
            text_path=text_path,
        )
        if success:
            return InferenceResult(success=True, output_dir=output_dir, image_paths=image_paths)
        return InferenceResult(success=False, message=message or "推理失败")
