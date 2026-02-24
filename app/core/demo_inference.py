"""
基于 app 内封装 run_inference 的进程内调用：文生图与图+文外扩，不依赖子进程与 demo.py。
推理成功后仅将 pano.png 上传 OSS，OSS 配置从环境变量读取。
"""
import logging
import os
from typing import List, Optional, Tuple

from app.config import Settings
from app.core.inference import InferenceResult, InferenceService
from app.core.oss_upload import upload_pano_to_oss
from app.core.pano_inference_impl import run_inference as run_pano_inference

logger = logging.getLogger(__name__)


def _run_in_process(
    project_root: str,
    text: str,
    image_path: Optional[str] = None,
    gen_video: bool = False,
    text_path: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[List[str]], str]:
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
        user_id: Optional[str] = None,
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
        if not success:
            return InferenceResult(success=False, message=message or "推理失败")

        pano_oss_url: Optional[str] = None
        pano_path = os.path.join(output_dir, "pano.png") if output_dir else None
        if pano_path and self.settings.oss_endpoint and self.settings.oss_access_key_id:
            pano_oss_url = upload_pano_to_oss(
                pano_path,
                user_id or "default",
                endpoint=self.settings.oss_endpoint,
                access_key_id=self.settings.oss_access_key_id,
                access_key_secret=self.settings.oss_access_key_secret or "",
                bucket_name=self.settings.oss_bucket_name or "",
                bucket_domain=self.settings.oss_bucket_domain,
            )
        return InferenceResult(
            success=True,
            output_dir=output_dir,
            image_paths=image_paths,
            pano_oss_url=pano_oss_url,
        )
