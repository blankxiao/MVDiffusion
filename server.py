"""
MVDiffusion HTTP 封装：供 Java (simple-market) 等通过内网调用。
运行方式（与 demo.py 同环境）：
  uvicorn server:app --host 0.0.0.0 --port 9000
接口：POST /api/inference  Body: {"text": "...", "user_id": "可选，用于 OSS 路径"}
推理成功后仅将 pano.png 上传 OSS，OSS 配置从环境变量读取（.env 或系统环境）。
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 项目根目录（server.py 所在目录，即 demo.py 所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_SCRIPT = PROJECT_ROOT / "demo.py"

app = FastAPI(title="MVDiffusion Inference", version="0.1.0")


class InferenceRequest(BaseModel):
    text: str
    user_id: str = "default"


class InferenceResponse(BaseModel):
    success: bool
    output_dir: Optional[str] = None
    image_paths: Optional[List[str]] = None
    pano_oss_url: Optional[str] = None
    message: Optional[str] = None


def run_demo(text: str) -> Tuple[bool, Optional[str], Optional[List[str]], str]:
    """执行 demo.py --text "..." 并解析输出目录。返回 (success, output_dir, image_paths, message)。"""
    if not DEMO_SCRIPT.exists():
        return False, None, None, f"demo.py not found at {DEMO_SCRIPT}"
    # 避免 shell 注入：用列表参数
    cmd = [sys.executable, str(DEMO_SCRIPT), "--text", text]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        # 解析 "saved to the folder: outputs/results--20250101-120000"
        match = re.search(r"saved to the folder:\s*(\S+)", stdout)
        if not match:
            return False, None, None, f"Could not parse output folder. stdout:\n{stdout[:500]}\nstderr:\n{stderr[:500]}"
        output_dir = match.group(1).strip()
        # 若为相对路径，转为绝对路径便于调用方使用
        if not os.path.isabs(output_dir):
            output_dir = str(PROJECT_ROOT / output_dir)
        image_paths = [os.path.join(output_dir, f"{i}.png") for i in range(8)]
        pano_path = os.path.join(output_dir, "pano.png")
        if not all(os.path.isfile(p) for p in image_paths) or not os.path.isfile(pano_path):
            return False, output_dir, None, "Some output images or pano.png missing after run."
        return True, output_dir, image_paths, ""
    except subprocess.TimeoutExpired:
        return False, None, None, "Inference timeout (600s)."
    except Exception as e:
        return False, None, None, str(e)


def _upload_pano_to_oss(local_path: str, user_id: str) -> Optional[str]:
    """从环境变量读取 OSS 配置并上传 pano.png，返回可访问 URL。"""
    endpoint = os.getenv("OSS_ENDPOINT")
    access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
    access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    bucket_name = os.getenv("OSS_BUCKET_NAME")
    bucket_domain = os.getenv("OSS_BUCKET_DOMAIN")
    if not endpoint or not access_key_id or not access_key_secret or not bucket_name:
        return None
    try:
        from app.core.oss_upload import upload_pano_to_oss as do_upload
        return do_upload(
            local_path,
            user_id or "default",
            endpoint=endpoint,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            bucket_name=bucket_name,
            bucket_domain=bucket_domain or None,
        )
    except Exception:
        return None


@app.post("/api/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    """接收文本，调用 demo.py 生成全景图，返回输出目录、图片路径列表及 pano.png 的 OSS URL。"""
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    success, output_dir, image_paths, message = run_demo(text)
    if not success:
        return InferenceResponse(success=False, message=message or "Inference failed.")
    pano_path = os.path.join(output_dir, "pano.png")
    pano_oss_url = _upload_pano_to_oss(pano_path, request.user_id or "default") if os.path.isfile(pano_path) else None
    return InferenceResponse(
        success=True,
        output_dir=output_dir,
        image_paths=image_paths,
        pano_oss_url=pano_oss_url,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
