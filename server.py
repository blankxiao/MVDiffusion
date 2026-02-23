"""
MVDiffusion HTTP 封装：供 Java (simple-market) 等通过内网调用。
运行方式（与 demo.py 同环境）：
  uvicorn server:app --host 0.0.0.0 --port 9000
接口：POST /api/inference  Body: {"text": "A beautiful kitchen..."}
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


class InferenceResponse(BaseModel):
    success: bool
    output_dir: Optional[str] = None
    image_paths: Optional[List[str]] = None
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
        if not all(os.path.isfile(p) for p in image_paths):
            return False, output_dir, None, "Some output images missing after run."
        return True, output_dir, image_paths, ""
    except subprocess.TimeoutExpired:
        return False, None, None, "Inference timeout (600s)."
    except Exception as e:
        return False, None, None, str(e)


@app.post("/api/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    """接收文本，调用 demo.py 生成全景图，返回输出目录与图片路径列表。"""
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    success, output_dir, image_paths, message = run_demo(text)
    if success:
        return InferenceResponse(success=True, output_dir=output_dir, image_paths=image_paths)
    return InferenceResponse(success=False, message=message or "Inference failed.")


@app.get("/health")
def health():
    return {"status": "ok"}
