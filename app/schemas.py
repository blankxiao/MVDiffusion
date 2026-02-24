"""
任务与结果消息的 Pydantic 模型（队列载荷）。
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# 推理模式：文生图 / 图+文外扩（与 demo.py 的 --image_path 有无对应）
InferenceMode = Literal["text2pano", "outpaint"]


class TaskMessage(BaseModel):
    """从任务队列消费的消息。"""
    task_id: str = Field(..., description="任务唯一标识")
    text: str = Field(..., description="全景图生成提示文案")
    mode: InferenceMode = Field(default="text2pano", description="text2pano=文生图, outpaint=图+文外扩")
    image_path: Optional[str] = Field(default=None, description="参考图路径，outpaint 时必填")
    text_path: Optional[str] = Field(default=None, description="8 行多视角 prompt 文件路径，可选")
    gen_video: bool = Field(default=False, description="是否生成视频")


class ResultMessage(BaseModel):
    """推理完成后写入结果队列的消息。"""
    task_id: str = Field(..., description="与 TaskMessage 中的 task_id 一致")
    success: bool = Field(..., description="推理是否成功")
    output_dir: Optional[str] = Field(default=None, description="输出目录路径")
    image_paths: Optional[List[str]] = Field(default=None, description="生成图片路径列表")
    message: Optional[str] = Field(default=None, description="错误或状态信息")


class TestInferenceRequest(BaseModel):
    """测试接口：文生图请求体。"""
    text: str = Field(..., description="全景图提示文案")


class TestInferenceResponse(BaseModel):
    """测试接口：文生图响应。"""
    success: bool = Field(..., description="是否成功")
    output_dir: Optional[str] = Field(default=None, description="输出目录路径")
    image_paths: Optional[List[str]] = Field(default=None, description="生成图片路径列表")
    message: Optional[str] = Field(default=None, description="错误或状态信息")
