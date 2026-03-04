"""
App 内封装：仿照 demo 的全景推理逻辑，文生图与图+文外扩，供服务进程内直接调用。
不修改 demo.py，所有路径基于 project_root。
"""
import logging
import os
import sys
import yaml
import numpy as np
import cv2
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
torch.manual_seed(0)

logger = logging.getLogger(__name__)


def _ensure_project_root_in_path(project_root: str) -> None:
    """将 project_root 加入 sys.path 首位，以便 import src、generate_video_tool。"""
    root = str(Path(project_root).resolve())
    if root in sys.path:
        return
    sys.path.insert(0, root)


def _import_models():
    from src.lightning_pano_gen import PanoGenerator
    from src.lightning_pano_outpaint import PanoOutpaintGenerator
    return PanoGenerator, PanoOutpaintGenerator


# 模型缓存：key=(project_root_resolved, "text2pano"|"outpaint") -> (config, model)
# 启动时预加载或首次任务时加载，后续任务直接复用，避免每次推理都加载权重
_loaded_models: dict = {}


def _load_text2pano(project_root: str):
    """加载文生图配置与模型并移至 GPU，返回 (config, model)。"""
    root = Path(project_root).resolve()
    _ensure_project_root_in_path(project_root)
    PanoGenerator, _ = _import_models()
    with open(root / "configs" / "pano_generation.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    logger.info("[进度] 加载权重 pano.ckpt（文生图）...")
    model = PanoGenerator(config)
    ckpt = torch.load(str(root / "weights" / "pano.ckpt"), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda()
    logger.info("[进度] 模型已加载并移至 GPU")
    return config, model


def _load_outpaint(project_root: str):
    """加载外扩配置与模型并移至 GPU，返回 (config, model)。"""
    root = Path(project_root).resolve()
    _ensure_project_root_in_path(project_root)
    _, PanoOutpaintGenerator = _import_models()
    with open(root / "configs" / "pano_generation_outpaint.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    logger.info("[进度] 加载权重 pano_outpaint.ckpt（外扩）...")
    model = PanoOutpaintGenerator(config)
    ckpt = torch.load(str(root / "weights" / "pano_outpaint.ckpt"), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda()
    logger.info("[进度] 模型已加载并移至 GPU")
    return config, model


def preload_models(project_root: str) -> None:
    """
    在项目启动时调用，预加载文生图与外扩模型到内存并移至 GPU。
    后续每次处理任务将直接复用，无需重复加载，可显著减少单次推理耗时。
    """
    root_str = str(Path(project_root).resolve())
    key_t = (root_str, "text2pano")
    key_o = (root_str, "outpaint")
    if key_t not in _loaded_models:
        logger.info("[进度] 预加载文生图模型...")
        _loaded_models[key_t] = _load_text2pano(project_root)
        logger.info("[进度] 文生图模型预加载完成")
    if key_o not in _loaded_models:
        logger.info("[进度] 预加载外扩模型...")
        _loaded_models[key_o] = _load_outpaint(project_root)
        logger.info("[进度] 外扩模型预加载完成")


def _get_K_R(FOV: float, THETA: float, PHI: float, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


def _resize_and_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    H, W, _ = img.shape
    if H == W:
        return cv2.resize(img, (size, size))
    if H > W:
        current_size = int(size * H / W)
        img = cv2.resize(img, (size, current_size))
        margin_l = (current_size - size) // 2
        margin_r = current_size - margin_l - size
        return img[margin_l:-margin_r, :]
    current_size = int(size * W / H)
    img = cv2.resize(img, (current_size, size))
    margin_l = (current_size - size) // 2
    margin_r = current_size - margin_l - size
    return img[:, margin_l:-margin_r]


def run_inference(
    project_root: str,
    text: str,
    image_path: Optional[str] = None,
    gen_video: bool = False,
    text_path: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    执行全景推理（与 demo 逻辑一致）。仅在 app 内使用，不依赖 demo.py。
    :return: (output_dir, image_paths)
    :raises: Exception on failure
    """
    mode = "outpaint" if (image_path and image_path.strip()) else "text2pano"
    logger.info("[进度] 开始全景推理 mode=%s", mode)

    _ensure_project_root_in_path(project_root)
    root = Path(project_root).resolve()
    root_str = str(root)
    cache_key = (root_str, mode)

    if cache_key not in _loaded_models:
        if mode == "text2pano":
            _loaded_models[cache_key] = _load_text2pano(project_root)
        else:
            _loaded_models[cache_key] = _load_outpaint(project_root)

    config, model = _loaded_models[cache_key]
    img = None

    if image_path and image_path.strip():
        logger.info("[进度] 使用已加载的外扩模型，读取参考图...")
        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = root / img_path
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = _resize_and_center_crop(img, config["dataset"]["resolution"])
        img = img / 127.5 - 1
        img = torch.tensor(img).float().cuda()
        try:
            from exiftool import ExifToolHelper
            with ExifToolHelper() as et:
                meta = et.get_metadata(str(img_path))
                if meta and "PNG:Parameters" in meta[0]:
                    lines = meta[0]["PNG:Parameters"].strip().split("\n")
                    if lines:
                        text = lines[0]
        except Exception:
            pass

    resolution = config["dataset"]["resolution"]
    Rs, Ks = [], []
    for i in range(8):
        degree = (45 * i) % 360
        K, R = _get_K_R(90, degree, 0, resolution, resolution)
        Rs.append(R)
        Ks.append(K)

    images = torch.zeros((1, 8, resolution, resolution, 3)).float().cuda()
    if img is not None:
        images[0, 0] = img

    if text_path:
        text_path_abs = root / text_path if not Path(text_path).is_absolute() else Path(text_path)
        with open(text_path_abs, "r", encoding="utf-8") as f:
            prompt = [line.strip() for line in f]
        if len(prompt) < 8:
            raise ValueError("text_path 需至少 8 行")
    else:
        prompt = [text] * 8

    K_t = torch.tensor(Ks).float().cuda()[None]
    R_t = torch.tensor(Rs).float().cuda()[None]
    batch = {"images": images, "prompt": prompt, "R": R_t, "K": K_t}
    logger.info("[进度] 开始模型推理（8 视角生成，耗时较长）...")
    images_pred = model.inference(batch)
    logger.info("[进度] 模型推理完成，保存 8 张视角图...")

    out_dir = root / "outputs" / ("results" + datetime.now().strftime("--%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(text, encoding="utf-8")
    image_paths = []
    for i in range(8):
        from PIL import Image
        arr = images_pred[0, i]
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
        p = out_dir / f"{i}.png"
        im.save(str(p))
        image_paths.append(str(p))
    logger.info("[进度] 8 张视角图已保存，生成全景图 pano.png ...")

    # 与 demo.py 一致：始终调用 generate_video 以生成 pano.png，gen_video 仅控制是否生成 video.mp4
    from generate_video_tool.pano_video_generation import generate_video
    generate_video(image_paths, str(out_dir), gen_video)
    image_paths.append(str(out_dir / "pano.png"))
    logger.info("[进度] 全景推理完成 output_dir=%s", out_dir)

    return str(out_dir), image_paths
