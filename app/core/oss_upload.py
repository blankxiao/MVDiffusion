"""
将生成的全景图 pano.png 上传到阿里云 OSS。
存储路径：{user_id}/{timestamp}.png，便于按用户与时间区分。
"""
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def upload_pano_to_oss(
    local_path: str,
    user_id: str,
    *,
    endpoint: str,
    access_key_id: str,
    access_key_secret: str,
    bucket_name: str,
    bucket_domain: Optional[str] = None,
) -> Optional[str]:
    """
    上传本地 pano.png 到 OSS。
    对象键：{user_id}/{YYYYMMdd_HHmmss}.png

    :param local_path: 本地 pano.png 完整路径
    :param user_id: 用户/任务标识，作为 OSS 前缀目录
    :return: 可访问的图片 URL；失败或未配置时返回 None
    """
    import os
    if not os.path.isfile(local_path):
        logger.warning("OSS 上传跳过：文件不存在 %s", local_path)
        return None
    if not endpoint or not access_key_id or not access_key_secret or not bucket_name:
        logger.debug("OSS 未配置，跳过上传")
        return None

    try:
        import oss2
    except ImportError:
        logger.warning("未安装 oss2，跳过 OSS 上传。请 pip install oss2")
        return None

    # 文件名：时间戳，避免重复
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 规范 user_id 为安全路径（去掉路径分隔符等）
    safe_prefix = (user_id or "default").strip().replace("\\", "/").strip("/")
    if not safe_prefix:
        safe_prefix = "default"
    object_key = f"{safe_prefix}/{ts}.png"

    try:
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        bucket.put_object_from_file(object_key, local_path)
        logger.info("OSS 上传成功: %s -> %s", local_path, object_key)

        if bucket_domain:
            domain = bucket_domain.rstrip("/")
            return f"{domain}/{object_key}"
        return f"https://{bucket_name}.{endpoint}/{object_key}"
    except Exception as e:
        logger.exception("OSS 上传失败: %s", e)
        return None
