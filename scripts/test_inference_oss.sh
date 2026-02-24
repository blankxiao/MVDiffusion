#!/usr/bin/env bash
# 测试「AI 推理 + 结果上传 OSS」的 HTTP 接口
# 使用前：确保服务已启动（./run_app.sh 或 uvicorn server:app --host 0.0.0.0 --port 9000）
# 若使用 app 服务：POST /api/test/inference；若使用根目录 server：POST /api/inference

set -e
BASE_URL="${BASE_URL:-http://localhost:9000}"
USER_ID="${USER_ID:-test_user_001}"

# 示例：全景图文案（可改成你自己的）
TEXT="${1:-A cozy living room with a large sofa, wooden floor, plants by the window and soft daylight.}"

echo "=== 请求 ==="
echo "URL: ${BASE_URL}"
echo "user_id: ${USER_ID}"
echo "text: ${TEXT:0:80}..."
echo ""

# 方式一：app 服务（进程内推理 + OSS）
# 接口：POST /api/test/inference
echo ">>> 调用 POST /api/test/inference (app 服务)"
RESP=$(curl -s -X POST "${BASE_URL}/api/test/inference" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"${TEXT}\", \"user_id\": \"${USER_ID}\"}")
echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"

# 若使用根目录 server.py，可改为：
# curl -s -X POST "${BASE_URL}/api/inference" \
#   -H "Content-Type: application/json" \
#   -d "{\"text\": \"${TEXT}\", \"user_id\": \"${USER_ID}\"}"
# 返回字段相同：success, output_dir, image_paths, pano_oss_url, message

echo ""
echo "=== 检查 pano_oss_url ==="
echo "$RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    url = d.get('pano_oss_url')
    if url:
        print('pano_oss_url:', url)
    else:
        print('pano_oss_url: (未返回，请检查 .env 中 OSS 配置)')
    print('success:', d.get('success'))
except Exception as e:
    print(e)
"
