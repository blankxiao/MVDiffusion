#!/bin/bash
# 通过 HTTP 访问 MVDiffusion 生成的图片和资源
# 用法: ./serve_images.sh [端口]

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PORT="${1:-8080}"

# 获取本机 IP（优先非 127 的地址）
get_url() {
    local ip
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -z "$ip" ]; then
        ip=$(ip route get 1 2>/dev/null | awk '{print $7; exit}')
    fi
    [ -z "$ip" ] && ip="127.0.0.1"
    echo "http://${ip}:${PORT}"
}

echo "================================================"
echo "  图片 HTTP 服务"
echo "================================================"
echo "  根目录: $ROOT"
echo "  端口:   $PORT"
echo "================================================"
echo ""
echo "  访问地址: $(get_url)"
echo ""
echo "  常用路径:"
echo "    - 生成结果: $(get_url)/outputs/"
echo "    - 资源文件: $(get_url)/assets/"
echo ""
echo "  按 Ctrl+C 停止服务"
echo "================================================"

exec python3 -m http.server "$PORT" --directory "$ROOT"
