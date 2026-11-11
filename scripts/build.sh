#!/usr/bin/env bash
# 编译自研推理框架、单元测试与命令行工具。
#
# 可选环境变量：
#   BUILD_DIR        构建目录（默认 <repo>/build）
#   BUILD_TYPE       CMake 构建类型（默认 Release）
#   JOBS             并行编译任务数（默认 CPU 核数）
#   MNN_BUILD_DIR    含 libMNN.so 的目录（默认 /home/leixian/workspaces/projects/AIInfer/MNN/build）
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT_DIR/build"}
BUILD_TYPE=${BUILD_TYPE:-Release}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

for tool in cmake c++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[build] 缺少 $tool，请先安装构建工具" >&2
        exit 1
    fi
done

# 构建目录由其他编译器生成时 CMake 无法复用缓存，直接清理重建。
CACHE="$BUILD_DIR/CMakeCache.txt"
if [ -f "$CACHE" ]; then
    cached_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$CACHE" | head -n 1)
    current_cxx=$(command -v c++)
    if [ -n "$cached_cxx" ] && [ "$cached_cxx" != "$current_cxx" ]; then
        echo "[build] 编译器由 $cached_cxx 变为 $current_cxx，清理 $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
fi

CMAKE_ARGS=(-S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE")
if [ -n "${MNN_BUILD_DIR:-}" ]; then
    CMAKE_ARGS+=(-DMNN_BUILD_DIR="$MNN_BUILD_DIR")
fi

echo "[build] 配置 $BUILD_DIR ($BUILD_TYPE)"
cmake "${CMAKE_ARGS[@]}"

echo "[build] 编译 (-j $JOBS)"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "[build] 完成，产物位于 $BUILD_DIR"
