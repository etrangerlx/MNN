#!/usr/bin/env bash
# 先编译，再运行单元测试和 benchmark 模型与 MNN 的精度对比。
#
# 可选环境变量：
#   BUILD_DIR        构建目录（默认 <repo>/build）
#   SKIP_BUILD=1     跳过编译，直接使用已有产物
#   THREADS          推理线程数（默认 8）
#   RUNS             性能统计的重复次数（默认 3）
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT_DIR/build"}
THREADS=${THREADS:-8}
RUNS=${RUNS:-3}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    BUILD_DIR="$BUILD_DIR" "$ROOT_DIR/scripts/build.sh" || exit 1
fi

failed=0

echo
echo "===== 单元测试 ====="
if "$BUILD_DIR/infer_test"; then
    echo "[test] 单元测试通过"
else
    echo "[test] 单元测试失败" >&2
    failed=$((failed + 1))
fi

if [ ! -x "$BUILD_DIR/infer_compare" ]; then
    echo "[test] 未找到 infer_compare（需要可用的 libMNN.so），跳过模型对比" >&2
    exit $((failed > 0))
fi

echo
echo "===== benchmark 模型精度对比（infer vs MNN） ====="
total=0
passed=0
skipped=0
skipped_names=()
failed_names=()
tmp_log=$(mktemp)

shopt -s nullglob
for model in "$ROOT_DIR"/models/*.mnn; do
    total=$((total + 1))
    name=$(basename "$model")
    echo
    echo "----- $name -----"
    if "$BUILD_DIR/infer_compare" "$model" --threads "$THREADS" --runs "$RUNS" >"$tmp_log" 2>&1; then
        cat "$tmp_log"
        passed=$((passed + 1))
    elif grep -q "without weights" "$tmp_log"; then
        cat "$tmp_log"
        echo "[skip] $name：模型不含权重，仅可用于性能测试"
        skipped=$((skipped + 1))
        skipped_names+=("$name")
    else
        cat "$tmp_log"
        failed=$((failed + 1))
        failed_names+=("$name")
    fi
done
shopt -u nullglob
rm -f "$tmp_log"

echo
echo "===== 汇总 ====="
echo "通过 $passed / 无权重跳过 $skipped / 失败 $failed（共 $total 个模型）"
if [ "${#skipped_names[@]}" -gt 0 ]; then
    echo "跳过：${skipped_names[*]}"
fi
if [ "${#failed_names[@]}" -gt 0 ]; then
    echo "失败：${failed_names[*]}" >&2
fi
if [ "$failed" -ne 0 ]; then
    exit 1
fi
