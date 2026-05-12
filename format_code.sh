#!/bin/bash

# ============================================================
# 格式化脚本：使用项目根目录下的 .clang-format 对 C/C++ 源码进行格式化
# 自动探测项目根目录，支持文件排除，配置集中管理
# ============================================================

set -euo pipefail

# ============================================================
# 用户可配置区域（按需修改）
# ============================================================

# --- 格式化使用的 .clang-format 配置文件名称 ---
CLANG_FORMAT_NAME=".clang-format"

# --- 默认格式化目录（相对于项目根目录） ---
DEFAULT_DIRS=("firmware" "hello" "mnist-fc" "runtime")

# --- 需要格式化的文件扩展名 ---
EXTENSIONS=("c" "h" "cpp" "hpp" "cc" "cxx")

# --- 排除模式（使用 find 的 -name 语法） ---
# 例如 "*.bin.h" 会排除所有 bin.h 结尾的文件
EXCLUDE_PATTERNS=("*.bin.h" "*.generated.cpp")

# --- 可通过环境变量 FORMAT_EXCLUDE 追加排除模式 ---
# 例如：FORMAT_EXCLUDE='*.bak' ./format.sh

# ============================================================
# 自动探测项目根目录（如不希望自动探测，可注释掉并直接赋值 PROJECT_ROOT）
# ============================================================
find_project_root() {
    local dir
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/${CLANG_FORMAT_NAME}" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    echo "Error: Could not find ${CLANG_FORMAT_NAME} in any parent directory." >&2
    return 1
}

PROJECT_ROOT=$(find_project_root)
if [[ -z "$PROJECT_ROOT" ]]; then
    exit 1
fi

CLANG_FORMAT_FILE="${PROJECT_ROOT}/${CLANG_FORMAT_NAME}"

# ============================================================
# 前置检查
# ============================================================
if [[ ! -f "${CLANG_FORMAT_FILE}" ]]; then
    echo "Error: ${CLANG_FORMAT_NAME} not found at ${CLANG_FORMAT_FILE}" >&2
    exit 1
fi

if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format is not installed. Please install it first." >&2
    exit 1
fi

# ============================================================
# 处理外部追加的排除模式
# ============================================================
if [[ -n "${FORMAT_EXCLUDE:-}" ]]; then
    read -ra EXTRA_EXCLUDE <<< "$FORMAT_EXCLUDE"
    EXCLUDE_PATTERNS+=("${EXTRA_EXCLUDE[@]}")
fi

# ============================================================
# 函数：显示用法
# ============================================================
usage() {
    echo "Usage: $0 [--help] [directory1] [directory2] ..."
    echo ""
    echo "Format C/C++ source files using the project's ${CLANG_FORMAT_NAME}."
    echo "If no directories are specified, the following default directories will be formatted:"
    printf "  %s\n" "${DEFAULT_DIRS[@]}"
    echo ""
    echo "Currently excluded patterns: ${EXCLUDE_PATTERNS[*]}"
    echo "You can add extra patterns via environment variable FORMAT_EXCLUDE."
    echo ""
    echo "Examples:"
    echo "  $0                    # Format default directories"
    echo "  $0 firmware runtime   # Format only firmware and runtime"
    echo "  $0 .                  # Format the entire project"
    echo "  FORMAT_EXCLUDE='*.bak' $0  # Also exclude .bak files"
    exit 1
}

# ============================================================
# 命令行参数解析
# ============================================================
if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
fi

if [[ $# -eq 0 ]]; then
    TARGET_DIRS=("${DEFAULT_DIRS[@]}")
else
    TARGET_DIRS=("$@")
fi

# ============================================================
# 构建 find 命令的条件参数
# ============================================================

# 构建文件扩展名条件（带 -o）
NAME_ARGS=()
for i in "${!EXTENSIONS[@]}"; do
    NAME_ARGS+=(-name "*.${EXTENSIONS[$i]}")
    if [[ $i -lt $((${#EXTENSIONS[@]} - 1)) ]]; then
        NAME_ARGS+=(-o)
    fi
done

# 构建排除条件
EXCLUDE_ARGS=()
if [[ ${#EXCLUDE_PATTERNS[@]} -gt 0 ]]; then
    EXCLUDE_ARGS+=(-not '(')
    for i in "${!EXCLUDE_PATTERNS[@]}"; do
        EXCLUDE_ARGS+=(-name "${EXCLUDE_PATTERNS[$i]}")
        if [[ $i -lt $((${#EXCLUDE_PATTERNS[@]} - 1)) ]]; then
            EXCLUDE_ARGS+=(-o)
        fi
    done
    EXCLUDE_ARGS+=(')')
fi

# ============================================================
# 执行格式化
# ============================================================
FORMATTED_COUNT=0

for dir in "${TARGET_DIRS[@]}"; do
    # 解析目录路径
    if [[ "$dir" == /* ]]; then
        TARGET_PATH="$dir"
    else
        TARGET_PATH="${PROJECT_ROOT}/${dir}"
    fi

    if [[ ! -d "$TARGET_PATH" ]]; then
        echo "Warning: Directory does not exist, skipping: $TARGET_PATH" >&2
        continue
    fi

    echo "Formatting directory: $TARGET_PATH"

    while IFS= read -r -d '' file; do
        echo "  Formatting: $file"
        clang-format -i --style="file:${CLANG_FORMAT_FILE}" "$file" || true
        ((FORMATTED_COUNT++)) || true
    done < <(find "$TARGET_PATH" -type f \( "${NAME_ARGS[@]}" \) "${EXCLUDE_ARGS[@]}" -print0)
done

echo ""
echo "Formatting complete! Total files formatted: $FORMATTED_COUNT"