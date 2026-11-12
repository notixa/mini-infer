#!/usr/bin/env bash
set -e

# 切换到脚本所在目录（即仓库根目录）
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# 检查是否传入了 commit message
if [ $# -eq 0 ]; then
    echo "用法: $0 <commit message>"
    exit 1
fi

COMMIT_MSG="$*"

echo "==> 暂存变更: git add ."
git add .

echo "==> 提交变更: git commit -m \"$COMMIT_MSG\""
git commit -m "$COMMIT_MSG"

echo "==> 推送到远程: git push -u origin main"
git push -u origin main

echo "==> 完成！"
