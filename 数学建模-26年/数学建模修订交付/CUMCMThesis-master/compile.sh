#!/bin/bash
# 将论文 TeX 源文件编译为 PDF。
# 注意：cumcmthesis 类强制要求 XeTeX，切勿用 pdflatex/latex 编译。
# 用法：
#   ./compile.sh          编译全部源文件（main.tex、AI工具使用详情.tex）
#   ./compile.sh clean    仅删除编译产生的辅助文件（保留 PDF）
set -euo pipefail
cd "$(dirname "$0")"

TEX_FILES=(main "AI工具使用详情")

if [[ "${1:-}" == "clean" ]]; then
    for f in "${TEX_FILES[@]}"; do
        rm -f "$f".aux "$f".log "$f".out "$f".xdv "$f".fls "$f".fdb_latexmk "$f".toc
        echo "已清理 ${f} 的辅助文件"
    done
    exit 0
fi

fail=0
for f in "${TEX_FILES[@]}"; do
    echo "==> 编译 ${f}.tex ..."
    if ! latexmk -xelatex -interaction=nonstopmode -halt-on-error -quiet "$f".tex; then
        echo "!! ${f}.tex 编译失败，详见 ${f}.log"
        fail=1
    else
        echo "    ${f}.pdf 已生成/更新"
    fi
done

if [[ $fail -eq 0 ]]; then
    echo "全部完成：$(ls main.pdf 'AI工具使用详情.pdf' | tr '\n' ' ')"
else
    echo "存在编译失败，请检查对应 .log 文件"
    exit 1
fi
