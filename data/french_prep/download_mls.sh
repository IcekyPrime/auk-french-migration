#!/usr/bin/env bash
set -e
# ============================================================
# 下载 MLS 法语（OpenSLR，非 HuggingFace / 非 GitHub / 非 SSH）
# 解压到 data/french_prep/mls_french，目录结构匹配 preprocess_mls.py 期望的
#   mls_french/{split}/transcripts.txt
#   mls_french/{split}/audio/<spk>/<book>/<spk>_<book>_<utt>.flac
# ============================================================
# 用法（在 AuK 项目根）:
#   bash data/french_prep/download_mls.sh
# 可用环境变量覆盖下载地址：
#   MLS_URL=... bash data/french_prep/download_mls.sh
# ============================================================

cd "$(dirname "$0")/../.."      # 切到 AuK 项目根，保证相对路径一致

# OpenSLR 94（MLS）法语 flac 版。文件名/大小请以 https://www.openslr.org/94/ 页面为准，
# 若页面提供 opus 版（体积小很多）但脚本需 flac，可自行替换 URL 并在 preprocess_mls.py 里
# 兼容 opus 解码。默认走 flac（约数十 GB，wget -c 支持断点续传）。
MLS_URL="${MLS_URL:-https://www.openslr.org/resources/94/mls_french.tar.gz}"
TARBALL=data/french_prep/mls_french.tar.gz

mkdir -p data/french_prep

if [ -d data/french_prep/mls_french ]; then
    echo "[mls] 已存在 data/french_prep/mls_french，跳过下载"
    exit 0
fi

echo "[mls] 下载: ${MLS_URL}"
wget -c "$MLS_URL" -O "$TARBALL"

echo "[mls] 解压到 data/french_prep/"
tar -xzf "$TARBALL" -C data/french_prep

echo "[mls] 删除 tarball（已解压）"
rm -f "$TARBALL"

echo "[mls] done -> data/french_prep/mls_french"
