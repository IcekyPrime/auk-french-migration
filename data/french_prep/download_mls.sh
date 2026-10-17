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

# MLS 法语数据不在 OpenSLR 上（体积太大），实际托管在 Meta 的 dl.fbaipublicfiles.com（AWS S3）。
# 两个版本：flac 61G / opus 16G。默认走 opus 省空间；如需 flac，设 MLS_URL 覆盖。
MLS_URL="${MLS_URL:-https://dl.fbaipublicfiles.com/mls/mls_french_opus.tar.gz}"
TARBALL=data/french_prep/mls_french.tar.gz

mkdir -p data/french_prep

# 目标目录：若是 symlink（重定向到个人目录），解析到真实路径；否则用项目内目录
MLS_DIR=data/french_prep/mls_french
if [ -L "$MLS_DIR" ]; then
    MLS_DIR=$(readlink -f "$MLS_DIR")
fi
mkdir -p "$MLS_DIR"

# 有真实数据才跳过（不能只看目录是否存在，空 symlink 会误判）
if [ -f "$MLS_DIR/train/transcripts.txt" ]; then
    echo "[mls] 已存在 $MLS_DIR/train/transcripts.txt，跳过下载"
    exit 0
fi

echo "[mls] 下载: ${MLS_URL}"
wget -c "$MLS_URL" -O "$TARBALL"

echo "[mls] 解压到 $MLS_DIR（跳过顶层 mls_french 目录）"
tar -xzf "$TARBALL" --strip-components=1 -C "$MLS_DIR"

echo "[mls] 删除 tarball（已解压）"
rm -f "$TARBALL"

echo "[mls] done -> $MLS_DIR"
