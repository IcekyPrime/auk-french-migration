#!/usr/bin/env bash
set -e

# ============================================================
# 法语 LoRA 合并 + 批量合成验证
# ============================================================
# 用法（在 AuK 项目根执行）:
#     bash data/french_prep/french_merge.sh
# 输出目录: ckpts/auk_lora_french/samples_fr/
# ============================================================

cd "$(dirname "$0")/../.."      # 切到 AuK 项目根
export PYTHONPATH=src

CKPT_DIR=ckpts/auk_lora_french
MERGED_CKPT=${CKPT_DIR}/auk_lora_french_merged.safetensors
OUT_DIR=${CKPT_DIR}/samples_fr
GEN_SECONDS=5                    # 目标合成时长（秒），按需改

# ---------- 1) 合并 LoRA（已合并则跳过） ----------
if [ ! -f "${MERGED_CKPT}" ]; then
    echo "[merge] 合并 LoRA -> ${MERGED_CKPT}"
    python data/merge_lora.py \
        --ckpt ${CKPT_DIR}/model_last.pt \
        --out ${MERGED_CKPT}
fi
# config.yaml 需与 merged ckpt 同目录（infer_cli 要求）
[ -f "${CKPT_DIR}/config.yaml" ] || cp ckpts/AuK/config.yaml ${CKPT_DIR}/config.yaml
mkdir -p ${OUT_DIR}

# ---------- 2) 参考音频（建议换成不同说话人，测音色多样性） ----------
# 查看有哪些说话人:
#     ls data/french_prep/french_data/ | sed 's/_.*//' | sort -u
REF_AUDIOS=(
    data/french_prep/french_data/101580_000006.wav
    data/french_prep/french_data/111068_000003.wav
    data/french_prep/french_data/1129_000077.wav
)

# ---------- 3) 目标法语文本（含撇号时用双引号包裹） ----------
TEXTS=(
    "Bonjour, comment allez-vous ?"
    "Le petit déjeuner est délicieux ce matin."
    "Je voudrais réserver une table pour deux personnes, s'il vous plaît."
    "La réunion commencera à neuf heures précises."
)

# ---------- 4) 批量合成 ----------
n=0
for ref in "${REF_AUDIOS[@]}"; do
    ref_stem=$(basename "${ref}" .wav)
    for i in "${!TEXTS[@]}"; do
        text="${TEXTS[$i]}"
        # 拼接法语指令模板（与训练时逐字一致）
        instruction=$(printf 'Dis le texte suivant dans la même voix : "%s"' "${text}")
        out=${OUT_DIR}/ref_${ref_stem}_txt$((i+1)).wav
        echo "[synth ${n}] ${ref_stem} + txt$((i+1)) -> ${out}"
        python -m auk.infer.infer_cli \
            --ckpt ${MERGED_CKPT} \
            --audio "${ref}" \
            --instruction "${instruction}" \
            --gen_seconds ${GEN_SECONDS} \
            --cpu_offload \
            --output "${out}"
        n=$((n+1))
    done
done

echo ""
echo "[done] 共合成 ${n} 条，输出目录: ${OUT_DIR}"
