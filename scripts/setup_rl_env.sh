#!/usr/bin/env bash
# ============================================================
# FastWAM Flow-GSPO RL 环境一键部署脚本
# ============================================================
# 用法:
#   chmod +x scripts/setup_rl_env.sh
#   nohup bash scripts/setup_rl_env.sh > setup_rl.log 2>&1 &
#   tail -f setup_rl.log
# ============================================================

set -euo pipefail

# -------------------- 配置区 --------------------
ENV_NAME="fastwam"
CONDA_BASE="/mnt/public/apps/miniconda3"
PROJECT_DIR="/mnt/users/zhouqm-20251002/FastWAM"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
PYTHON_VERSION="3.10"

# 清华源
TSINGHUA_PIP="https://pypi.tuna.tsinghua.edu.cn/simple"
TSINGHUA_CONDA="https://mirrors.tuna.tsinghua.edu.cn/anaconda"

# HuggingFace 镜像
HF_MIRROR="https://hf-mirror.com"

# 数据目录
DATA_DIR="${PROJECT_DIR}/data"
CKPT_DIR="${PROJECT_DIR}/checkpoints"

# LIBERO 数据集下载链接
LIBERO_DATASET_URL="https://hf-mirror.com/datasets/yuanty/LIBERO-fastwam"

# 日志
LOG_FILE="${PROJECT_DIR}/setup_rl.log"
# -------------------- 配置区结束 --------------------

exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_err() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }

# ====================== Step 0: 前置检查 ======================
log "===== Step 0: 前置检查 ====="

if [ ! -d "${PROJECT_DIR}" ]; then
    log_err "项目目录不存在: ${PROJECT_DIR}"
    exit 1
fi

cd "${PROJECT_DIR}"

if [ -d "${ENV_DIR}" ]; then
    log "conda 环境 '${ENV_NAME}' 已存在, 跳过创建"
else
    log "===== Step 1: 创建 conda 环境 ====="

    # 配置清华 conda 镜像
    cat > /tmp/fastwam_condarc <<EOF
channels:
  - ${TSINGHUA_CONDA}/pkgs/main
  - ${TSINGHUA_CONDA}/pkgs/r
  - ${TSINGHUA_CONDA}/pkgs/msys2
  - defaults
show_channel_urls: true
default_threads: 8
EOF

    CONDA_RC_FLAG=""
    if [ ! -f "${CONDA_BASE}/.condarc" ]; then
        cp /tmp/fastwam_condarc "${CONDA_BASE}/.condarc"
    fi

    source "${CONDA_BASE}/etc/profile.d/conda.sh"

    log "创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y -q

    log "conda 环境创建完成: ${ENV_DIR}"
fi

# ====================== Step 2: 安装 PyTorch + 项目依赖 ======================
log "===== Step 2: 安装依赖 ====="

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

log "pip 配置清华源..."
pip config set global.index-url "${TSINGHUA_PIP}"
pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn"
pip install -U pip -q

log "安装 PyTorch (cu128)..."
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

log "安装项目依赖 (pip install -e .)..."
pip install -e "." -q

log "验证 torch..."
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')"

# ====================== Step 3: 安装 LIBERO 环境 ======================
log "===== Step 3: 安装 LIBERO 仿真环境 ====="

# egl_probe/hf-egl-probe 编译与新版 cmake 不兼容, 且 RL 训练不需要 (仅用于 EGL 渲染探测)
# 策略: libero/robomimic 用 --no-deps 安装, 手动装真正需要的依赖, 跳过 egl_probe

log "安装 mujoco==3.3.2..."
pip install mujoco==3.3.2

log "安装 libero (跳过 egl_probe)..."
pip install libero --no-deps

log "安装 robosuite + robomimic (跳过 egl_probe)..."
pip install robosuite==1.4.0 robomimic==0.2.0 --no-deps

log "安装 LIBERO 依赖..."
pip install bddl easydict opencv-python gymnasium h5py tensorboard \
    tensorboardX matplotlib cloudpickle thop future numba scipy

# 预创建 LIBERO 配置文件, 避免首次 import 时交互式询问路径导致 nohup 卡死
LIBERO_PKG_DIR=$(python -c "
import importlib.util, os
spec = importlib.util.find_spec('libero.libero')
if spec and spec.origin:
    print(os.path.dirname(spec.origin))
" 2>/dev/null || echo "")

if [ -z "${LIBERO_PKG_DIR}" ]; then
    # fallback: 从 pip show 推断
    LIBERO_PKG_DIR=$(python -c "
import subprocess, os, re
out = subprocess.check_output(['pip', 'show', 'libero']).decode()
loc = re.search(r'Location: (.+)', out).group(1).strip()
print(os.path.join(loc, 'libero', 'libero'))
")
fi

LIBERO_CONFIG_DIR="${HOME}/.libero"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_DIR}/config.yaml"

if [ ! -f "${LIBERO_CONFIG_FILE}" ]; then
    log "创建 LIBERO 配置: ${LIBERO_CONFIG_FILE}"
    mkdir -p "${LIBERO_CONFIG_DIR}"
    cat > "${LIBERO_CONFIG_FILE}" << YAMLHERE
benchmark_root: ${LIBERO_PKG_DIR}
bddl_files: ${LIBERO_PKG_DIR}/bddl_files
init_states: ${LIBERO_PKG_DIR}/init_files
datasets: ${DATA_DIR}/libero_datasets
assets: ${LIBERO_PKG_DIR}/assets
YAMLHERE
else
    log "LIBERO 配置已存在: ${LIBERO_CONFIG_FILE}"
fi

# 验证
python -c "from libero.libero import benchmark; print('LIBERO imported OK')"
python -c "import robosuite; print('robosuite imported OK')"

# ====================== Step 3b: 下载 LIBERO Assets ======================
log "===== Step 3b: 下载 LIBERO Assets (仿真物体/场景) ====="

# LIBERO assets: 586 个 3D 模型/场景文件, 约 500MB
# 来源: jadechoghari/libero-assets (HuggingFace)
LIBERO_ASSETS_DIR="${LIBERO_PKG_DIR}/assets"

if [ -d "${LIBERO_ASSETS_DIR}" ] && [ "$(find "${LIBERO_ASSETS_DIR}" -type f -not -path '*/.cache/*' | wc -l)" -ge 500 ]; then
    log "LIBERO assets 已存在 ($(find "${LIBERO_ASSETS_DIR}" -type f -not -path '*/.cache/*' | wc -l) files), 跳过下载"
else
    log "从 HuggingFace 镜像下载 LIBERO assets..."
    mkdir -p "${LIBERO_ASSETS_DIR}"
    export HF_ENDPOINT="${HF_MIRROR}"
    export HF_HUB_ENABLE_HF_TRANSFER=0

    # 镜像可能限流, 逐批重试
    MAX_ASSET_RETRIES=30
    for i in $(seq 1 ${MAX_ASSET_RETRIES}); do
        log "[Assets] 第 ${i}/${MAX_ASSET_RETRIES} 次尝试..."
        if huggingface-cli download jadechoghari/libero-assets --repo-type model --local-dir "${LIBERO_ASSETS_DIR}"; then
            log "[Assets] 下载完成"
            break
        fi
        log "[Assets] 连接中断, 30秒后重试..."
        sleep 30
    done

    # 清理下载缓存
    rm -rf "${LIBERO_ASSETS_DIR}/.cache"

    ASSET_COUNT=$(find "${LIBERO_ASSETS_DIR}" -type f | wc -l)
    log "LIBERO assets: ${ASSET_COUNT} files"
fi

# 配置 robosuite macros (消除警告)
ROBOSUITE_DIR=$(python -c "import robosuite; import os; print(os.path.dirname(robosuite.__file__))")
if [ ! -f "${ROBOSUITE_DIR}/macros_private.py" ]; then
    log "配置 robosuite macros..."
    python "${ROBOSUITE_DIR}/scripts/setup_macros.py" 2>/dev/null || true
fi

# ====================== Step 4: 下载模型权重 ======================
log "===== Step 4: 下载模型权重 ====="

log "安装 modelscope..."
pip install modelscope -q

mkdir -p "${CKPT_DIR}"

# HF 镜像 + 普通下载 (hf_transfer 在此机器不可用)
export HF_ENDPOINT="${HF_MIRROR}"
export HF_HUB_ENABLE_HF_TRANSFER=0

# 带重试的下载函数 (镜像连接不稳定, 支持断点续传)
MAX_RETRIES=20

hf_download_retry() {
    local desc="$1"
    shift
    local attempt=1
    while [ ${attempt} -le ${MAX_RETRIES} ]; do
        log "[${desc}] 第 ${attempt}/${MAX_RETRIES} 次尝试..."
        if huggingface-cli download "$@"; then
            log "[${desc}] 下载完成"
            return 0
        fi
        log "[${desc}] 连接中断, 10秒后断点续传..."
        sleep 10
        attempt=$((attempt + 1))
    done
    log "[${desc}] ${MAX_RETRIES} 次重试后仍失败"
    return 1
}

# 4a: 下载 FastWAM 预训练权重 + dataset stats (约 12GB)
FASTWAM_RELEASE_DIR="${CKPT_DIR}/fastwam_release"

if [ -f "${FASTWAM_RELEASE_DIR}/libero_uncond_2cam224.pt" ]; then
    log "FastWAM checkpoint 已存在, 跳过下载"
else
    hf_download_retry "FastWAM" yuanty/fastwam \
        libero_uncond_2cam224.pt \
        libero_uncond_2cam224_dataset_stats.json \
        --local-dir "${FASTWAM_RELEASE_DIR}"
fi

log "FastWAM checkpoint:"
ls -lh "${FASTWAM_RELEASE_DIR}/"

# 4b: Wan2.2-TI2V-5B 基座模型 (ModelScope 国内加速, 约 10GB)
WAN_DIR="${CKPT_DIR}/Wan-AI/Wan2.2-TI2V-5B"

if [ -d "${WAN_DIR}" ] && [ "$(ls -A "${WAN_DIR}" 2>/dev/null)" ]; then
    log "Wan2.2 基座模型已存在, 跳过下载"
else
    log "从 ModelScope 下载 Wan2.2-TI2V-5B 基座模型 (国内加速, 约 10GB)..."
    mkdir -p "${WAN_DIR}"
    modelscope download --model Wan-AI/Wan2.2-TI2V-5B \
        --local_dir "${WAN_DIR}" \
        || log "[WARN] Wan2.2 下载不完整, 可稍后重试或运行时自动下载"
fi

export DIFFSYNTH_MODEL_BASE_PATH="${CKPT_DIR}"
log "Step 4 完成"

# ====================== Step 5: 生成 ActionDiT Backbone ======================
log "===== Step 5: 生成 ActionDiT Backbone ====="

ACTION_DIT_CKPT="${CKPT_DIR}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

if [ -f "${ACTION_DIT_CKPT}" ]; then
    log "ActionDiT backbone 已存在, 跳过生成"
else
    log "从 Wan22 DiT 生成 ActionDiT backbone (需要 GPU)..."
    export DIFFSYNTH_MODEL_BASE_PATH="${CKPT_DIR}"
    python scripts/preprocess_action_dit_backbone.py \
        --model-config configs/model/fastwam.yaml \
        --output "${ACTION_DIT_CKPT}" \
        --device cuda \
        --dtype bfloat16

    log "ActionDiT backbone 生成完成"
    ls -lh "${ACTION_DIT_CKPT}"
fi

# ====================== Step 6: 下载 LIBERO 数据集 ======================
log "===== Step 6: 下载 LIBERO 数据集 ====="

LIBERO_DATA_DIR="${DATA_DIR}/libero_mujoco3.3.2"

if [ -d "${LIBERO_DATA_DIR}/libero_spatial_no_noops_lerobot" ]; then
    log "LIBERO 数据集已存在, 跳过下载"
else
    log "从 HuggingFace 下载 LIBERO 数据集..."
    mkdir -p "${LIBERO_DATA_DIR}"

    # 使用 huggingface-cli 下载整个 dataset
    huggingface-cli download yuanty/LIBERO-fastwam \
        --repo-type dataset \
        --local-dir "${LIBERO_DATA_DIR}_download"

    log "解压数据集..."
    cd "${LIBERO_DATA_DIR}_download"
    for f in *.tar.gz; do
        if [ -f "$f" ]; then
            log "  解压: $f"
            tar -xzf "$f" -C "${LIBERO_DATA_DIR}"
        fi
    done
    cd "${PROJECT_DIR}"

    # 验证目录结构
    log "验证数据目录..."
    for suite in libero_spatial_no_noops_lerobot libero_object_no_noops_lerobot \
                 libero_goal_no_noops_lerobot libero_10_no_noops_lerobot; do
        if [ -d "${LIBERO_DATA_DIR}/${suite}" ]; then
            log "  [OK] ${suite}"
        else
            log_err "  [MISSING] ${suite}"
        fi
    done
fi

# ====================== Step 7: 预计算 T5 Text Embedding ======================
log "===== Step 7: 预计算 T5 Text Embedding ====="

TEXT_CACHE_DIR="${DATA_DIR}/text_embeds_cache/libero"

if [ -d "${TEXT_CACHE_DIR}" ] && [ "$(ls -A "${TEXT_CACHE_DIR}" 2>/dev/null)" ]; then
    log "Text embedding 缓存已存在, 跳过"
else
    log "预计算 T5 text embedding..."
    export DIFFSYNTH_MODEL_BASE_PATH="${CKPT_DIR}"
    python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
    log "Text embedding 预计算完成"
fi

# ====================== 验证 ======================
log ""
log "=========================================="
log "  安装完成! 验证摘要:"
log "=========================================="

log ""
log "[环境]"
log "  conda env: ${ENV_DIR}"
log "  $(python --version 2>&1)"
log "  $(python -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}")')"

log ""
log "[权重]"
for f in "${CKPT_DIR}/fastwam_release/libero_uncond_2cam224.pt" \
         "${CKPT_DIR}/fastwam_release/libero_uncond_2cam224_dataset_stats.json" \
         "${ACTION_DIT_CKPT}"; do
    if [ -f "$f" ]; then
        log "  [OK] $(basename $f) ($(du -sh "$f" | cut -f1))"
    else
        log "  [MISSING] $f"
    fi
done

log ""
log "[数据集]"
for suite in libero_spatial_no_noops_lerobot libero_object_no_noops_lerobot \
             libero_goal_no_noops_lerobot libero_10_no_noops_lerobot; do
    if [ -d "${LIBERO_DATA_DIR}/${suite}" ]; then
        log "  [OK] ${suite}"
    else
        log "  [MISSING] ${suite}"
    fi
done

log ""
log "[LIBERO]"
python -c "from libero.libero import benchmark; b = benchmark.get_benchmark_dict(); print(f'  [OK] 可用 suites: {list(b.keys())}')" 2>&1 || log "  [FAIL] LIBERO import 失败"
ASSET_COUNT=$(find "${LIBERO_ASSETS_DIR}" -type f -not -path '*/.cache/*' 2>/dev/null | wc -l)
log "  Assets: ${ASSET_COUNT} files in ${LIBERO_ASSETS_DIR}"

log ""
log "=========================================="
log "  启动 RL 训练示例:"
log "=========================================="
log ""
log "  source ${CONDA_BASE}/etc/profile.d/conda.sh"
log "  conda activate ${ENV_NAME}"
log "  cd ${PROJECT_DIR}"
log "  export DIFFSYNTH_MODEL_BASE_PATH=\"${CKPT_DIR}\""
log ""
log "  # Exp-2a: traj_chunk + match"
log "  nohup python scripts/train_rl.py \\"
log "      EVALUATION.task_suite_name=libero_spatial \\"
log "      rl.variant=traj_chunk rl.action_horizon=10 rl.exec_horizon=null \\"
log "      ckpt=${CKPT_DIR}/fastwam_release/libero_uncond_2cam224.pt \\"
log "      EVALUATION.dataset_stats_path=${CKPT_DIR}/fastwam_release/libero_uncond_2cam224_dataset_stats.json \\"
log "      EVALUATION.rand_device=cuda \\"
log "      wandb.enabled=false \\"
log "      > logs/exp2a.log 2>&1 &"
log ""
log "=========================================="
log "全部完成!"
