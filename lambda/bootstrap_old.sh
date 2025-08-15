#!/usr/bin/env bash
set -euo pipefail

# --------- User parameters (edit as needed) -----------------------------------
# Name of the *local* filesystem you attached in the Lambda console for this session
LOCAL_FS_NAME="${LOCAL_FS_NAME:-local_fs}"

# UUID of your S3-compatible Lambda filesystem (the bucket-like ID in us-east-3)
S3_UUID="${S3_UUID:-CHANGE-ME-UUID}"

# Source subdir inside that S3 filesystem to copy (example: a LoRA checkpoint tree)
S3_SRC_SUBDIR="${S3_SRC_SUBDIR:-checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156}"

# Git repo + branch for your training code
REPO_URL="${REPO_URL:-https://github.com/PhilSaad333/RL_Practice.git}"
REPO_DIR="${REPO_DIR:-$HOME/RL_Practice}"
REPO_BRANCH="${REPO_BRANCH:-main}"

# TensorBoard port (remote) and tmux session names
TB_PORT="${TB_PORT:-16006}"
TMUX_TB="${TMUX_TB:-tb}"
TMUX_TRAIN="${TMUX_TRAIN:-train}"

# (Optional) Training config and checkpoint directory (destination after copy)
TRAIN_CFG="${TRAIN_CFG:-rl_training/cfg/testconfig.yaml}"
# -----------------------------------------------------------------------------

# Internal paths (derived)
LOCAL_ROOT="/lambda/nfs/${LOCAL_FS_NAME}"
DEST_DIR="${LOCAL_ROOT}/$(dirname "${S3_SRC_SUBDIR}")"
CKPT_DIR="${LOCAL_ROOT}/${S3_SRC_SUBDIR}"
export RUN_ROOT="${LOCAL_ROOT}/rl_runs"

# --- Helper -------------------------------------------------------------------
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found"; exit 1; }
}

# --- 0) Ensure Miniconda + environment ----------------------------------------
ensure_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[conda] Installing Miniconda..."
    curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
  fi
  # shell hook + env
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx rl; then
    echo "[conda] Creating env 'rl'..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
    conda create -n rl python=3.11 -y
  fi
  conda activate rl
}

# --- 1) Clone/refresh repo ----------------------------------------------------
ensure_repo() {
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  cd "$REPO_DIR"
  git fetch --all -p
  git checkout "$REPO_BRANCH"
  git pull --ff-only
  # project deps
  pip install -r requirements.txt
}

# --- 2) rclone configuration for Lambda S3 endpoint ---------------------------
# Uses environment variables from ~/.lambda_s3.env (created by you)
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_REGION=us-east-3
#   export S3_ENDPOINT_URL="https://files.us-east-3.lambda.ai"
#   export AWS_EC2_METADATA_DISABLED=true
ensure_rclone() {
  if ! command -v rclone >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y rclone
  fi
  if [ -f "$HOME/.lambda_s3.env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.lambda_s3.env"
  else
    echo "WARNING: $HOME/.lambda_s3.env not found. Create it with your S3 credentials."
  fi
  require_cmd rclone
  rclone config create lambda_east3 s3 provider=Other env_auth=true \
    region="${AWS_REGION:-us-east-3}" endpoint="${S3_ENDPOINT_URL:-https://files.us-east-3.lambda.ai}" >/dev/null 2>&1 || true
  # Rclone S3 backend honors env_auth and custom endpoints; see docs. [1][2]
}

# --- 3) Copy checkpoints from S3 to local FS ---------------------------------
copy_checkpoints() {
  mkdir -p "${DEST_DIR}"
  echo "[rclone] Copying s3://${S3_UUID}/${S3_SRC_SUBDIR}  ->  ${DEST_DIR}"
  rclone copy \
    "lambda_east3:${S3_UUID}/${S3_SRC_SUBDIR}" \
    "${DEST_DIR}" \
    --transfers=8 --checkers=8 --s3-chunk-size=64M \
    --retries=5 --low-level-retries=20 \
    --ignore-checksum --size-only --progress

  echo "[rclone] Verifying sizes..."
  rclone check \
    "lambda_east3:${S3_UUID}/${S3_SRC_SUBDIR}" \
    "${DEST_DIR}/$(basename "${S3_SRC_SUBDIR}")" \
    --size-only || true

  ls -lh "${DEST_DIR}/$(basename "${S3_SRC_SUBDIR}")" || true
}

# --- 4) Persistent run directory for TensorBoard ------------------------------
ensure_run_root() {
  mkdir -p "${RUN_ROOT}"
  test -w "${RUN_ROOT}" || { echo "ERROR: RUN_ROOT not writable: ${RUN_ROOT}"; exit 1; }
  echo "[runs] RUN_ROOT=${RUN_ROOT}"
  # TensorBoard consumes this directory. CLI flags are standard. [3]
}

# --- 5) Start TensorBoard in tmux ---------------------------------------------
start_tensorboard() {
  require_cmd tmux
  if tmux has-session -t "${TMUX_TB}" 2>/dev/null; then
    echo "[tmux] Session '${TMUX_TB}' already exists; leaving it running."
  else
    echo "[tmux] Starting TensorBoard on 127.0.0.1:${TB_PORT} in session '${TMUX_TB}'"
    tmux new -s "${TMUX_TB}" -d \
      "tensorboard --logdir '${RUN_ROOT}' --host 127.0.0.1 --port ${TB_PORT}"
  fi
  tmux ls | grep "${TMUX_TB}" || true
  # You will tunnel from Windows with: ssh -L ${TB_PORT}:127.0.0.1:${TB_PORT} ... [4][5]
}

# --- 6) GPU monitoring (nvtop) ------------------------------------------------
ensure_nvtop() {
  if ! command -v nvtop >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y nvtop
  fi
  echo "To run nvtop interactively:  nvtop"
  echo "Or a lightweight monitor:    watch -n 1 -- 'nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv'"
  # nvtop project reference. [6]
}

# --- 7) (Optional) Start training in tmux -------------------------------------
start_training_stub() {
  require_cmd tmux
  cd "$REPO_DIR"
  if tmux has-session -t "${TMUX_TRAIN}" 2>/dev/null; then
    echo "[tmux] Training session '${TMUX_TRAIN}' already exists."
  else
    echo "[tmux] Launching training in '${TMUX_TRAIN}'"
    tmux new -s "${TMUX_TRAIN}" -d \
      "torchrun --nproc_per_node=1 -m rl_training.runners.rl_runner --cfg '${TRAIN_CFG}' --ckpt '${CKPT_DIR}'"
  fi
  tmux ls | grep "${TMUX_TRAIN}" || true
}

# --- Execute all steps --------------------------------------------------------
ensure_conda
ensure_repo
ensure_rclone
copy_checkpoints
ensure_run_root
start_tensorboard
ensure_nvtop
# Uncomment to start training automatically:
# start_training_stub

echo "Done. Next (on Windows): run windows_port_forward.ps1 and open http://localhost:${TB_PORT}"
