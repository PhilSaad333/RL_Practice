#!/usr/bin/env bash
set -euo pipefail

# ========= USER OVERRIDES (edit or pass as env at runtime) ====================
# Name of the per-session local filesystem you attached in the Lambda console:
: "${LOCAL_FS_NAME:=localfs}"

# UUID of your S3-compatible Lambda filesystem (the bucket-like ID in us-east-3):
: "${S3_UUID:=CHANGE-ME-UUID}"

# Which checkpoint subtree to copy from S3 to the local filesystem:
: "${S3_SRC_SUBDIR:=checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156}"

# Training repo:
: "${REPO_URL:=https://github.com/PhilSaad333/RL_Practice.git}"
: "${REPO_DIR:=$HOME/RL_Practice}"
: "${REPO_BRANCH:=main}"

# TensorBoard port (remote). If busy, change and re-run:
: "${TB_PORT:=16006}"

# tmux session names:
: "${TMUX_TB:=tb}"
: "${TMUX_TRAIN:=train}"

# Training config (example). Adjust as needed:
: "${TRAIN_CFG:=rl_training/cfg/testconfig.yaml}"

# Whether to auto-start training from this script (0 = no, 1 = yes):
: "${START_TRAIN:=0}"
# ============================================================================

LOCAL_ROOT="/lambda/nfs/${LOCAL_FS_NAME}"
DEST_DIR="${LOCAL_ROOT}/$(dirname "${S3_SRC_SUBDIR}")"
CKPT_DIR="${LOCAL_ROOT}/${S3_SRC_SUBDIR}"
export RUN_ROOT="${LOCAL_ROOT}/rl_runs"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found"; exit 1; }; }

msg() { printf "\n[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

# 0) Credentials
if [ -f "$HOME/.lambda_s3.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.lambda_s3.env"
else
  cat <<EOF
ERROR: \$HOME/.lambda_s3.env not found.

Create it once with your S3 credentials, e.g.:

cat > ~/.lambda_s3.env <<'ENV'
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
export AWS_REGION=us-east-3
export S3_ENDPOINT_URL=https://files.us-east-3.lambda.ai
export AWS_EC2_METADATA_DISABLED=true
ENV
chmod 600 ~/.lambda_s3.env

Then re-run this script.
EOF
  exit 1
fi

# 1) Miniconda + env
if ! command -v conda >/dev/null 2>&1; then
  msg "Installing Miniconda…"
  curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
fi
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx rl; then
  msg "Creating conda env 'rl'…"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  conda create -n rl python=3.11 -y
fi
conda activate rl

# 2) Repo
if [ ! -d "$REPO_DIR/.git" ]; then
  msg "Cloning repo…"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --all -p
git checkout "$REPO_BRANCH"
git pull --ff-only
pip install -r requirements.txt

# 3) rclone
if ! command -v rclone >/dev/null 2>&1; then
  msg "Installing rclone…"
  sudo apt-get update -y
  sudo apt-get install -y rclone
fi
# Configure rclone S3 remote using env credentials + custom endpoint
rclone config create lambda_east3 s3 provider=Other env_auth=true \
  region="${AWS_REGION}" endpoint="${S3_ENDPOINT_URL}" >/dev/null 2>&1 || true

# 4) Copy checkpoints from S3 to local FS
msg "Ensuring local FS path: ${DEST_DIR}"
mkdir -p "${DEST_DIR}"

msg "Copying from S3 (uuid=${S3_UUID}) subdir '${S3_SRC_SUBDIR}' to ${DEST_DIR}"
rclone copy \
  "lambda_east3:${S3_UUID}/${S3_SRC_SUBDIR}" \
  "${DEST_DIR}" \
  --transfers=8 --checkers=8 --s3-chunk-size=64M \
  --retries=5 --low-level-retries=20 \
  --ignore-checksum --size-only --progress

# 5) Make runs persistent on local FS (NO symlink needed in repo)
msg "Creating persistent RUN_ROOT at: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
test -w "${RUN_ROOT}" || { echo "ERROR: RUN_ROOT not writable: ${RUN_ROOT}"; exit 1; }
export RUN_ROOT

# 6) TensorBoard in tmux
need tmux
if tmux has-session -t "${TMUX_TB}" 2>/dev/null; then
  msg "tmux session '${TMUX_TB}' already exists; leaving it running."
else
  msg "Starting TensorBoard on 127.0.0.1:${TB_PORT} in tmux session '${TMUX_TB}'"
  tmux new -s "${TMUX_TB}" -d \
    "tensorboard --logdir '${RUN_ROOT}' --host 127.0.0.1 --port ${TB_PORT}"
fi

# 7) GPU monitoring utility
if ! command -v nvtop >/dev/null 2>&1; then
  msg "Installing nvtop…"
  sudo apt-get update -y
  sudo apt-get install -y nvtop
fi

# 8) Optional: auto-start training in tmux
if [ "${START_TRAIN}" = "1" ]; then
  if tmux has-session -t "${TMUX_TRAIN}" 2>/dev/null; then
    msg "tmux session '${TMUX_TRAIN}' already exists; leaving it running."
  else
    msg "Starting training in tmux session '${TMUX_TRAIN}'"
    tmux new -s "${TMUX_TRAIN}" -d \
      "cd '${REPO_DIR}' && torchrun --nproc_per_node=1 -m rl_training.runners.rl_runner --cfg '${TRAIN_CFG}' --ckpt '${CKPT_DIR}'"
  fi
fi

msg "All set."
echo "Next (on Windows): run your port-forward script and open http://localhost:${TB_PORT}"
echo "To attach TensorBoard:   tmux attach -t ${TMUX_TB}"
echo "To attach training:      tmux attach -t ${TMUX_TRAIN}"
