#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_SCRIPT="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/ds_metaworld_ppo_mlp_add_vtrace_with_param_more_stats.py"
CLIP_CONFIG_PATH="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/config/clip.yml"

# --- Default Parameters for Single Run ---
DEFAULT_TASK_NAME="coffee-button-v3"
DEFAULT_CLIP_MODE="sapo_soft_clip"
DEFAULT_SEED=42
DEFAULT_CUDA_DEVICES="6,7"
DEFAULT_NUM_TRAINER_GPUS=1
DEFAULT_NUM_ROLLOUT_WORKERS=16
DEFAULT_NUM_EVAL_WORKERS=16
DEFAULT_TRAIN_BATCH_SIZE=512
DEFAULT_TRAIN_ITERS=100000
# log_gauss_clip 专用：sigma 列表（空格分隔多值自动展开）
DEFAULT_LOG_GAUSS_SIGMAS="0.5 1 2"
# 新增：采样过滤参数（用于 Recency Window Ablation）
DEFAULT_REPLAY_RECENT_FRAC=1.0        # 最新比例窗口：只从最新 f% 数据采样
DEFAULT_REPLAY_MAX_VERSION_GAP="inf"  # 版本差窗口：只采样 version_gap <= max_gap 的数据
# 日志后端配置
DEFAULT_LOG_BACKEND="tensorboard"            # 日志后端选择: tensorboard/swanlab/both
# 如果外部未显式传 LOG_BACKEND，则使用默认值；防止环境变量残留把后端改成 swanlab
LOG_BACKEND="${LOG_BACKEND:-${DEFAULT_LOG_BACKEND}}"
# 全局开关：是否启用 SwanLab
if [[ "${LOG_BACKEND}" == "swanlab" || "${LOG_BACKEND}" == "both" ]]; then
  USE_SWANLAB=1
else
  USE_SWANLAB=0
fi

# --- 监控相关配置 ---
MONITOR_INTERVAL=1800          # 监控刷新间隔（秒，半小时）
HEALTH_CHECK_INTERVAL=3600     # 健康检查间隔（秒）
STALL_THRESHOLD=3600           # 判定卡住阈值（秒，日志长时间无更新，1 小时）
STATUS_FILE_NAME="job_status.csv"
META_FILE_NAME="job_meta.csv"
TASK_DONE_FILE_NAME="tasks_completed.log"
TASK_SWANLAB_LOGS_NAME="task_swanlab_logs.txt"
SWANLAB_STATUS_FILE_NAME="swanlab_status.log"

# --- DeepSpeed 初始化配置 ---
DEEPSPEED_INIT_TIMEOUT="${DEEPSPEED_INIT_TIMEOUT:-600}"  # DeepSpeed 初始化超时时间（秒，默认 10 分钟）

# --- 并行执行配置 ---
MAX_PARALLEL_JOBS=6  # 最大并行任务数（默认6个）
# 多 GPU 配置：每个并行任务使用的 GPU 组合
# 注意：多个任务可以共享同一组 GPU（前提是显存足够大）
# 格式：每行一个 GPU 配置（逗号分隔的 GPU ID）
GPU_CONFIGS=(
  "6,7"  # 第1个任务使用 GPU 6,7
  "6,7"  # 第2个任务使用 GPU 6,7
  "6,7"  # 第3个任务使用 GPU 6,7
  "6,7"  # 第4个任务使用 GPU 6,7
  "6,7"  # 第5个任务使用 GPU 6,7
  "6,7"  # 第6个任务使用 GPU 6,7
)

# 启动延迟（秒）：避免多个任务同时初始化导致资源竞争和端口冲突
# ⚠️ 重要：这个延迟必须足够长，让前一个任务完全启动并占用端口
STARTUP_DELAY=90  # 每个新任务启动前等待的秒数（推荐 15-25 秒）

# --- Batch Experiment Configuration ---
# 根据注释区整理的任务列表（vtrace 版本常用任务）
TASKS=(
  "sweep-into-v3"
  "drawer-open-v3"
  "door-open-v3"
  "button-press-topdown-v3"
  "handle-press-v3"
  "push-v3"
  # "peg-insert-side-v3"
  # "pick-place-v3"
  # "plate-slide-v3"
  # "coffee-button-v3"
  # "soccer-v3"
)

# 根据注释区整理的裁剪模式列表（vtrace 版本常用模式）
CLIP_MODES=(
  "clip"                # 对应注释中的 "ppo" (PPO standard hard clip)
  "soft_clip_alpha-0"   # 对应注释中的 "soft clip(alpha=0)"
  # "soft_clip_alpha-1"   # 对应注释中的 "soft clip(alpha=1)"
  # "soft_clip_alpha-2"   # 对应注释中的 "soft clip(alpha=2)"
  "sapo_soft_clip"      # 对应注释中的 "sapo"
  "log_gauss_clip"      # 对应注释中的 "log gauss clip"
)

# --- Helper: generate clip config for log_gauss_clip ---
make_log_gauss_clip_config() {
  local sigma="$1"
  local tmp_cfg
  tmp_cfg=$(mktemp /tmp/clip_log_gauss_XXXX.yml)
  cat > "${tmp_cfg}" <<EOF
log_gauss_clip:
  # coeff = exp(-0.5 * (log(r+eps)/sigma)^2)
  sigma: ${sigma}
  eps: 1e-9
EOF
  echo "${tmp_cfg}"
}

# --- Helper: Record job status to CSV ---
record_job_status() {
  local job_idx="$1"
  local task="$2"
  local clip_mode="$3"
  local status="$4"  # "completed", "failed", "timeout", "killed"
  local exit_code="${5:-}"
  local error_msg="${6:-}"
  local start_time="$7"
  local end_time="$8"
  local log_file="${9:-}"
  local sigma="${10:-}"
  
  # 计算持续时间（秒）
  local duration_seconds=0
  if [[ -n "${start_time}" && -n "${end_time}" ]]; then
    # 尝试使用 date -d（GNU date），如果失败则使用其他方法
    local start_epoch=$(date -d "${start_time}" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "${start_time}" +%s 2>/dev/null || echo "0")
    local end_epoch=$(date -d "${end_time}" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "${end_time}" +%s 2>/dev/null || echo "0")
    if [[ "${start_epoch}" != "0" && "${end_epoch}" != "0" ]]; then
      duration_seconds=$((end_epoch - start_epoch))
    fi
  fi
  
  # 构建CSV行
  local csv_line="${job_idx},${task},${clip_mode}"
  if [[ -n "${sigma}" ]]; then
    csv_line="${csv_line}_sigma-${sigma}"
  fi
  csv_line="${csv_line},${status},${exit_code},\"${error_msg}\",${start_time},${end_time},${duration_seconds},${log_file}"
  
  # 追加到状态文件（如果LOG_DIR已设置）
  if [[ -n "${LOG_DIR:-}" ]]; then
    local status_file="${LOG_DIR}/job_status.csv"
    # 如果文件不存在，先写入表头
    if [[ ! -f "${status_file}" ]]; then
      echo "job_idx,task,clip_mode,status,exit_code,error_msg,start_time,end_time,duration_seconds,log_file" > "${status_file}"
    fi
    echo "${csv_line}" >> "${status_file}"
    echo "[Status] 已记录任务状态: Job ${job_idx} - ${status}"
  fi
}

# --- Helper: Extract error message from log file ---
extract_error_from_log() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo "Log file not found"
    return
  fi
  
  # 尝试提取最后几行中的错误信息
  local last_lines=$(tail -n 50 "${log_file}" 2>/dev/null)
  
  # 查找常见的错误模式
  if echo "${last_lines}" | grep -q "Traceback"; then
    # 提取Traceback后的错误信息
    echo "${last_lines}" | grep -A 5 "Traceback" | tail -n 3 | tr '\n' ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  elif echo "${last_lines}" | grep -q "Error\|Exception\|Failed\|Fatal"; then
    # 提取包含Error/Exception/Failed/Fatal的行
    echo "${last_lines}" | grep -i "Error\|Exception\|Failed\|Fatal" | tail -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  else
    # 返回最后一行
    echo "${last_lines}" | tail -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  fi
}

# --- Helper: Check if job is stuck (no log update for a long time) ---
check_job_stuck() {
  local log_file="$1"
  local timeout_seconds="${2:-7200}"  # 默认2小时无更新视为卡住
  
  if [[ ! -f "${log_file}" ]]; then
    return 1  # 日志文件不存在，无法判断
  fi
  
  local last_modify=$(stat -c %Y "${log_file}" 2>/dev/null || stat -f %m "${log_file}" 2>/dev/null)
  local current_time=$(date +%s)
  local time_since_update=$((current_time - last_modify))
  
  if [[ ${time_since_update} -gt ${timeout_seconds} ]]; then
    return 0  # 卡住了
  else
    return 1  # 正常
  fi
}


# --- Core Training Function ---
run_training() {
  local log_prefix="${JOB_PREFIX:-}"
  local start_time=$(date '+%Y-%m-%d %H:%M:%S')
  
  # ✅ 修复：根据当前 LOG_BACKEND 动态计算 USE_SWANLAB（不依赖全局变量）
  local USE_SWANLAB=0
  if [[ "${LOG_BACKEND}" == "swanlab" || "${LOG_BACKEND}" == "both" ]]; then
    USE_SWANLAB=1
  fi
  
  echo "===================================================================="
  echo "${log_prefix}Starting training:"
  echo "${log_prefix}  -> Task:                 ${TASK_NAME}"
  echo "${log_prefix}  -> Clip Mode:            ${CLIP_MODE}"
  echo "${log_prefix}  -> Seed:                 ${SEED}"
  echo "${log_prefix}  -> CUDA:                 ${CUDA_VISIBLE_DEVICES}"
  echo "${log_prefix}  -> Replay Recent Frac:   ${REPLAY_RECENT_FRAC}"
  echo "${log_prefix}  -> Replay Max Version Gap: ${REPLAY_MAX_VERSION_GAP}"
  echo "${log_prefix}  -> Train Iters:          ${TRAIN_ITERS}"
  echo "${log_prefix}  -> Log Backend:          ${LOG_BACKEND}"
  if [[ "${CLIP_MODE}" == "log_gauss_clip" ]]; then
    echo "${log_prefix}  -> Log Gauss Sigma:      ${LOG_GAUSS_SIGMA}"
    echo "${log_prefix}  -> Clip Config:          ${CLIP_CONFIG}"
  fi
  echo "${log_prefix}"
  echo "${log_prefix}📂 日志目录结构:"
  echo "${log_prefix}  -> 任务基础目录: runs/MetaWorld/${TASK_NAME}/"
  if [[ "${USE_SWANLAB}" -eq 1 ]]; then
    if [[ -n "${SWANLAB_DIR:-}" ]]; then
      echo "${log_prefix}  -> SwanLab 日志: ${SWANLAB_DIR}"
    else
      echo "${log_prefix}  -> SwanLab 日志: （在 ${CLIP_MODE}/ 子目录下，带时间戳）"
    fi
  else
    echo "${log_prefix}  -> SwanLab 已禁用 (LOG_BACKEND=${LOG_BACKEND})"
  fi
  if [[ -n "${TENSORBOARD_DIR:-}" ]]; then
    echo "${log_prefix}  -> TensorBoard 日志: ${TENSORBOARD_DIR}"
  else
    echo "${log_prefix}  -> TensorBoard 日志: （在 ${CLIP_MODE}/ 子目录下，带时间戳）"
  fi
  echo "===================================================================="

  # 显式导出 CUDA 可见设备，便于底层库读取
  export CUDA_VISIBLE_DEVICES
  
  # 设置 SwanLab 离线模式（无需外网连接）
  if [[ "${USE_SWANLAB}" -eq 1 ]]; then
    export SWANLAB_OFFLINE=1
    # 如果设置了 SWANLAB_DIR，导出它
    if [[ -n "${SWANLAB_DIR:-}" ]]; then
      export SWANLAB_DIR
    fi
  else
    unset SWANLAB_OFFLINE
    unset SWANLAB_DIR
  fi
  
  # 导出 TensorBoard 目录（如果设置了）
  if [[ -n "${TENSORBOARD_DIR:-}" ]]; then
    export TENSORBOARD_DIR
  fi
  
  # 设置 DeepSpeed 初始化超时时间
  export DEEPSPEED_INIT_TIMEOUT="${DEEPSPEED_INIT_TIMEOUT:-600}"

  # 执行 Python 训练脚本（将输出重定向到日志文件，在并行模式下）
  if [[ -n "${LOG_FILE:-}" ]]; then
    # 在日志文件第一行记录进程信息
    {
      echo "[PROCESS_INFO] PID=$$ | Task=${TASK_NAME} | ClipMode=${CLIP_MODE} | Seed=${SEED} | CUDA=${CUDA_VISIBLE_DEVICES} | ReplayRecentFrac=${REPLAY_RECENT_FRAC} | ReplayMaxVersionGap=${REPLAY_MAX_VERSION_GAP} | TrainIters=${TRAIN_ITERS} | LogBackend=${LOG_BACKEND}${LOG_GAUSS_SIGMA:+ | LogGaussSigma=${LOG_GAUSS_SIGMA}} | StartTime=${start_time}"
      ${PYTHON_BIN} "${PYTHON_SCRIPT}" \
        --task_name "${TASK_NAME}" \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --num-trainer-gpus "${NUM_TRAINER_GPUS}" \
        --num-rollout-workers "${NUM_ROLLOUT_WORKERS}" \
        --num-eval-workers "${NUM_EVAL_WORKERS}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --seed "${SEED}" \
        --clip-mode "${CLIP_MODE}" \
        --clip-config "${CLIP_CONFIG}" \
        --replay-recent-frac "${REPLAY_RECENT_FRAC}" \
        --replay-max-version-gap "${REPLAY_MAX_VERSION_GAP}" \
        --train-iters "${TRAIN_ITERS}" \
        --log-backend "${LOG_BACKEND}" \
        2>&1
    } > "${LOG_FILE}"
  else
    ${PYTHON_BIN} "${PYTHON_SCRIPT}" \
      --task_name "${TASK_NAME}" \
      --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
      --num-trainer-gpus "${NUM_TRAINER_GPUS}" \
      --num-rollout-workers "${NUM_ROLLOUT_WORKERS}" \
      --num-eval-workers "${NUM_EVAL_WORKERS}" \
      --train-batch-size "${TRAIN_BATCH_SIZE}" \
      --seed "${SEED}" \
      --clip-mode "${CLIP_MODE}" \
      --clip-config "${CLIP_CONFIG}" \
      --replay-recent-frac "${REPLAY_RECENT_FRAC}" \
      --replay-max-version-gap "${REPLAY_MAX_VERSION_GAP}" \
      --train-iters "${TRAIN_ITERS}" \
      --log-backend "${LOG_BACKEND}"
  fi
  
  local exit_code=$?
  local end_time=$(date '+%Y-%m-%d %H:%M:%S')
  local error_msg=""
  
  if [[ ${exit_code} -eq 0 ]]; then
    echo "${log_prefix}--- ✓ Finished training for Task: ${TASK_NAME}, Clip Mode: ${CLIP_MODE} ---"
    # 记录成功状态
    if [[ -n "${JOB_IDX:-}" ]]; then
      record_job_status "${JOB_IDX}" "${TASK_NAME}" "${CLIP_MODE}" "completed" "${exit_code}" "" "${start_time}" "${end_time}" "${LOG_FILE:-}" "${LOG_GAUSS_SIGMA:-}"
    fi
  else
    echo "${log_prefix}--- ✗ Failed training for Task: ${TASK_NAME}, Clip Mode: ${CLIP_MODE} (exit code: ${exit_code}) ---"
    # 提取错误信息
    if [[ -n "${LOG_FILE:-}" && -f "${LOG_FILE}" ]]; then
      error_msg=$(extract_error_from_log "${LOG_FILE}")
    else
      error_msg="Exit code: ${exit_code}"
    fi
    # 记录失败状态
    if [[ -n "${JOB_IDX:-}" ]]; then
      record_job_status "${JOB_IDX}" "${TASK_NAME}" "${CLIP_MODE}" "failed" "${exit_code}" "${error_msg}" "${start_time}" "${end_time}" "${LOG_FILE:-}" "${LOG_GAUSS_SIGMA:-}"
    fi
  fi
  echo
  
  return ${exit_code}
}

# --- 全局变量：跟踪所有启动的任务 PID ---
declare -a RUNNING_JOB_PIDS=()

# --- Helper Function: 更新存活的子shell，并统计运行中的任务 ---
count_running_jobs() {
  local new_shell_pids=()
  local shell_count=0
  for pid in "${RUNNING_JOB_PIDS[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      new_shell_pids+=("${pid}")
      shell_count=$((shell_count + 1))
    fi
  done
  RUNNING_JOB_PIDS=("${new_shell_pids[@]}")
  echo "${shell_count}"
}

# --- Helper Function: 通过子shell查找其子进程中的 Python 训练进程 ---
count_running_jobs_by_python() {
  local python_script_name
  python_script_name="$(basename "${PYTHON_SCRIPT}")"
  local count=0
  local seen=""

  # 优先基于记录的子 shell PID 找子进程，避免错误匹配其他用户的训练
  for shell_pid in "${RUNNING_JOB_PIDS[@]}"; do
    # 子 shell 不存活则跳过
    if ! ps -p "${shell_pid}" > /dev/null 2>&1; then
      continue
    fi
    # 查找子 shell 的直接子进程里命令行包含 python_script_name 的进程
    local childs
    childs=$(pgrep -P "${shell_pid}" -f "${python_script_name}" 2>/dev/null || true)
    if [[ -n "${childs}" ]]; then
      # 去重计数
      for c in ${childs}; do
        if [[ " ${seen} " != *" ${c} "* ]]; then
          seen="${seen} ${c}"
          count=$((count + 1))
        fi
      done
    fi
  done

  # ✅ 修改：只统计当前脚本实例启动的任务，不使用全局兜底逻辑
  # 这样多个批量训练脚本实例可以独立运行，互不干扰
  # 兜底逻辑已注释，避免误统计其他脚本实例的任务
  # if [[ ${count} -eq 0 ]]; then
  #   count=$(pgrep -f "${python_script_name}" 2>/dev/null | wc -l)
  # fi

  echo "${count}"
}

# --- Helper Function: Wait for Available GPU Slot ---
wait_for_slot() {
  # ✅ 简化：使用 jobs -r 统计当前 shell 的后台任务
  # 这样多个脚本实例可以独立运行，互不干扰
  # jobs -r 只显示当前 shell 启动的运行中任务，不受其他脚本影响
  local running_jobs=$(jobs -r | wc -l)
  while [[ $running_jobs -ge ${MAX_PARALLEL_JOBS} ]]; do
    echo "当前有 $running_jobs 个任务在运行，等待空闲槽位..."
    sleep 5
    running_jobs=$(jobs -r | wc -l)
  done
  echo "检测到空闲槽位，当前运行任务数: $running_jobs"
}

# --- Helper Function: Check GPU Memory ---
check_gpu_memory() {
  if command -v nvidia-smi &> /dev/null; then
    echo
    echo "========== GPU 显存状态 =========="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
      awk -F', ' '{printf "GPU %s (%s): %d MB / %d MB (%.1f%% used)\n", $1, $2, $3, $4, ($3/$4)*100}'
    echo "=================================="
    echo
  fi
}

# --- Helper: 构造进度条 ---
make_progress_bar() {
  local percent="$1"
  local width=30
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local bar_filled=""
  local bar_empty=""
  if (( filled > 0 )); then
    printf -v bar_filled "%*s" "${filled}" ""
    bar_filled=${bar_filled// /█}
  fi
  if (( empty > 0 )); then
    printf -v bar_empty "%*s" "${empty}" ""
    bar_empty=${bar_empty// /░}
  fi
  printf "%s%s" "${bar_filled}" "${bar_empty}"
}

# --- Helper: 记录 SwanLab 离线日志状态 ---
record_swanlab_status() {
  local job_id="$1"
  local task="$2"
  local clip="$3"
  local status="$4"   # ok / missing / error
  local message="$5"
  local status_file="${LOG_DIR}/${SWANLAB_STATUS_FILE_NAME}"
  mkdir -p "${LOG_DIR}"
  if [[ ! -f "${status_file}" ]]; then
    echo "job_id,task,clip,status,message,timestamp" > "${status_file}"
  fi
  echo "${job_id},${task},${clip},${status},\"${message}\",$(date '+%Y-%m-%d %H:%M:%S')" >> "${status_file}"
}

# --- Helper: 监控所有任务状态，并在异常时终止 ---
monitor_jobs() {
  local log_dir="$1"
  local total_jobs="$2"
  local expected_per_task="$3"

  local meta_file="${log_dir}/${META_FILE_NAME}"
  local status_file="${log_dir}/${STATUS_FILE_NAME}"
  local task_done_file="${log_dir}/${TASK_DONE_FILE_NAME}"
  local task_swanlab_file="${log_dir}/${TASK_SWANLAB_LOGS_NAME}"

  mkdir -p "${log_dir}"
  [[ ! -f "${task_done_file}" ]] && touch "${task_done_file}"
  [[ ! -f "${task_swanlab_file}" ]] && touch "${task_swanlab_file}"

  local last_health_check=0

  while true; do
    local now
    now=$(date +%s)

    local completed=0
    local failed=0
    local running=0
    local stalled=0

    declare -A task_done_count=()
    declare -A task_running_count=()
    declare -A task_failed_count=()
    declare -A task_fail_flag=()

    local tmp_status="${status_file}.tmp"
    echo "job_id,task,clip,status,pid,start_time,log_file,swanlab_dir,tensorboard_dir,last_update" > "${tmp_status}"

    # 读取任务元数据并计算状态
    while IFS=, read -r job_id task clip pid log_file swanlab_dir tensorboard_dir start_time; do
      # 跳过表头
      if [[ "${job_id}" == "job_id" ]] || [[ -z "${job_id}" ]]; then
        continue
      fi

      local status="running"
      local last_update=0
      if [[ -f "${log_file}" ]]; then
        last_update=$(stat -c %Y "${log_file}" 2>/dev/null || echo 0)
      else
        last_update=${now}
      fi

      if kill -0 "${pid}" 2>/dev/null; then
        # 进程仍在运行，检查是否卡住
        if (( now - last_update > STALL_THRESHOLD )); then
          status="stalled"
          stalled=$((stalled + 1))
          failed=$((failed + 1))
          echo "[监控] 检测到任务卡住，终止进程: Job=${job_id}, Task=${task}, Clip=${clip}, PID=${pid}"
          kill -9 "${pid}" 2>/dev/null || true
          # 标记失败文件
          local clip_dir
          clip_dir="$(dirname "${log_file}")"
          touch "${clip_dir}/FAILED" 2>/dev/null || true
        else
          status="running"
          running=$((running + 1))
          local cur_run=${task_running_count["${task}"]:-0}
          task_running_count["${task}"]=$((cur_run + 1))
        fi
      else
        # 进程已结束，判断成功/失败
        if [[ -f "${log_file}" ]] && grep -q "Finished training" "${log_file}"; then
          status="done"
          completed=$((completed + 1))
          # 标记完成文件
          local clip_dir
          clip_dir="$(dirname "${log_file}")"
          touch "${clip_dir}/DONE" 2>/dev/null || true
          
          # ✅ 修复：从日志文件中提取 LOG_BACKEND，决定检查哪些日志
          local log_backend="tensorboard"  # 默认值
          if [[ -f "${log_file}" ]]; then
            log_backend=$(grep -oP '(?<=LogBackend=)[^ |]+' "${log_file}" 2>/dev/null | head -n 1 || echo "tensorboard")
          fi
          
          # 根据 LOG_BACKEND 检查相应的日志目录
          local log_check_failed=0
          if [[ "${log_backend}" == "swanlab" || "${log_backend}" == "both" ]]; then
            # 检查 SwanLab 离线日志是否存在
            if [[ ! -d "${swanlab_dir}" ]] || [[ -z "$(find "${swanlab_dir}" -maxdepth 1 -type d -name 'run-*' 2>/dev/null)" ]]; then
              log_check_failed=1
              record_swanlab_status "${job_id}" "${task}" "${clip}" "missing" "未发现 SwanLab 离线日志目录或 run-*"
            else
              record_swanlab_status "${job_id}" "${task}" "${clip}" "ok" "SwanLab 离线日志已生成"
            fi
          fi
          
          # TensorBoard 不需要特殊检查（训练过程中持续写入，没有固定的目录结构要求）
          
          if [[ ${log_check_failed} -eq 1 ]]; then
            status="failed"
            failed=$((failed + 1))
            task_fail_flag["${task}"]=1
            touch "${clip_dir}/FAILED" 2>/dev/null || true
          fi
        elif [[ -f "${log_file}" ]] && (grep -qi "Failed training" "${log_file}" || grep -qi "Traceback" "${log_file}"); then
          status="failed"
          failed=$((failed + 1))
          local clip_dir
          clip_dir="$(dirname "${log_file}")"
          touch "${clip_dir}/FAILED" 2>/dev/null || true
          record_swanlab_status "${job_id}" "${task}" "${clip}" "error" "训练失败，日志含错误/Traceback"
        else
          status="failed"
          failed=$((failed + 1))
          local clip_dir
          clip_dir="$(dirname "${log_file}")"
          touch "${clip_dir}/FAILED" 2>/dev/null || true
          record_swanlab_status "${job_id}" "${task}" "${clip}" "error" "训练异常退出，未检测到完成标记"
        fi
      fi

      if [[ "${status}" == "done" ]]; then
        local cur_done=${task_done_count["${task}"]:-0}
        task_done_count["${task}"]=$((cur_done + 1))
      elif [[ "${status}" == "failed" || "${status}" == "stalled" ]]; then
        task_fail_flag["${task}"]=1
        local cur_fail=${task_failed_count["${task}"]:-0}
        task_failed_count["${task}"]=$((cur_fail + 1))
      fi

      echo "${job_id},${task},${clip},${status},${pid},${start_time},${log_file},${swanlab_dir},${tensorboard_dir},${last_update}" >> "${tmp_status}"
    done < "${meta_file}"

    mv "${tmp_status}" "${status_file}"

    # 打印整体进度
    local percent=0
    if (( total_jobs > 0 )); then
      percent=$((completed * 100 / total_jobs))
    fi
    local bar
    bar=$(make_progress_bar "${percent}")
    echo "[${bar}] ${percent}% | 完成 ${completed}/${total_jobs} | 运行 ${running} | 失败 ${failed} | 卡住 ${stalled}"

    # 每小时记录一次健康检查标记
    if (( now - last_health_check >= HEALTH_CHECK_INTERVAL )); then
      last_health_check=${now}
      echo "[监控] 健康检查: $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # 按任务打印实时状态与进度条
    echo "---- 按任务进度 ----"
    for task_name in "${TASKS[@]}"; do
      local done_cnt=${task_done_count["${task_name}"]:-0}
      local run_cnt=${task_running_count["${task_name}"]:-0}
      local fail_cnt=${task_failed_count["${task_name}"]:-0}
      local expected_cnt=${expected_per_task}
      local task_pct=0
      if (( expected_cnt > 0 )); then
        task_pct=$((done_cnt * 100 / expected_cnt))
      fi
      local task_bar
      task_bar=$(make_progress_bar "${task_pct}")
      echo "  ${task_name}: [${task_bar}] ${done_cnt}/${expected_cnt} 完成 | 运行 ${run_cnt} | 失败 ${fail_cnt}"
    done
    echo "-------------------"

    # 检查每个任务是否全部完成
    for task_name in "${!task_done_count[@]}"; do
      local done_cnt=${task_done_count["${task_name}"]}
      local expected_cnt=${expected_per_task}
      if (( done_cnt == expected_cnt )) && [[ -z "${task_fail_flag[${task_name}]:-}" ]]; then
        # 如果尚未记录完成，则追加
        if ! grep -q "^${task_name}$" "${task_done_file}" 2>/dev/null; then
          echo "${task_name}" >> "${task_done_file}"
          echo "任务 ${task_name} 已全部完成 (${done_cnt}/${expected_cnt})."
          echo "任务 ${task_name} 已全部完成 (${done_cnt}/${expected_cnt})." >> "${task_swanlab_file}"
          # 记录该任务所有组合的日志目录
          while IFS=, read -r jid tname clip status pid st log swdir tbdir lu; do
            if [[ "${tname}" == "${task_name}" ]]; then
              echo "  - ${clip}:" >> "${task_swanlab_file}"
              echo "      SwanLab: ${swdir}" >> "${task_swanlab_file}"
              echo "      TensorBoard: ${tbdir}" >> "${task_swanlab_file}"
            fi
          done < "${status_file}"
        fi
      fi
    done

    # 终止条件
    if (( completed + failed == total_jobs )); then
      echo "[监控] 所有任务已结束。完成: ${completed}, 失败/异常: ${failed}"
      break
    fi

    sleep "${MONITOR_INTERVAL}"
  done
}

# --- Helper: 重新运行失败/卡死/缺失离线日志的配置组合（顺序重试一次） ---
rerun_failed_jobs() {
  local log_dir="$1"
  local meta_file="${log_dir}/${META_FILE_NAME}"
  local status_file="${log_dir}/${STATUS_FILE_NAME}"
  local retry_idx=0
  declare -A seen

  if [[ ! -f "${status_file}" ]]; then
    return
  fi

  echo "[重试] 开始检查失败/卡死/缺失离线日志的配置..."

  while IFS=, read -r job_id task clip status pid start_time log_file swanlab_dir tensorboard_dir last_update; do
    [[ "${job_id}" == "job_id" || -z "${job_id}" ]] && continue
    if [[ "${status}" == "done" ]]; then
      continue
    fi
    local key="${task}|||${clip}"
    seen["${key}"]=1
  done < "${status_file}"

  if (( ${#seen[@]} == 0 )); then
    echo "[重试] 未发现需要重试的配置组合。"
    return
  fi

  for key in "${!seen[@]}"; do
    local task="${key%%%|||*}"
    local clip_label="${key#*|||}"
    local clip_mode="${clip_label}"
    local sigma=""
    local clip_config_local="${CLIP_CONFIG_PATH}"

    if [[ "${clip_label}" =~ ^log_gauss_clip_sigma-([0-9.+-]+)$ ]]; then
      clip_mode="log_gauss_clip"
      sigma="${BASH_REMATCH[1]}"
      clip_config_local=$(make_log_gauss_clip_config "${sigma}")
    fi

    local gpu_idx=$((retry_idx % ${#GPU_CONFIGS[@]}))
    local assigned_gpu="${GPU_CONFIGS[$gpu_idx]}"
    local task_dir="${log_dir}/${task}/${clip_mode}"
    mkdir -p "${task_dir}"
    local log_file="${task_dir}/job_retry_${retry_idx}_${clip_label}.log"
    local swanlab_dir="${log_dir}/${task}/swanlab_all"
    local tensorboard_dir="${log_dir}/${task}/tensorboard_all"
    mkdir -p "${swanlab_dir}" "${tensorboard_dir}"

    echo "----------------------------------------"
    echo "[重试 $((retry_idx+1))] Task=${task}, Clip=${clip_label}, GPU=${assigned_gpu}"
    echo "[重试 $((retry_idx+1))] Log file: ${log_file}"

    (
      TASK_NAME="${task}"
      CLIP_MODE="${clip_mode}"
      LOG_GAUSS_SIGMA="${sigma}"
      SEED="${SEED:-${DEFAULT_SEED}}"
      CUDA_VISIBLE_DEVICES="${assigned_gpu}"
      NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
      NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
      NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
      CLIP_CONFIG="${clip_config_local}"
      REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
      REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
      TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
      LOG_BACKEND="${LOG_BACKEND:-${DEFAULT_LOG_BACKEND}}"
      LOG_FILE="${log_file}"
      JOB_PREFIX="[Retry $((retry_idx+1))] "
      JOB_IDX="retry_${retry_idx}"
      SWANLAB_DIR="${swanlab_dir}"
      TENSORBOARD_DIR="${tensorboard_dir}"
      export LOG_DIR SWANLAB_DIR TENSORBOARD_DIR

      run_training

      echo "[Retry $((retry_idx+1))] Completed: Task=${task}, Clip=${clip_label}"
    )

    # 追加 meta 以便溯源
    local start_ts=$(date +%s)
    echo "retry_${retry_idx},${task},${clip_label},-,${log_file},${swanlab_dir},${tensorboard_dir},${start_ts}" >> "${meta_file}"

    ((retry_idx++))
  done
}

# --- Helper: Print running background jobs (PID + command) ---
print_running_jobs() {
  # ✅ 简化：使用 jobs 命令统计当前 shell 的后台任务
  local running_jobs=$(jobs -r | wc -l)
  echo "当前后台任务数: ${running_jobs}"
  
  if [[ ${running_jobs} -gt 0 ]]; then
    echo "运行中的后台任务:"
    jobs -r -l
  fi
  
  # 额外信息：显示所有相关的 Python 训练进程（调试用）
  local python_script_name=$(basename "${PYTHON_SCRIPT}")
  local my_python_pids=$(pgrep -f "${python_script_name}" -u $(whoami) 2>/dev/null || true)
  if [[ -n "${my_python_pids}" ]]; then
    local count=$(echo "${my_python_pids}" | wc -l)
    echo "当前用户的 Python 训练进程总数: ${count}"
  fi
}

# --- Helper: Cleanup on exit (e.g., Ctrl+C) ---
cleanup_and_kill_jobs() {
  echo "捕获到退出信号，正在终止后台任务..."
  # 使用 PID 数组
  if [[ ${#RUNNING_JOB_PIDS[@]} -gt 0 ]]; then
    echo "将杀死以下后台进程: ${RUNNING_JOB_PIDS[*]}"
    for pid in "${RUNNING_JOB_PIDS[@]}"; do
      if ps -p "${pid}" > /dev/null 2>&1; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
    done
  fi
  # 也尝试使用 jobs -p（兼容性）
  local pids
  pids=$(jobs -p 2>/dev/null)
  if [[ -n "${pids}" ]]; then
    echo "将杀死以下后台进程（jobs）: ${pids}"
    kill -9 ${pids} 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

trap cleanup_and_kill_jobs INT TERM

# --- Main Script Logic ---
# 如果第一个参数是 "batch"，则运行批处理实验（并行模式）
if [[ "${1:-}" == "batch" ]]; then
  # 解析 log_gauss sigma 列表（空格分隔）
  read -ra LOG_GAUSS_SIGMAS <<< "${LOG_GAUSS_SIGMAS:-${DEFAULT_LOG_GAUSS_SIGMAS}}"
  
  echo "===== Starting Batch Experiment Run (Parallel Mode) ====="
  echo "Maximum parallel jobs: ${MAX_PARALLEL_JOBS}"
  echo "GPU configurations: ${GPU_CONFIGS[@]}"
  echo "Startup delay between jobs: ${STARTUP_DELAY}s"
  echo "⚠️  注意：多个任务共享同一组GPU，请确保显存足够！"
  
  # 显示当前GPU状态
  check_gpu_memory
  echo
  
  # 创建日志目录
  LOG_DIR="logs/parallel_runs_vtrace_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${LOG_DIR}"
  echo "训练日志将保存到: ${LOG_DIR}"
  echo
  
  # 任务组合计数
  job_idx=0  # 任务索引，用于分配 GPU
  total_jobs=0
  expected_per_task=0
  for clip_mode in "${CLIP_MODES[@]}"; do
    if [[ "${clip_mode}" == "log_gauss_clip" ]]; then
      expected_per_task=$((expected_per_task + ${#LOG_GAUSS_SIGMAS[@]}))
    else
      expected_per_task=$((expected_per_task + 1))
    fi
  done
  for task in "${TASKS[@]}"; do
    task_shared_swanlab="${LOG_DIR}/${task}/swanlab_all"
    task_shared_tensorboard="${LOG_DIR}/${task}/tensorboard_all"
    mkdir -p "${task_shared_swanlab}"
    mkdir -p "${task_shared_tensorboard}"
    for clip_mode in "${CLIP_MODES[@]}"; do
      if [[ "${clip_mode}" == "log_gauss_clip" ]]; then
        total_jobs=$((total_jobs + ${#LOG_GAUSS_SIGMAS[@]}))
      else
        total_jobs=$((total_jobs + 1))
      fi
    done
  done

  # 元数据/状态文件
  META_FILE="${LOG_DIR}/${META_FILE_NAME}"
  STATUS_FILE="${LOG_DIR}/${STATUS_FILE_NAME}"
  TASK_DONE_FILE="${LOG_DIR}/${TASK_DONE_FILE_NAME}"
  TASK_SWANLAB_FILE="${LOG_DIR}/${TASK_SWANLAB_LOGS_NAME}"
  echo "job_id,task,clip,pid,log_file,swanlab_dir,tensorboard_dir,start_time" > "${META_FILE}"
  echo "job_id,task,clip,status,pid,start_time,log_file,swanlab_dir,tensorboard_dir,last_update" > "${STATUS_FILE}"
  touch "${TASK_DONE_FILE}" "${TASK_SWANLAB_FILE}"
  
  for task in "${TASKS[@]}"; do
    # ✅ 修复：在每次任务循环开始时，重新设置共享目录路径
    task_shared_swanlab="${LOG_DIR}/${task}/swanlab_all"
    task_shared_tensorboard="${LOG_DIR}/${task}/tensorboard_all"
    
    for clip_mode in "${CLIP_MODES[@]}"; do
      if [[ "${clip_mode}" == "log_gauss_clip" ]]; then
        for sigma in "${LOG_GAUSS_SIGMAS[@]}"; do
          wait_for_slot
          gpu_idx=$((job_idx % ${#GPU_CONFIGS[@]}))
          assigned_gpu="${GPU_CONFIGS[$gpu_idx]}"
          task_dir="${LOG_DIR}/${task}/${clip_mode}"
          mkdir -p "${task_dir}"
          clip_cfg_path=$(make_log_gauss_clip_config "${sigma}")
          clip_label="${clip_mode}_sigma-${sigma}"
          log_file="${task_dir}/job_${job_idx}_${clip_label}.log"

          echo "========================================"
          echo "[Job $((job_idx+1))/${total_jobs}] Starting: Task=${task}, Clip=${clip_mode}, Sigma=${sigma}, GPU=${assigned_gpu}"
          echo "[Job $((job_idx+1))/${total_jobs}] Log file: ${log_file}"
          echo "========================================"

          (
            TASK_NAME="${task}"
            CLIP_MODE="${clip_mode}"
            LOG_GAUSS_SIGMA="${sigma}"
            SEED="${SEED:-${DEFAULT_SEED}}"
            CUDA_VISIBLE_DEVICES="${assigned_gpu}"
            NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
            NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
            NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
            TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
            CLIP_CONFIG="${clip_cfg_path}"
            REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
            REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
            TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
            LOG_BACKEND="${LOG_BACKEND:-${DEFAULT_LOG_BACKEND}}"
            LOG_FILE="${log_file}"
            JOB_PREFIX="[Job $((job_idx+1))] "
            JOB_IDX="${job_idx}"
            SWANLAB_DIR="${task_shared_swanlab}"
            TENSORBOARD_DIR="${task_shared_tensorboard}"
            export LOG_DIR SWANLAB_DIR TENSORBOARD_DIR

            run_training

            echo "[Job $((job_idx+1))] Completed: Task=${task}, Clip=${clip_mode}, Sigma=${sigma}"
          ) &

          last_pid=$!
          RUNNING_JOB_PIDS+=("${last_pid}")
          start_ts=$(date +%s)
          echo "${job_idx},${task},${clip_label},${last_pid},${log_file},${task_shared_swanlab},${task_shared_tensorboard},${start_ts}" >> "${META_FILE}"
          echo "[Job $((job_idx+1))] 任务已启动，PID=${last_pid}"
          job_idx=$((job_idx + 1))

          if [[ $job_idx -lt ${total_jobs} ]]; then
            echo ""
            echo "等待 ${STARTUP_DELAY}s，让任务完全启动后再启动下一个..."
            echo "当前运行任务数: $(jobs -r | wc -l)"
            sleep ${STARTUP_DELAY}
            echo ""
          fi
        done
      else
        # 等待有空闲的执行槽位
        wait_for_slot
        
        # 为当前任务分配 GPU（循环使用 GPU_CONFIGS）
        gpu_idx=$((job_idx % ${#GPU_CONFIGS[@]}))
        assigned_gpu="${GPU_CONFIGS[$gpu_idx]}"
        
        # 生成任务/配置专属目录与日志文件
        task_dir="${LOG_DIR}/${task}/${clip_mode}"
        mkdir -p "${task_dir}"
        log_file="${task_dir}/job_${job_idx}.log"
        
        echo "========================================"
        echo "[Job $((job_idx+1))/${total_jobs}] Starting: Task=${task}, Clip=${clip_mode}, GPU=${assigned_gpu}"
        echo "[Job $((job_idx+1))/${total_jobs}] Log file: ${log_file}"
        echo "========================================"
        
        # 在后台启动训练任务（使用子 shell 避免环境变量污染）
        (
          # 为 run_training 函数设置环境变量
          TASK_NAME="${task}"
          CLIP_MODE="${clip_mode}"
          SEED="${SEED:-${DEFAULT_SEED}}"
          CUDA_VISIBLE_DEVICES="${assigned_gpu}"  # 使用分配的 GPU
          NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
          NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
          NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
          TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
          CLIP_CONFIG="${CLIP_CONFIG:-${CLIP_CONFIG_PATH}}"
          REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
          REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
          TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
          LOG_BACKEND="${LOG_BACKEND:-${DEFAULT_LOG_BACKEND}}"
          LOG_FILE="${log_file}"
          JOB_PREFIX="[Job $((job_idx+1))] "
          JOB_IDX="${job_idx}"
          SWANLAB_DIR="${task_shared_swanlab}"
          TENSORBOARD_DIR="${task_shared_tensorboard}"
          export LOG_DIR SWANLAB_DIR TENSORBOARD_DIR
          
          run_training
          
          echo "[Job $((job_idx+1))] Completed: Task=${task}, Clip=${clip_mode}"
        ) &  # 在后台运行
        
        # 获取刚启动的任务的 PID
        last_pid=$!
        RUNNING_JOB_PIDS+=("${last_pid}")
        start_ts=$(date +%s)
        echo "${job_idx},${task},${clip_mode},${last_pid},${log_file},${task_shared_swanlab},${task_shared_tensorboard},${start_ts}" >> "${META_FILE}"
        echo "[Job $((job_idx+1))] 任务已启动，PID=${last_pid}"
        echo "[Job $((job_idx+1))] 当前后台任务列表："
        print_running_jobs

        job_idx=$((job_idx + 1))

        # 启动延迟：给每个任务足够的时间完成初始化
        # 重要：必须等待足够长的时间，让任务完全启动
        if [[ $job_idx -lt ${total_jobs} ]]; then
          echo ""
          echo "等待 ${STARTUP_DELAY}s，让任务完全启动后再启动下一个..."
          echo "当前运行任务数: $(jobs -r | wc -l)"
          sleep ${STARTUP_DELAY}
          echo ""
        fi
      fi
    done
  done
  
  # 等待并监控所有后台任务
  echo
  echo "所有任务已启动，监控中..."
  monitor_jobs "${LOG_DIR}" "${total_jobs}" "${expected_per_task}"
  wait || true

  # 重试失败的配置（顺序一次）
  echo
  echo "[重试] 开始顺序重试失败/卡死/离线日志缺失的配置..."
  rerun_failed_jobs "${LOG_DIR}"
  
  echo
  echo "===== Batch Experiment Run Finished ====="
  echo "所有日志保存在: ${LOG_DIR}/"
  echo "任务状态记录: ${LOG_DIR}/${STATUS_FILE_NAME}"
  echo "任务完成列表: ${LOG_DIR}/${TASK_DONE_FILE_NAME}"
  echo "SwanLab 离线日志索引: ${LOG_DIR}/${TASK_SWANLAB_LOGS_NAME}"
  
  # 显示最终GPU状态
  check_gpu_memory
elif [[ "${1:-}" == "batch-serial" ]]; then
  read -ra LOG_GAUSS_SIGMAS <<< "${LOG_GAUSS_SIGMAS:-${DEFAULT_LOG_GAUSS_SIGMAS}}"

  # 串行模式（保留原始的串行执行逻辑）
  echo "===== Starting Batch Experiment Run (Serial Mode) ====="
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      if [[ "${clip_mode}" == "log_gauss_clip" ]]; then
        for sigma in "${LOG_GAUSS_SIGMAS[@]}"; do
          TASK_NAME="${task}"
          CLIP_MODE="${clip_mode}"
          LOG_GAUSS_SIGMA="${sigma}"
          CLIP_CONFIG="$(make_log_gauss_clip_config "${sigma}")"
          SEED="${SEED:-${DEFAULT_SEED}}"
          CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
          NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
          NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
          NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
          TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
          REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
          REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
          TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
          LOG_BACKEND="${LOG_BACKEND:-${DEFAULT_LOG_BACKEND}}"
          run_training
        done
      else
        TASK_NAME="${task}"
        CLIP_MODE="${clip_mode}"
        SEED="${SEED:-${DEFAULT_SEED}}"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
        NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
        NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
        NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
        CLIP_CONFIG="${CLIP_CONFIG:-${CLIP_CONFIG_PATH}}"
        REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
        REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
        TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
        LOG_BACKEND="${LOG_BACKEND:-${DEFAULT_LOG_BACKEND}}"
        run_training
      fi
    done
  done
  echo "===== Batch Experiment Run Finished ====="
else
  # 否则，像以前一样运行单次实验
  echo "===== Starting Single Experiment Run ====="
  read -ra LOG_GAUSS_SIGMAS <<< "${LOG_GAUSS_SIGMAS:-${DEFAULT_LOG_GAUSS_SIGMAS}}"
  # 使用环境变量或默认值
  TASK_NAME="${TASK_NAME:-${DEFAULT_TASK_NAME}}"
  CLIP_MODE="${CLIP_MODE:-${DEFAULT_CLIP_MODE}}"
  SEED="${SEED:-${DEFAULT_SEED}}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
  NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
  NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
  NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
  CLIP_CONFIG="${CLIP_CONFIG:-${CLIP_CONFIG_PATH}}"
  if [[ "${CLIP_MODE}" == "log_gauss_clip" ]]; then
    LOG_GAUSS_SIGMA="${LOG_GAUSS_SIGMAS[0]}"  # 选第一个 sigma 作为单次运行默认
    CLIP_CONFIG="$(make_log_gauss_clip_config "${LOG_GAUSS_SIGMA}")"
  fi
  REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
  REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
  TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
  LOG_BACKEND="${LOG_BACKEND:-${DEFAULT_LOG_BACKEND}}"
  run_training
fi
