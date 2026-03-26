#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ACTION="${1:-start}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lcx-openvla-oft2}"

TASK_NAME="${TASK_NAME:-reach-v3}"
ROLLOUT_STEPS_PER_ITER="${ROLLOUT_STEPS_PER_ITER:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
BUFFER_HORIZON_STEPS="${BUFFER_HORIZON_STEPS:-20000}"
POLICY_LR="${POLICY_LR:-1e-4}"
VALUE_LR="${VALUE_LR:-1e-3}"
GAMMA="${GAMMA:-0.99}"
LAMBDA_VALUE="${LAMBDA_VALUE:-0.95}"
ENT_COEF="${ENT_COEF:-0.00}"
BASE_EXP_NAME="${BASE_EXP_NAME:-simple}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
SEEDS_STRING="${SEEDS:-22 64 99 234 360}"
CLIP_MODES_STRING="${CLIP_MODES:-ppo sapo gipo}"
GPU_IDS_STRING="${GPU_IDS:-6 7}"
GIPO_SIGMAS_STRING="${GIPO_SIGMAS:-0.1 0.2 0.5 1.0 2.0}"

SESSION_ROOT="${SESSION_ROOT:-logs/metaworld_ppo_discrete_simple_multi}"
RUNS_ROOT="${RUNS_ROOT:-runs/MetaWorldSimple/${TASK_NAME}/100k-stale-1e-4}"
LATEST_SESSION_FILE="${SESSION_ROOT}/latest_session.txt"

AUTO_TENSORBOARD="${AUTO_TENSORBOARD:-0}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-0.0.0.0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-}"
FORCE_STOP="${FORCE_STOP:-0}"

usage() {
  cat <<EOF
用法:
  bash $(basename "$0") start
  bash $(basename "$0") status
  bash $(basename "$0") stop

常用环境变量:
  CONDA_ENV_NAME=lcx-openvla-oft2
  SEEDS="142 23 64 450 99"
  GPU_IDS="4 5 6 7"
  GIPO_SIGMAS="0.2 0.5 1.0 2.0"
  AUTO_TENSORBOARD=1
  SESSION_DIR=logs/metaworld_ppo_discrete_simple_multi/<timestamp>
  FORCE_STOP=1
EOF
}

ensure_conda_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "未找到 conda，请先安装或手动激活 ${CONDA_ENV_NAME}。" >&2
    exit 1
  fi

  local conda_base
  conda_base="$(conda info --base 2>/dev/null)" || {
    echo "无法获取 conda base 路径。" >&2
    exit 1
  }

  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}" || {
    echo "无法激活 conda 环境 ${CONDA_ENV_NAME}。" >&2
    exit 1
  }
}

check_python_ready() {
  python - <<'PY'
from torch.utils.tensorboard import SummaryWriter
print("SummaryWriter check passed.")
PY
}

pick_free_port() {
  python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}

is_pid_running() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

resolve_session_dir() {
  if [[ -n "${SESSION_DIR:-}" ]]; then
    printf '%s\n' "${SESSION_DIR}"
    return 0
  fi

  if [[ -f "${LATEST_SESSION_FILE}" ]]; then
    local latest_session
    read -r latest_session < "${LATEST_SESSION_FILE}"
    if [[ -n "${latest_session}" ]]; then
      printf '%s\n' "${latest_session}"
      return 0
    fi
  fi

  echo "未找到会话目录，请先执行 start，或通过 SESSION_DIR 指定。" >&2
  exit 1
}

print_all_session_pids() {
  local session_dir="$1"

  if [[ -f "${session_dir}/all_pids.txt" ]]; then
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] && printf '%s\n' "${pid}"
    done < "${session_dir}/all_pids.txt"
    return 0
  fi

  local pid_file
  shopt -s nullglob
  for pid_file in "${session_dir}"/*.pid; do
    local pid
    read -r pid < "${pid_file}"
    [[ -n "${pid}" ]] && printf '%s\n' "${pid}"
  done
  shopt -u nullglob
}

launch_tensorboard() {
  local session_dir="$1"
  local all_pid_file="$2"

  if [[ "${AUTO_TENSORBOARD}" != "1" ]]; then
    return 0
  fi

  if ! command -v tensorboard >/dev/null 2>&1; then
    echo "警告: 当前环境未找到 tensorboard，跳过 TensorBoard 启动。" >&2
    return 0
  fi

  local tb_port="${TENSORBOARD_PORT}"
  if [[ -z "${tb_port}" ]]; then
    tb_port="$(pick_free_port)"
  fi

  local tb_log_file="${session_dir}/tensorboard.log"
  nohup tensorboard \
    --logdir "${RUNS_ROOT}" \
    --host "${TENSORBOARD_HOST}" \
    --port "${tb_port}" > "${tb_log_file}" 2>&1 &

  local tb_pid=$!
  printf '%s\n' "${tb_pid}" > "${session_dir}/tensorboard.pid"
  printf '%s\n' "${tb_port}" > "${session_dir}/tensorboard.port"
  printf '%s\n' "${RUNS_ROOT}" > "${session_dir}/runs_root.txt"
  printf '%s\n' "${tb_pid}" >> "${all_pid_file}"

  echo "[tensorboard] pid=${tb_pid} logdir=${RUNS_ROOT} port=${tb_port}"
}

start_jobs() {
  ensure_conda_env
  check_python_ready

  mkdir -p "${SESSION_ROOT}"

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"

  local session_dir="${SESSION_DIR:-${SESSION_ROOT}/${timestamp}}"
  mkdir -p "${session_dir}"
  printf '%s\n' "${session_dir}" > "${LATEST_SESSION_FILE}"

  local -a seed_array
  local -a clip_modes_array
  local -a gpu_ids_array
  local -a gipo_sigmas_array
  read -r -a seed_array <<< "${SEEDS_STRING}"
  read -r -a clip_modes_array <<< "${CLIP_MODES_STRING}"
  read -r -a gpu_ids_array <<< "${GPU_IDS_STRING}"
  read -r -a gipo_sigmas_array <<< "${GIPO_SIGMAS_STRING}"

  if [[ ${#seed_array[@]} -eq 0 ]]; then
    echo "SEEDS 不能为空。" >&2
    exit 1
  fi

  if [[ ${#clip_modes_array[@]} -eq 0 ]]; then
    echo "CLIP_MODES 不能为空。" >&2
    exit 1
  fi

  if [[ ${#gpu_ids_array[@]} -eq 0 ]]; then
    echo "GPU_IDS 不能为空。" >&2
    exit 1
  fi

  if [[ " ${clip_modes_array[*]} " == *" gipo "* ]] && [[ ${#gipo_sigmas_array[@]} -eq 0 ]]; then
    echo "启用 gipo 时，GIPO_SIGMAS 不能为空。" >&2
    exit 1
  fi

  local train_pid_file="${session_dir}/train_pids.tsv"
  local all_pid_file="${session_dir}/all_pids.txt"
  : > "${train_pid_file}"
  : > "${all_pid_file}"
  printf '%s\n' "${CONDA_DEFAULT_ENV:-unknown}" > "${session_dir}/conda_env.txt"
  printf '%s\n' "$(command -v python)" > "${session_dir}/python_path.txt"

  echo "Conda 环境: ${CONDA_DEFAULT_ENV:-unknown}"
  echo "Python: $(command -v python)"
  echo "任务名: ${TASK_NAME}"
  echo "clip-mode: ${clip_modes_array[*]}"
  echo "seed: ${seed_array[*]}"
  echo "GPU: ${gpu_ids_array[*]}"
  echo "gipo sigma: ${gipo_sigmas_array[*]}"
  echo "会话目录: ${session_dir}"
  echo "训练日志根目录: ${RUNS_ROOT}"
  echo

  launch_tensorboard "${session_dir}" "${all_pid_file}"

  local job_idx=0
  local launch_count=0
  for clip_mode in "${clip_modes_array[@]}"; do
    if [[ "${clip_mode}" == "gipo" ]]; then
      local sigma
      for sigma in "${gipo_sigmas_array[@]}"; do
        local sigma_tag="sigma${sigma//./p}"
        for seed in "${seed_array[@]}"; do
          local gpu_id="${gpu_ids_array[$((job_idx % ${#gpu_ids_array[@]}))]}"
          local run_name="${BASE_EXP_NAME}_${clip_mode}_${sigma_tag}_seed${seed}"
          local exp_name="${BASE_EXP_NAME}_seed${seed}_${clip_mode}_${sigma_tag}"
          local log_file="${session_dir}/${run_name}.log"
          local pid_file="${session_dir}/${run_name}.pid"

          local -a cmd=(
            env CUDA_VISIBLE_DEVICES="${gpu_id}"
            python rl/metaworld_ppo_discrete_simple.py
            --task-name "${TASK_NAME}"
            --rollout-steps-per-iter "${ROLLOUT_STEPS_PER_ITER}"
            --warmup-steps "${WARMUP_STEPS}"
            --train-batch-size "${TRAIN_BATCH_SIZE}"
            --buffer-horizon-steps "${BUFFER_HORIZON_STEPS}"
            --policy-lr "${POLICY_LR}"
            --value-lr "${VALUE_LR}"
            --gamma "${GAMMA}"
            --lambda "${LAMBDA_VALUE}"
            --ent-coef "${ENT_COEF}"
            --clip-mode "${clip_mode}"
            --sigma "${sigma}"
            --seed "${seed}"
            --exp-name "${exp_name}"
            --no-bf16
            --cuda-visible-devices "${gpu_id}"
            --train-iters "${TRAIN_ITERS}"
            --log-dir "${RUNS_ROOT}"
          )

          echo "[launch] clip_mode=${clip_mode} sigma=${sigma} seed=${seed} gpu=${gpu_id}"
          nohup "${cmd[@]}" > "${log_file}" 2>&1 &

          local pid=$!
          printf '%s\n' "${pid}" > "${pid_file}"
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${pid}" "${clip_mode}" "${seed}" "${sigma}" "${gpu_id}" "${log_file}" >> "${train_pid_file}"
          printf '%s\n' "${pid}" >> "${all_pid_file}"
          echo "         pid=${pid} log=${log_file}"

          job_idx=$((job_idx + 1))
          launch_count=$((launch_count + 1))
        done
      done
    else
      for seed in "${seed_array[@]}"; do
        local gpu_id="${gpu_ids_array[$((job_idx % ${#gpu_ids_array[@]}))]}"
        local run_name="${BASE_EXP_NAME}_${clip_mode}_seed${seed}"
        local exp_name="${BASE_EXP_NAME}_seed${seed}_${clip_mode}"
        local log_file="${session_dir}/${run_name}.log"
        local pid_file="${session_dir}/${run_name}.pid"

        local -a cmd=(
          env CUDA_VISIBLE_DEVICES="${gpu_id}"
          python rl/metaworld_ppo_discrete_simple.py
          --task-name "${TASK_NAME}"
          --rollout-steps-per-iter "${ROLLOUT_STEPS_PER_ITER}"
          --warmup-steps "${WARMUP_STEPS}"
          --train-batch-size "${TRAIN_BATCH_SIZE}"
          --buffer-horizon-steps "${BUFFER_HORIZON_STEPS}"
          --policy-lr "${POLICY_LR}"
          --value-lr "${VALUE_LR}"
          --gamma "${GAMMA}"
          --lambda "${LAMBDA_VALUE}"
          --ent-coef "${ENT_COEF}"
          --clip-mode "${clip_mode}"
          --seed "${seed}"
          --exp-name "${exp_name}"
          --no-bf16
          --cuda-visible-devices "${gpu_id}"
          --train-iters "${TRAIN_ITERS}"
          --log-dir "${RUNS_ROOT}"
        )

        echo "[launch] clip_mode=${clip_mode} seed=${seed} gpu=${gpu_id}"
        nohup "${cmd[@]}" > "${log_file}" 2>&1 &

        local pid=$!
        printf '%s\n' "${pid}" > "${pid_file}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${pid}" "${clip_mode}" "${seed}" "-" "${gpu_id}" "${log_file}" >> "${train_pid_file}"
        printf '%s\n' "${pid}" >> "${all_pid_file}"
        echo "         pid=${pid} log=${log_file}"

        job_idx=$((job_idx + 1))
        launch_count=$((launch_count + 1))
      done
    fi
  done

  echo
  echo "已启动 ${launch_count} 个训练进程。"
  echo "查看状态:"
  echo "  bash $(basename "$0") status"
  echo "停止全部:"
  echo "  bash $(basename "$0") stop"
  if [[ -f "${session_dir}/tensorboard.port" ]]; then
    local tb_port
    read -r tb_port < "${session_dir}/tensorboard.port"
    echo "TensorBoard 端口: ${tb_port}"
  fi
}

status_jobs() {
  local session_dir
  session_dir="$(resolve_session_dir)"

  local train_pid_file="${session_dir}/train_pids.tsv"
  echo "会话目录: ${session_dir}"

  if [[ -f "${session_dir}/tensorboard.pid" ]]; then
    local tb_pid
    read -r tb_pid < "${session_dir}/tensorboard.pid"
    if is_pid_running "${tb_pid}"; then
      if [[ -f "${session_dir}/tensorboard.port" ]]; then
        local tb_port
        read -r tb_port < "${session_dir}/tensorboard.port"
        echo "[RUNNING] tensorboard pid=${tb_pid} port=${tb_port}"
      else
        echo "[RUNNING] tensorboard pid=${tb_pid}"
      fi
    else
      echo "[EXITED] tensorboard pid=${tb_pid}"
    fi
  fi

  local running_count=0
  local exited_count=0
  if [[ -f "${train_pid_file}" ]]; then
    while IFS=$'\t' read -r pid clip_mode seed sigma gpu_id log_file; do
      [[ -z "${pid}" ]] && continue
      if [[ -z "${log_file}" ]]; then
        log_file="${gpu_id}"
        gpu_id="${sigma}"
        sigma="-"
      fi
      if is_pid_running "${pid}"; then
        if [[ "${sigma}" == "-" || -z "${sigma}" ]]; then
          echo "[RUNNING] pid=${pid} clip_mode=${clip_mode} seed=${seed} gpu=${gpu_id} log=${log_file}"
        else
          echo "[RUNNING] pid=${pid} clip_mode=${clip_mode} sigma=${sigma} seed=${seed} gpu=${gpu_id} log=${log_file}"
        fi
        running_count=$((running_count + 1))
      else
        if [[ "${sigma}" == "-" || -z "${sigma}" ]]; then
          echo "[EXITED] pid=${pid} clip_mode=${clip_mode} seed=${seed} gpu=${gpu_id} log=${log_file}"
        else
          echo "[EXITED] pid=${pid} clip_mode=${clip_mode} sigma=${sigma} seed=${seed} gpu=${gpu_id} log=${log_file}"
        fi
        exited_count=$((exited_count + 1))
      fi
    done < "${train_pid_file}"
  else
    local pid_file
    shopt -s nullglob
    for pid_file in "${session_dir}"/*.pid; do
      local pid
      local run_name
      local log_file
      local base_name
      base_name="$(basename "${pid_file}")"
      if [[ "${base_name}" == "tensorboard.pid" ]]; then
        continue
      fi

      read -r pid < "${pid_file}"
      run_name="${base_name%.pid}"
      log_file="${session_dir}/${run_name}.log"
      if is_pid_running "${pid}"; then
        echo "[RUNNING] pid=${pid} run=${run_name} log=${log_file}"
        running_count=$((running_count + 1))
      else
        echo "[EXITED] pid=${pid} run=${run_name} log=${log_file}"
        exited_count=$((exited_count + 1))
      fi
    done
    shopt -u nullglob
  fi

  echo "汇总: running=${running_count} exited=${exited_count}"
}

stop_jobs() {
  local session_dir
  session_dir="$(resolve_session_dir)"

  local stopped_count=0
  local skipped_count=0
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    if is_pid_running "${pid}"; then
      kill "${pid}"
      echo "[STOP] pid=${pid}"
      stopped_count=$((stopped_count + 1))
    else
      echo "[SKIP] pid=${pid} 已退出"
      skipped_count=$((skipped_count + 1))
    fi
  done < <(print_all_session_pids "${session_dir}")

  if [[ "${FORCE_STOP}" == "1" ]]; then
    sleep 2
    while IFS= read -r pid; do
      [[ -z "${pid}" ]] && continue
      if is_pid_running "${pid}"; then
        kill -9 "${pid}" 2>/dev/null || true
        echo "[KILL] pid=${pid}"
      fi
    done < <(print_all_session_pids "${session_dir}")
  fi

  echo "已发送停止信号: ${stopped_count} 个；已退出: ${skipped_count} 个。"
}

case "${ACTION}" in
  start)
    start_jobs
    ;;
  status)
    status_jobs
    ;;
  stop)
    stop_jobs
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "未知动作: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac
