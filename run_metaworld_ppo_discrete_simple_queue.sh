#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ACTION="${1:-start}"

USER_SET_MAX_PARALLEL="${MAX_PARALLEL+x}"
USER_MAX_PARALLEL_VALUE="${MAX_PARALLEL-}"
USER_SET_GPU_IDS="${GPU_IDS+x}"
USER_GPU_IDS_VALUE="${GPU_IDS-}"
USER_SET_POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS+x}"
USER_POLL_INTERVAL_SECS_VALUE="${POLL_INTERVAL_SECS-}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lcx-openvla-oft2}"
# assembly-v3 basketball-v3 bin-picking-v3 box-close-v3 button-press-topdown-v3 button-press-topdown-wall-v3 button-press-v3 button-press-wall-v3 coffee-button-v3 coffee-pull-v3 coffee-push-v3 dial-turn-v3 disassemble-v3 door-close-v3 door-lock-v3 door-open-v3 door-unlock-v3 drawer-close-v3 drawer-open-v3 faucet-close-v3 faucet-open-v3 hammer-v3 hand-insert-v3 handle-press-side-v3 handle-press-v3 handle-pull-side-v3 handle-pull-v3 lever-pull-v3 peg-insert-side-v3 peg-unplug-side-v3 pick-out-of-hole-v3 pick-place-v3 pick-place-wall-v3 plate-slide-back-side-v3 plate-slide-back-v3 plate-slide-side-v3 plate-slide-v3 push-back-v3 push-v3 push-wall-v3 reach-v3 reach-wall-v3 shelf-place-v3 soccer-v3 stick-pull-v3 stick-push-v3 sweep-into-v3 sweep-v3 window-close-v3 window-open-v3
TASKS_STRING="${TASKS:-assembly-v3 basketball-v3 bin-picking-v3 box-close-v3 button-press-topdown-v3 button-press-topdown-wall-v3 button-press-v3 button-press-wall-v3 coffee-button-v3 coffee-pull-v3}"
ROLLOUT_STEPS_PER_ITER="${ROLLOUT_STEPS_PER_ITER:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
SAMPLE_ROUNDS="${SAMPLE_ROUNDS:-5}"
REUSE_PER_BATCH="${REUSE_PER_BATCH:-2}"
ACTOR_EVERY="${ACTOR_EVERY:-2}"
BUFFER_HORIZON_STEPS="${BUFFER_HORIZON_STEPS:-20000}"
POLICY_LR="${POLICY_LR:-3e-4}"
VALUE_LR="${VALUE_LR:-3e-3}"
GAMMA="${GAMMA:-0.99}"
LAMBDA_VALUE="${LAMBDA_VALUE:-0.95}"
ENT_COEF="${ENT_COEF:-0.00}"
REWARD_SCALE="${REWARD_SCALE:-0.001}"
BASE_EXP_NAME="${BASE_EXP_NAME:-fresh}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
SEEDS_STRING="${SEEDS:-22 64 99}"
CLIP_MODES_STRING="${CLIP_MODES:-ppo sapo gipo}"
GPU_IDS_STRING="${GPU_IDS:-4 5 6 7}"
GIPO_SIGMAS_STRING="${GIPO_SIGMAS:-0.2 0.5 1.0}"
GIPO_SIGMA_NEG_RATIOS_STRING="${GIPO_SIGMA_NEG_RATIOS:-0.5 1.0}"

RUNS_BASE_ROOT="${RUNS_BASE_ROOT:-runs/MetaWorldSimple}"
RUN_GROUP_NAME="${RUN_GROUP_NAME:-10k-stale-sample5-reuse2-actor2}"
SESSION_ROOT="${SESSION_ROOT:-logs/metaworld_ppo_discrete_simple_queue}"
LATEST_SESSION_FILE="${SESSION_ROOT}/latest_session.txt"

MAX_PARALLEL="${MAX_PARALLEL:-80}"
POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-30}"
COMPLETION_RATIO="${COMPLETION_RATIO:-0.95}"
FORCE_STOP="${FORCE_STOP:-0}"
DRY_RUN="${DRY_RUN:-0}"

STATUS_HELPER_PATH="${SCRIPT_DIR}/rl/plot/inspect_metaworld_simple_run_status.py"

CURRENT_SESSION_DIR=""
JOB_TOTAL_COUNT=0
TOTAL_COUNT=0
DONE_COUNT=0
RUNNING_COUNT=0
STALE_COUNT=0
MISSING_COUNT=0

declare -a task_array=()
declare -a seed_array=()
declare -a clip_modes_array=()
declare -a gpu_ids_array=()
declare -a gipo_sigmas_array=()
declare -a gipo_sigma_neg_ratios_array=()
declare -a used_gpu_ids_array=()

usage() {
  cat <<EOF
用法:
  bash $(basename "$0") start
  bash $(basename "$0") resume
  bash $(basename "$0") status
  bash $(basename "$0") stop

常用环境变量:
  TASKS="assembly-v3 basketball-v3 reach-v3"
  SEEDS="22 64 99 234 360"
  CLIP_MODES="ppo sapo gipo"
  GIPO_SIGMAS="0.1 0.2 0.5 1.0 2.0"
  GIPO_SIGMA_NEG_RATIOS="0.5 1.0"
  GPU_IDS="0 1 2 3"
  MAX_PARALLEL=8
  POLL_INTERVAL_SECS=30
  COMPLETION_RATIO=0.95
  SESSION_DIR=logs/metaworld_ppo_discrete_simple_queue/<timestamp>
  RUNS_BASE_ROOT=runs/MetaWorldSimple
  RUN_GROUP_NAME=1k-stale-sample10-reuse10-actor10-3e-4
  # 支持 MAX_PARALLEL > GPU_IDS 数量，单卡可并发多个训练进程
  DRY_RUN=1
  FORCE_STOP=1
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
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
from tensorboard.compat.proto import event_pb2  # noqa: F401
from torch.utils.tensorboard import SummaryWriter  # noqa: F401

print("python dependency check passed.")
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

  echo "未找到会话目录，请先执行 start/resume，或通过 SESSION_DIR 指定。" >&2
  exit 1
}

materialize_arrays() {
  read -r -a task_array <<< "${TASKS_STRING}"
  read -r -a seed_array <<< "${SEEDS_STRING}"
  read -r -a clip_modes_array <<< "${CLIP_MODES_STRING}"
  read -r -a gpu_ids_array <<< "${GPU_IDS_STRING}"
  read -r -a gipo_sigmas_array <<< "${GIPO_SIGMAS_STRING}"
  read -r -a gipo_sigma_neg_ratios_array <<< "${GIPO_SIGMA_NEG_RATIOS_STRING}"
}

validate_runtime_config() {
  materialize_arrays

  if [[ ${#task_array[@]} -eq 0 ]]; then
    echo "TASKS 不能为空。" >&2
    exit 1
  fi
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
  if [[ " ${clip_modes_array[*]} " == *" gipo "* ]] && [[ ${#gipo_sigma_neg_ratios_array[@]} -eq 0 ]]; then
    echo "启用 gipo 时，GIPO_SIGMA_NEG_RATIOS 不能为空。" >&2
    exit 1
  fi

  if [[ -z "${MAX_PARALLEL}" ]]; then
    MAX_PARALLEL="${#gpu_ids_array[@]}"
  fi
  if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL}" == "0" ]]; then
    echo "MAX_PARALLEL 必须是正整数。" >&2
    exit 1
  fi
  if ! [[ "${POLL_INTERVAL_SECS}" =~ ^[0-9]+$ ]] || [[ "${POLL_INTERVAL_SECS}" == "0" ]]; then
    echo "POLL_INTERVAL_SECS 必须是正整数。" >&2
    exit 1
  fi
}

write_session_env() {
  local session_dir="$1"
  local session_env="${session_dir}/session.env"
  {
    printf 'CONDA_ENV_NAME=%q\n' "${CONDA_ENV_NAME}"
    printf 'TASKS_STRING=%q\n' "${TASKS_STRING}"
    printf 'ROLLOUT_STEPS_PER_ITER=%q\n' "${ROLLOUT_STEPS_PER_ITER}"
    printf 'WARMUP_STEPS=%q\n' "${WARMUP_STEPS}"
    printf 'TRAIN_BATCH_SIZE=%q\n' "${TRAIN_BATCH_SIZE}"
    printf 'SAMPLE_ROUNDS=%q\n' "${SAMPLE_ROUNDS}"
    printf 'REUSE_PER_BATCH=%q\n' "${REUSE_PER_BATCH}"
    printf 'ACTOR_EVERY=%q\n' "${ACTOR_EVERY}"
    printf 'BUFFER_HORIZON_STEPS=%q\n' "${BUFFER_HORIZON_STEPS}"
    printf 'POLICY_LR=%q\n' "${POLICY_LR}"
    printf 'VALUE_LR=%q\n' "${VALUE_LR}"
    printf 'GAMMA=%q\n' "${GAMMA}"
    printf 'LAMBDA_VALUE=%q\n' "${LAMBDA_VALUE}"
    printf 'ENT_COEF=%q\n' "${ENT_COEF}"
    printf 'REWARD_SCALE=%q\n' "${REWARD_SCALE}"
    printf 'BASE_EXP_NAME=%q\n' "${BASE_EXP_NAME}"
    printf 'TRAIN_ITERS=%q\n' "${TRAIN_ITERS}"
    printf 'SEEDS_STRING=%q\n' "${SEEDS_STRING}"
    printf 'CLIP_MODES_STRING=%q\n' "${CLIP_MODES_STRING}"
    printf 'GPU_IDS_STRING=%q\n' "${GPU_IDS_STRING}"
    printf 'GIPO_SIGMAS_STRING=%q\n' "${GIPO_SIGMAS_STRING}"
    printf 'GIPO_SIGMA_NEG_RATIOS_STRING=%q\n' "${GIPO_SIGMA_NEG_RATIOS_STRING}"
    printf 'RUNS_BASE_ROOT=%q\n' "${RUNS_BASE_ROOT}"
    printf 'RUN_GROUP_NAME=%q\n' "${RUN_GROUP_NAME}"
    printf 'MAX_PARALLEL=%q\n' "${MAX_PARALLEL}"
    printf 'POLL_INTERVAL_SECS=%q\n' "${POLL_INTERVAL_SECS}"
    printf 'COMPLETION_RATIO=%q\n' "${COMPLETION_RATIO}"
  } > "${session_env}"
}

load_session_env() {
  local session_dir="$1"
  local session_env="${session_dir}/session.env"
  if [[ ! -f "${session_env}" ]]; then
    echo "会话缺少 session.env: ${session_env}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${session_env}"

  if [[ "${USER_SET_GPU_IDS}" == "x" ]]; then
    GPU_IDS_STRING="${USER_GPU_IDS_VALUE}"
  fi
  if [[ "${USER_SET_MAX_PARALLEL}" == "x" ]]; then
    MAX_PARALLEL="${USER_MAX_PARALLEL_VALUE}"
  fi
  if [[ "${USER_SET_POLL_INTERVAL_SECS}" == "x" ]]; then
    POLL_INTERVAL_SECS="${USER_POLL_INTERVAL_SECS_VALUE}"
  fi

  validate_runtime_config
}

ensure_session_runtime_dirs() {
  local session_dir="$1"
  mkdir -p "${session_dir}"
  mkdir -p "${session_dir}/logs"
  mkdir -p "${session_dir}/pids"
  mkdir -p "${session_dir}/meta"
  touch "${session_dir}/all_pids.txt"
  if [[ ! -f "${session_dir}/launch_history.tsv" ]]; then
    printf 'timestamp\tjob_key\ttask_name\tclip_mode\tsigma\tsigma_neg_ratio\tseed\tgpu_id\tpid\tlog_file\n' > "${session_dir}/launch_history.tsv"
  fi
}

create_jobs_manifest() {
  local session_dir="$1"
  local jobs_file="${session_dir}/jobs.tsv"
  JOB_TOTAL_COUNT=0

  printf 'job_index\tjob_key\ttask_name\tclip_mode\tsigma\tsigma_neg_ratio\tseed\texp_name\ttask_runs_root\ttrain_iters\n' > "${jobs_file}"

  local task_name clip_mode seed sigma sigma_tag sigma_neg_ratio sigma_neg_ratio_tag job_key task_runs_root
  for task_name in "${task_array[@]}"; do
    task_runs_root="${RUNS_BASE_ROOT}/${task_name}/${RUN_GROUP_NAME}"
    mkdir -p "${task_runs_root}"
    for clip_mode in "${clip_modes_array[@]}"; do
      if [[ "${clip_mode}" == "gipo" ]]; then
        for sigma in "${gipo_sigmas_array[@]}"; do
          sigma_tag="sigma${sigma//./p}"
          for sigma_neg_ratio in "${gipo_sigma_neg_ratios_array[@]}"; do
            sigma_neg_ratio_tag="neg${sigma_neg_ratio//./p}"
            for seed in "${seed_array[@]}"; do
              JOB_TOTAL_COUNT=$((JOB_TOTAL_COUNT + 1))
              job_key="${BASE_EXP_NAME}_${task_name}_seed${seed}_${clip_mode}_${sigma_tag}_${sigma_neg_ratio_tag}"
              printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${JOB_TOTAL_COUNT}" "${job_key}" "${task_name}" "${clip_mode}" "${sigma}" "${sigma_neg_ratio}" "${seed}" \
                "${job_key}" "${task_runs_root}" "${TRAIN_ITERS}" >> "${jobs_file}"
            done
          done
        done
      else
        for seed in "${seed_array[@]}"; do
          JOB_TOTAL_COUNT=$((JOB_TOTAL_COUNT + 1))
          job_key="${BASE_EXP_NAME}_${task_name}_seed${seed}_${clip_mode}"
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${JOB_TOTAL_COUNT}" "${job_key}" "${task_name}" "${clip_mode}" "-" "-" "${seed}" \
            "${job_key}" "${task_runs_root}" "${TRAIN_ITERS}" >> "${jobs_file}"
        done
      fi
    done
  done
}

write_summary_json() {
  local session_dir="$1"
  local summary_file="${session_dir}/summary.json"
  local conda_env_value="${CONDA_DEFAULT_ENV:-unknown}"
  python - <<PY
import json

summary = {
    "session_dir": ${session_dir@Q},
    "runs_base_root": ${RUNS_BASE_ROOT@Q},
    "run_group_name": ${RUN_GROUP_NAME@Q},
    "tasks": ${TASKS_STRING@Q}.split(),
    "seeds": [int(value) for value in ${SEEDS_STRING@Q}.split()],
    "clip_modes": ${CLIP_MODES_STRING@Q}.split(),
    "gpu_ids": ${GPU_IDS_STRING@Q}.split(),
    "gipo_sigmas": [float(value) for value in ${GIPO_SIGMAS_STRING@Q}.split()] if ${GIPO_SIGMAS_STRING@Q}.strip() else [],
    "gipo_sigma_neg_ratios": [float(value) for value in ${GIPO_SIGMA_NEG_RATIOS_STRING@Q}.split()] if ${GIPO_SIGMA_NEG_RATIOS_STRING@Q}.strip() else [],
    "train_iters": int(${TRAIN_ITERS@Q}),
    "max_parallel": int(${MAX_PARALLEL@Q}),
    "poll_interval_secs": int(${POLL_INTERVAL_SECS@Q}),
    "completion_ratio": float(${COMPLETION_RATIO@Q}),
    "total_jobs": int(${JOB_TOTAL_COUNT}),
    "base_exp_name": ${BASE_EXP_NAME@Q},
    "conda_env": ${conda_env_value@Q},
}

with open(${summary_file@Q}, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)
PY
}

prepare_new_session() {
  mkdir -p "${SESSION_ROOT}"

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  CURRENT_SESSION_DIR="${SESSION_DIR:-${SESSION_ROOT}/${timestamp}}"

  if [[ -f "${CURRENT_SESSION_DIR}/jobs.tsv" ]]; then
    echo "会话已存在: ${CURRENT_SESSION_DIR}。如需续跑，请使用 resume。" >&2
    exit 1
  fi

  ensure_session_runtime_dirs "${CURRENT_SESSION_DIR}"
  printf '%s\n' "${CURRENT_SESSION_DIR}" > "${LATEST_SESSION_FILE}"
  write_session_env "${CURRENT_SESSION_DIR}"
  create_jobs_manifest "${CURRENT_SESSION_DIR}"
  write_summary_json "${CURRENT_SESSION_DIR}"
}

prepare_existing_session() {
  CURRENT_SESSION_DIR="$(resolve_session_dir)"
  load_session_env "${CURRENT_SESSION_DIR}"
  ensure_session_runtime_dirs "${CURRENT_SESSION_DIR}"
  if [[ ! -f "${CURRENT_SESSION_DIR}/jobs.tsv" ]]; then
    echo "会话缺少 jobs.tsv: ${CURRENT_SESSION_DIR}" >&2
    exit 1
  fi
  printf '%s\n' "${CURRENT_SESSION_DIR}" > "${LATEST_SESSION_FILE}"
}

register_supervisor() {
  printf '%s\n' "$$" > "${CURRENT_SESSION_DIR}/supervisor.pid"
}

clear_supervisor_file() {
  local supervisor_file="${CURRENT_SESSION_DIR}/supervisor.pid"
  if [[ -f "${supervisor_file}" ]]; then
    local recorded_pid=""
    read -r recorded_pid < "${supervisor_file}" || true
    if [[ "${recorded_pid}" == "$$" ]]; then
      rm -f "${supervisor_file}"
    fi
  fi
}

on_supervisor_interrupt() {
  log "收到中断信号，调度器退出。已启动的训练进程会继续后台运行，可稍后执行 resume。"
  exit 130
}

generate_status_snapshot() {
  local status_file="${CURRENT_SESSION_DIR}/status_snapshot.tsv"
  python "${STATUS_HELPER_PATH}" \
    --jobs-file "${CURRENT_SESSION_DIR}/jobs.tsv" \
    --session-dir "${CURRENT_SESSION_DIR}" \
    --completion-ratio "${COMPLETION_RATIO}" \
    --output-format tsv > "${status_file}"
}

parse_status_snapshot() {
  local status_file="${CURRENT_SESSION_DIR}/status_snapshot.tsv"
  TOTAL_COUNT=0
  DONE_COUNT=0
  RUNNING_COUNT=0
  STALE_COUNT=0
  MISSING_COUNT=0
  used_gpu_ids_array=()

  local job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file
  while IFS=$'\t' read -r job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file; do
    [[ "${job_index}" == "job_index" ]] && continue
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    case "${status}" in
      done)
        DONE_COUNT=$((DONE_COUNT + 1))
        ;;
      running)
        RUNNING_COUNT=$((RUNNING_COUNT + 1))
        if [[ -n "${gpu_id}" && "${gpu_id}" != "-" ]]; then
          used_gpu_ids_array+=("${gpu_id}")
        fi
        ;;
      stale)
        STALE_COUNT=$((STALE_COUNT + 1))
        ;;
      missing)
        MISSING_COUNT=$((MISSING_COUNT + 1))
        ;;
    esac
  done < "${status_file}"
}

print_progress_summary() {
  parse_status_snapshot
  local pending_count=$((TOTAL_COUNT - DONE_COUNT - RUNNING_COUNT))
  log "进度: completed=${DONE_COUNT}/${TOTAL_COUNT} running=${RUNNING_COUNT} pending=${pending_count} stale=${STALE_COUNT} missing=${MISSING_COUNT}"
}

cleanup_stale_runs() {
  local status_file="${CURRENT_SESSION_DIR}/status_snapshot.tsv"
  local deleted_count=0
  local job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file
  while IFS=$'\t' read -r job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file; do
    [[ "${job_index}" == "job_index" ]] && continue
    if [[ "${status}" != "stale" ]] || [[ -z "${stale_run_dirs}" ]] || [[ "${stale_run_dirs}" == "-" ]]; then
      continue
    fi

    local stale_dir
    IFS='|' read -r -a stale_dir_array <<< "${stale_run_dirs}"
    for stale_dir in "${stale_dir_array[@]}"; do
      [[ -z "${stale_dir}" ]] && continue
      if [[ -d "${stale_dir}" ]]; then
        rm -rf -- "${stale_dir}"
        deleted_count=$((deleted_count + 1))
        log "[cleanup] 删除未完成 run: ${stale_dir}"
      fi
    done
  done < "${status_file}"

  if (( deleted_count > 0 )); then
    rm -f "${CURRENT_SESSION_DIR}/status_snapshot.tsv"
  fi
}

append_launch_history() {
  local timestamp="$1"
  local job_key="$2"
  local task_name="$3"
  local clip_mode="$4"
  local sigma="$5"
  local sigma_neg_ratio="$6"
  local seed="$7"
  local gpu_id="$8"
  local pid="$9"
  local log_file="${10}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${timestamp}" "${job_key}" "${task_name}" "${clip_mode}" "${sigma}" "${sigma_neg_ratio}" "${seed}" "${gpu_id}" "${pid}" "${log_file}" \
    >> "${CURRENT_SESSION_DIR}/launch_history.tsv"
}

write_meta_file() {
  local meta_file="$1"
  local job_key="$2"
  local task_name="$3"
  local clip_mode="$4"
  local sigma="$5"
  local sigma_neg_ratio="$6"
  local seed="$7"
  local gpu_id="$8"
  local pid="$9"
  local log_file="${10}"
  local launched_at="${11}"
  python - <<PY
import json

payload = {
    "job_key": ${job_key@Q},
    "task_name": ${task_name@Q},
    "clip_mode": ${clip_mode@Q},
    "sigma": None if ${sigma@Q} == "-" else ${sigma@Q},
    "sigma_neg_ratio": None if ${sigma_neg_ratio@Q} == "-" else ${sigma_neg_ratio@Q},
    "seed": int(${seed@Q}),
    "gpu_id": ${gpu_id@Q},
    "pid": int(${pid@Q}),
    "log_file": ${log_file@Q},
    "launched_at": ${launched_at@Q},
}

with open(${meta_file@Q}, "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2, ensure_ascii=False)
PY
}

launch_job() {
  local job_key="$1"
  local task_name="$2"
  local clip_mode="$3"
  local sigma="$4"
  local sigma_neg_ratio="$5"
  local seed="$6"
  local gpu_id="$7"

  local task_runs_root="${RUNS_BASE_ROOT}/${task_name}/${RUN_GROUP_NAME}"
  local log_file="${CURRENT_SESSION_DIR}/logs/${job_key}.log"
  local pid_file="${CURRENT_SESSION_DIR}/pids/${job_key}.pid"
  local meta_file="${CURRENT_SESSION_DIR}/meta/${job_key}.json"
  local launched_at
  launched_at="$(date '+%Y-%m-%d %H:%M:%S')"

  mkdir -p "${task_runs_root}"

  local -a cmd=(
    env CUDA_VISIBLE_DEVICES="${gpu_id}"
    python rl/metaworld_ppo_discrete_simple.py
    --task-name "${task_name}"
    --rollout-steps-per-iter "${ROLLOUT_STEPS_PER_ITER}"
    --warmup-steps "${WARMUP_STEPS}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --sample-rounds "${SAMPLE_ROUNDS}"
    --reuse-per-batch "${REUSE_PER_BATCH}"
    --actor-every "${ACTOR_EVERY}"
    --buffer-horizon-steps "${BUFFER_HORIZON_STEPS}"
    --policy-lr "${POLICY_LR}"
    --value-lr "${VALUE_LR}"
    --gamma "${GAMMA}"
    --lambda "${LAMBDA_VALUE}"
    --ent-coef "${ENT_COEF}"
    --clip-mode "${clip_mode}"
    --seed "${seed}"
    --exp-name "${job_key}"
    --no-bf16
    --cuda-visible-devices "${gpu_id}"
    --train-iters "${TRAIN_ITERS}"
    --log-dir "${task_runs_root}"
    --reward-scale "${REWARD_SCALE}"
  )

  if [[ "${clip_mode}" == "gipo" ]]; then
    cmd+=( --sigma "${sigma}" --sigma-neg-ratio "${sigma_neg_ratio}" )
  fi

  log "[launch] task=${task_name} clip_mode=${clip_mode} sigma=${sigma} sigma_neg_ratio=${sigma_neg_ratio} seed=${seed} gpu=${gpu_id}"
  nohup "${cmd[@]}" > "${log_file}" 2>&1 &

  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf '%s\n' "${pid}" >> "${CURRENT_SESSION_DIR}/all_pids.txt"
  append_launch_history "${launched_at}" "${job_key}" "${task_name}" "${clip_mode}" "${sigma}" "${sigma_neg_ratio}" "${seed}" "${gpu_id}" "${pid}" "${log_file}"
  write_meta_file "${meta_file}" "${job_key}" "${task_name}" "${clip_mode}" "${sigma}" "${sigma_neg_ratio}" "${seed}" "${gpu_id}" "${pid}" "${log_file}" "${launched_at}"
  log "[launch] pid=${pid} log=${log_file}"
}

launch_missing_jobs() {
  parse_status_snapshot

  local available_slots=$((MAX_PARALLEL - RUNNING_COUNT))
  if (( available_slots <= 0 )); then
    return 0
  fi

  local -A gpu_loads=()
  local gpu_id
  for gpu_id in "${gpu_ids_array[@]}"; do
    gpu_loads["${gpu_id}"]=0
  done

  for gpu_id in "${used_gpu_ids_array[@]}"; do
    if [[ -n "${gpu_loads[${gpu_id}]+x}" ]]; then
      gpu_loads["${gpu_id}"]=$((gpu_loads["${gpu_id}"] + 1))
    fi
  done

  local launched_count=0
  local status_file="${CURRENT_SESSION_DIR}/status_snapshot.tsv"
  local job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive current_gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file
  while IFS=$'\t' read -r job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive current_gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file; do
    [[ "${job_index}" == "job_index" ]] && continue
    [[ "${status}" != "missing" ]] && continue

    if (( launched_count >= available_slots )); then
      break
    fi

    local selected_gpu=""
    local selected_load=-1
    local candidate_gpu
    local candidate_load
    for candidate_gpu in "${gpu_ids_array[@]}"; do
      candidate_load="${gpu_loads[${candidate_gpu}]}"
      if [[ -z "${selected_gpu}" ]] || (( candidate_load < selected_load )); then
        selected_gpu="${candidate_gpu}"
        selected_load="${candidate_load}"
      fi
    done

    if [[ -z "${selected_gpu}" ]]; then
      break
    fi

    launch_job "${job_key}" "${task_name}" "${clip_mode}" "${sigma}" "${sigma_neg_ratio}" "${seed}" "${selected_gpu}"
    gpu_loads["${selected_gpu}"]=$((gpu_loads["${selected_gpu}"] + 1))
    launched_count=$((launched_count + 1))
  done < "${status_file}"
}

run_queue_loop() {
  trap clear_supervisor_file EXIT
  trap on_supervisor_interrupt INT TERM
  register_supervisor

  while true; do
    generate_status_snapshot
    cleanup_stale_runs
    if [[ ! -f "${CURRENT_SESSION_DIR}/status_snapshot.tsv" ]]; then
      generate_status_snapshot
    fi

    print_progress_summary
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "DRY_RUN=1，仅生成/读取 manifest 与当前状态，不实际启动实验。"
      break
    fi

    if (( DONE_COUNT == TOTAL_COUNT )); then
      log "全部实验已完成，共 ${TOTAL_COUNT} 个。"
      break
    fi

    launch_missing_jobs

    generate_status_snapshot
    print_progress_summary
    if (( DONE_COUNT == TOTAL_COUNT )); then
      log "全部实验已完成，共 ${TOTAL_COUNT} 个。"
      break
    fi

    sleep "${POLL_INTERVAL_SECS}"
  done
}

print_supervisor_status() {
  local session_dir="$1"
  local supervisor_file="${session_dir}/supervisor.pid"
  if [[ -f "${supervisor_file}" ]]; then
    local supervisor_pid
    read -r supervisor_pid < "${supervisor_file}"
    if is_pid_running "${supervisor_pid}"; then
      echo "[RUNNING] supervisor pid=${supervisor_pid}"
    else
      echo "[EXITED] supervisor pid=${supervisor_pid}"
    fi
  else
    echo "[EXITED] supervisor pid=none"
  fi
}

print_status_details() {
  local status_file="${CURRENT_SESSION_DIR}/status_snapshot.tsv"
  local job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file
  while IFS=$'\t' read -r job_index job_key task_name clip_mode sigma sigma_neg_ratio seed status pid pid_alive gpu_id last_iter target_iters progress_ratio done_reason run_dir stale_run_dirs log_file; do
    [[ "${job_index}" == "job_index" ]] && continue
    local status_tag="[${status^^}]"
    local progress_text="${last_iter}/${target_iters}"
    if [[ "${sigma}" == "-" || -z "${sigma}" ]]; then
      echo "${status_tag} #${job_index} task=${task_name} clip_mode=${clip_mode} seed=${seed} progress=${progress_text} gpu=${gpu_id:--} pid=${pid:--} log=${log_file:--}"
    elif [[ "${sigma_neg_ratio}" == "-" || -z "${sigma_neg_ratio}" ]]; then
      echo "${status_tag} #${job_index} task=${task_name} clip_mode=${clip_mode} sigma=${sigma} seed=${seed} progress=${progress_text} gpu=${gpu_id:--} pid=${pid:--} log=${log_file:--}"
    else
      echo "${status_tag} #${job_index} task=${task_name} clip_mode=${clip_mode} sigma=${sigma} sigma_neg_ratio=${sigma_neg_ratio} seed=${seed} progress=${progress_text} gpu=${gpu_id:--} pid=${pid:--} log=${log_file:--}"
    fi
    if [[ -n "${run_dir}" && "${run_dir}" != "-" ]]; then
      echo "          run_dir=${run_dir}"
    fi
  done < "${status_file}"
}

start_action() {
  ensure_conda_env
  check_python_ready
  validate_runtime_config
  prepare_new_session

  log "会话目录: ${CURRENT_SESSION_DIR}"
  log "任务列表: ${TASKS_STRING}"
  log "总实验数: ${JOB_TOTAL_COUNT}"
  log "并发上限: ${MAX_PARALLEL}"
  run_queue_loop
}

resume_action() {
  ensure_conda_env
  check_python_ready
  prepare_existing_session

  log "续跑会话: ${CURRENT_SESSION_DIR}"
  run_queue_loop
}

status_action() {
  ensure_conda_env
  check_python_ready
  prepare_existing_session
  echo "会话目录: ${CURRENT_SESSION_DIR}"
  print_supervisor_status "${CURRENT_SESSION_DIR}"
  generate_status_snapshot
  print_progress_summary
  print_status_details
}

stop_action() {
  CURRENT_SESSION_DIR="$(resolve_session_dir)"
  echo "会话目录: ${CURRENT_SESSION_DIR}"

  local stopped_count=0
  local skipped_count=0

  local supervisor_file="${CURRENT_SESSION_DIR}/supervisor.pid"
  if [[ -f "${supervisor_file}" ]]; then
    local supervisor_pid
    read -r supervisor_pid < "${supervisor_file}"
    if is_pid_running "${supervisor_pid}"; then
      kill "${supervisor_pid}"
      echo "[STOP] supervisor pid=${supervisor_pid}"
      stopped_count=$((stopped_count + 1))
    else
      echo "[SKIP] supervisor pid=${supervisor_pid} 已退出"
      skipped_count=$((skipped_count + 1))
    fi
  fi

  shopt -s nullglob
  local pid_file
  for pid_file in "${CURRENT_SESSION_DIR}"/pids/*.pid; do
    local pid
    read -r pid < "${pid_file}"
    if is_pid_running "${pid}"; then
      kill "${pid}"
      echo "[STOP] pid=${pid} file=$(basename "${pid_file}")"
      stopped_count=$((stopped_count + 1))
    else
      echo "[SKIP] pid=${pid} file=$(basename "${pid_file}") 已退出"
      skipped_count=$((skipped_count + 1))
    fi
  done
  shopt -u nullglob

  if [[ "${FORCE_STOP}" == "1" ]]; then
    sleep 2
    if [[ -f "${supervisor_file}" ]]; then
      local supervisor_pid_force
      read -r supervisor_pid_force < "${supervisor_file}"
      if is_pid_running "${supervisor_pid_force}"; then
        kill -9 "${supervisor_pid_force}" 2>/dev/null || true
        echo "[KILL] supervisor pid=${supervisor_pid_force}"
      fi
    fi

    shopt -s nullglob
    for pid_file in "${CURRENT_SESSION_DIR}"/pids/*.pid; do
      local pid_force
      read -r pid_force < "${pid_file}"
      if is_pid_running "${pid_force}"; then
        kill -9 "${pid_force}" 2>/dev/null || true
        echo "[KILL] pid=${pid_force} file=$(basename "${pid_file}")"
      fi
    done
    shopt -u nullglob
  fi

  echo "已发送停止信号: ${stopped_count} 个；已退出: ${skipped_count} 个。"
}

case "${ACTION}" in
  start)
    start_action
    ;;
  resume)
    resume_action
    ;;
  status)
    status_action
    ;;
  stop)
    stop_action
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
