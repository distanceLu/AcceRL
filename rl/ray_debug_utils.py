"""
Ray Actor 调试工具：在每个 Ray Actor 进程中启动 debugpy 服务端，
让 VS Code 可以 attach 到任意 Actor 进程进行断点调试。

使用方式：
  1. 命令行添加 --debug 启动调试（Actor 启动 debugpy.listen）
  2. 命令行添加 --debug-wait 让 Actor 等待 VS Code attach 后才继续
  3. 在 VS Code launch.json 中选择对应的 attach 配置连接到指定端口

端口分配表：
  ┌──────────────────────┬────────────┬──────────────────┐
  │ Actor 类型            │ 基础端口   │ 端口范围          │
  ├──────────────────────┼────────────┼──────────────────┤
  │ TrainerActor          │ 5700       │ 5700 + rank       │
  │ InferenceActor        │ 5710       │ 5710 + actor_id   │
  │ RewardInferenceActor  │ 5720       │ 5720 + actor_id   │
  │ DenoiserInferenceActor│ 5730       │ 5730 + actor_id   │
  │ RolloutWorkerActor    │ 5740       │ 5740 + wid        │
  │ EvaluationWorkerActor │ 5760       │ 5760 + wid        │
  │ StatsActor            │ 5780       │ 5780              │
  │ ReplayBufferActor     │ 5781       │ 5781 + index      │
  │ WMReplayBufferActor   │ 5785       │ 5785 + index      │
  └──────────────────────┴────────────┴──────────────────┘
"""
import os
import sys

# Actor 类型 -> 基础端口映射
PORT_MAP = {
    "trainer": 5700,
    "inference": 5710,
    "reward_inference": 5720,
    "denoiser_inference": 5730,
    "ctrl_world_inference": 5730,
    "rollout_worker": 5740,
    "eval_worker": 5760,
    "stats": 5780,
    "replay": 5781,
    "wm_replay": 5785,
}


def setup_debugger(actor_type: str, actor_id: int = 0, wait: bool = None):
    """
    在 Ray Actor 进程中启动 debugpy 服务端。

    Args:
        actor_type:  Actor 类型字符串，用于查找基础端口（见 PORT_MAP）
        actor_id:    Actor 的 ID/rank/wid，用作端口偏移
        wait:        是否等待调试器 attach 后再继续。
                     None 时读取 RAY_DEBUG_WAIT 环境变量。

    环境变量：
        RAY_DEBUG:       设为 "1" 时启用调试（否则本函数直接返回）
        RAY_DEBUG_WAIT:  设为 "1" 时 Actor 会阻塞等待 VS Code attach
    """
    if not os.environ.get("RAY_DEBUG"):
        return

    try:
        import debugpy
    except ImportError:
        print(
            f"[DEBUG] {actor_type}_{actor_id}: debugpy 未安装，跳过调试。"
            f" 请运行: pip install debugpy",
            flush=True,
        )
        return

    base_port = PORT_MAP.get(actor_type, 5800)
    port = base_port + actor_id

    try:
        debugpy.listen(port)
    except (OSError, RuntimeError) as e:
        # 端口可能被占用，尝试 +10000 偏移
        fallback_port = port + 10000
        print(
            f"[DEBUG] {actor_type}_{actor_id}: 端口 {port} 被占用 ({e})，"
            f"尝试备用端口 {fallback_port}",
            flush=True,
        )
        try:
            debugpy.listen(fallback_port)
            port = fallback_port
        except Exception:
            print(
                f"[DEBUG] {actor_type}_{actor_id}: 无法启动 debugpy，跳过调试。",
                flush=True,
            )
            return

    print(
        f"[DEBUG] {actor_type}_{actor_id} debugpy 已监听端口 {port}"
        f" (PID={os.getpid()})。"
        f" 在 VS Code 中使用 attach 配置连接此端口。",
        flush=True,
    )

    if wait is None:
        wait = os.environ.get("RAY_DEBUG_WAIT", "").lower() in ("1", "true", "yes")

    if wait:
        print(
            f"[DEBUG] {actor_type}_{actor_id} 正在等待 VS Code 调试器 attach"
            f" (端口 {port})...",
            flush=True,
        )
        debugpy.wait_for_client()
        print(f"[DEBUG] {actor_type}_{actor_id} 调试器已连接！", flush=True)


def get_port(actor_type: str, actor_id: int = 0) -> int:
    """查询某个 Actor 的调试端口（供外部使用）。"""
    base_port = PORT_MAP.get(actor_type, 5800)
    return base_port + actor_id
