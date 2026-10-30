"""CPU Ray shells for WEAVER's independent native training/inference workers."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from world_backends.runtime import load_backend_config
from weaver_subprocess import WeaverProcessClient, WeaverRemoteOperations
from weaver_inference_actor import detached_reply_error


class OnlineProcessClient(WeaverProcessClient):
    def __init__(self, config, rank, role):
        options = config['backend_options']
        python = options['training_python' if role == 'trainer' else 'inference_python']
        options = dict(options, inference_python=python)
        config = dict(config, backend_options=options)
        command = [python, '-u', str(Path(__file__).with_name('weaver_online_gpu_worker.py')),
                   '--run-id', os.environ['ACCERL_RUN_ID'], '--rank', str(rank), '--role', role]
        super().__init__(config, rank, command=command)

    def call(self, method, *args, **kwargs):
        result = super().call(method, *args, **kwargs)
        if isinstance(result, dict) and result.get('_weaver_recoverable_error') == 'stale_rollout':
            # Expected version invalidation is a failed request, not a dead
            # inference process. Raise outside the transport's fatal cleanup.
            raise ValueError(result['message'])
        return result


def make_weaver_online_trainer(runtime):
    class WeaverOnlineTrainer:
        """One rank owns one CPU stream and one actual WEAVER-env GPU owner."""
        def __init__(self, rank, world_size, wm_replay_buffer, batch_coordinator, ctrl_world_args,
                     ctrl_batch_size=4, ctrl_accumulation_steps=64, training_role='ctrl', **unused):
            if training_role != 'ctrl' or world_size != 4:
                raise ValueError('Native WEAVER trainer shell requires one of four World owners')
            from capture_telemetry import Telemetry
            self.rank, self.world_size, self.training_role = rank, world_size, 'ctrl'
            self.capture_role = 'ctrl_trainer_bridge'
            self.cpu_gpu_bridge = True
            self.world_config = load_backend_config({'world_backend_config': ctrl_world_args['world_backend_config']})
            self.world_config.validate_admission()
            if not self.world_config.world_trainable:
                raise ValueError('Online WEAVER trainer requires a trainable World configuration')
            self.micro, self.gas = int(ctrl_batch_size), int(ctrl_accumulation_steps)
            if self.micro * self.gas * world_size != 1024:
                raise ValueError('Native WEAVER requires global World batch1024')
            self.local_size = self.micro * self.gas
            self.wm_replay_buffer, self.batch_coordinator = wm_replay_buffer, batch_coordinator
            self.telemetry = Telemetry('ctrl_trainer_bridge', rank)
            factory = runtime.get('_online_process_client', OnlineProcessClient)
            self.worker_client = factory(self.world_config.to_dict(), rank, 'trainer')
            self.global_step = self.ctrl_step = 0
            self.port = None
            self.receipt = None
            self.capture_wm_metadata = None

        def get_node_ip(self):
            return runtime['ray'].util.get_node_ip_address()

        def setup_deepspeed_group(self, master_addr, master_port):
            # Retain only the caller's established RPC name. The returned
            # receipt explicitly identifies native DDP, never a ZeRO engine.
            self.port = int(master_port)
            self.receipt = self.worker_client.call('setup', {'master_addr': master_addr,
                'master_port': self.port, 'micro_batch': self.micro, 'accumulation_steps': self.gas}, timeout=1200)
            self.worker_client.receipt = self.receipt
            return self.receipt

        def get_training_state(self):
            state = self.worker_client.call('state')
            return {'rank': self.rank, 'world_size': 4, 'distributed_world_size': state['distributed_world_size'],
                'training_role': 'ctrl', 'pid': self.receipt['worker_identity']['pid'],
                'gpu': self.receipt['gpu'], 'group_port': self.port,
                'split_train_serial': False, 'cpu_replay_prefetch': True,
                'policy_steps': self.global_step, 'ctrl_steps': state['ctrl_steps'], 'reward_steps': 0,
                'engines': {'ctrl': {'global_steps': state['global_steps'],
                    'skipped_steps': state['skipped_steps'], 'global_batch': state['global_batch']}},
                'world_engine': state, 'world_engine_backend': 'torch_ddp_native_flow_matching',
                'rm_updates':state.get('rm_updates',0),'rm_disabled_updates':state.get('rm_disabled_updates',0),
                'actual_gpu_worker': self.receipt['worker_identity'],
                'cpu_bridge_pid': os.getpid(), 'world_backend': 'weaver'}

        async def run_training_epoch(self, policy_step):
            if type(policy_step) is not int or policy_step != self.global_step + 1:
                raise ValueError('World shell requires the next shared Policy boundary')
            self.worker_client.call('advance', policy_step)
            self.global_step = policy_step
            metrics = {'trained_ctrl': False, 'trained_reward': False,
                       'ctrl_sample_unique_episodes': 0, 'ctrl_sample_fresh_episode_ratio': 0.}
            performance = {'wm_sample_time': 0., 'wm_prep_time': 0., 'wm_batch_wait_time': 0.}
            cohort = None
            learning_rates = {}
            if policy_step % 5 == 0:
                started = time.monotonic()
                while await self.wm_replay_buffer.size.remote() < self.local_size * 4:
                    await asyncio.sleep(1)
                stream = await self.batch_coordinator.reserve_wm_stream.remote(self.rank, self.local_size, 'ctrl')
                metadata = stream['capture_meta']
                if len(metadata['sample_ids']) != self.local_size:
                    raise ValueError('Native World stream lacks full ordered rank sample identities')
                self.capture_wm_metadata = metadata
                metrics['ctrl_sample_unique_episodes'] = stream.get('unique_episodes', 0)
                metrics['ctrl_sample_fresh_episode_ratio'] = stream.get('fresh_episode_ratio', 0.)
                performance['wm_batch_wait_time'] = time.monotonic() - started
                self.worker_client.call('begin_update', policy_step, metadata['reward_cohort'])
                samples, prepare = 0., 0.
                final = None
                for start in range(0, self.local_size, self.micro):
                    fetch = time.monotonic()
                    raw = await self.batch_coordinator.sample_wm_stream.remote(stream['stream_id'], start, start + self.micro)
                    samples += time.monotonic() - fetch
                    observations, actions, rewards, instructions, statistics = raw
                    payloads = statistics.get('backend_samples')
                    if payloads is None or len(payloads) != self.micro:
                        raise ValueError('Native World needs complete measured40-frame state/action payloads')
                    expected = metadata['sample_ids'][start:start + self.micro]
                    if statistics['capture_meta']['sample_ids'] != expected:
                        raise ValueError('Native World micro identities differ from its reserved ordered shard')
                    windows = [dict(payload, obs=observations[i], actions=actions[i], instruction=instructions[i])
                               for i, payload in enumerate(payloads)]
                    before = time.monotonic()
                    result = self.worker_client.call('train_micro', windows, metadata, start, start + self.micro)
                    prepare += time.monotonic() - before
                    final = result['result'] if result['committed'] else final
                    # Never retain a GPU-prepared batch or a whole projected
                    # global shard. The next RPC materializes one micro only.
                    del windows, raw, observations, actions, payloads
                if final is None or final['successful_updates'] != policy_step // 5:
                    raise RuntimeError('Native World did not commit exactly its required optimizer update')
                await self.batch_coordinator.complete_wm_stream.remote(stream['stream_id'])
                self.ctrl_step = final['successful_updates']
                cohort = {key: metadata[key] for key in ('reservation_id', 'shard_start', 'shard_end')}
                cohort['sample_ids_sha256'] = hashlib.sha256(json.dumps(metadata['sample_ids'],
                    separators=(',', ':')).encode()).hexdigest()
                metrics.update(trained_ctrl=True, ctrl_loss=final['loss'],
                    ctrl_grad_norm=final['grad_norm'], ctrl_clip_ratio=float(final['grad_norm'] > 1.),
                    native_reward_trained=final.get('rm_active',False),
                    native_reward_updates=final.get('rm_updates',0))
                loss_key = ('native_half_mse_loss' if self.world_config.backend_options.get(
                    'native_reward_head', {}).get('mode') == 'libero_discounted_success_value_v1'
                    else 'native_half_brier_loss')
                metrics[loss_key] = final.get(loss_key)
                performance.update(wm_sample_time=samples, wm_prep_time=0.,
                    ctrl_train_time=final['update_wall_seconds'], native_world_stream_wall_seconds=prepare)
                learning_rates['ctrl'] = final['learning_rate']
            return {'kind': 'ctrl', 'rank': self.rank, 'policy_step': policy_step,
                    'model_step': self.ctrl_step, 'learning_rates': learning_rates,
                    'metrics': metrics, 'performance': performance, 'wm_cohort': cohort}

        def serial_sample_identity(self, policy_step):
            if policy_step != self.global_step:
                raise RuntimeError('World sample identity requested before the common boundary')
            metadata = self.capture_wm_metadata if policy_step % 5 == 0 else None
            cohort = ({key: metadata[key] for key in ('reservation_id', 'shard_start', 'shard_end', 'sample_ids')}
                      if metadata is not None else None)
            return {'model': 'ctrl', 'rank': self.rank, 'policy_step': policy_step, 'cohort': cohort}

        def export_rank_weights(self, kind):
            if kind != 'ctrl':
                raise ValueError('Native World only publishes its own model')
            return self.worker_client.call('export')

        def finish_ipc_weight_publish(self, kind, token, ack):
            if kind != 'ctrl':
                raise ValueError('Native World cannot ACK another model')
            return self.worker_client.call('release', token, ack)

        def export_native_snapshot(self, directory, reason, policy_step):
            if self.rank != 0 or policy_step != self.global_step:
                raise ValueError('Native World snapshots require the common rank0 boundary')
            return self.worker_client.call('snapshot', str(directory), str(reason), timeout=900)

    return WeaverOnlineTrainer


def make_weaver_online_inference(mixin, communication_base, runtime):
    class WeaverOnlineInference(mixin, communication_base):
        def __init__(self, actor_id, stats_actor, args, inference_batch, inference_timeout_ms):
            self.actor_id, self.stats_actor = actor_id, stats_actor
            self.capture_role, self.cpu_gpu_bridge = 'ctrl_infer', True
            self.world_config = load_backend_config(args)
            self.world_config.validate_admission()
            factory = runtime.get('_online_process_client', OnlineProcessClient)
            self.worker_client = factory(self.world_config.to_dict(), actor_id, 'inference')
            self.gpu_binding = self.worker_client.receipt['gpu']
            self.capture_model_version = self.model_version = 0
            self.model, self.device = None, 'cpu'
            self.batch_size = int(inference_batch)
            self.training_batch_limit = self.batch_size
            self.requests, self.promises = [], []
            self._bg_task = asyncio.get_event_loop().create_task(self._loop())
            self._bg_task.add_done_callback(self._on_done)
        def _on_done(self, task):
            if not task.cancelled() and task.exception() is not None:
                self._fail_pending_requests(task.exception())
        def get_context_geometry(self):
            return self.worker_client.receipt['geometry']
        def init_rollout(self, context, published_version=None):
            self._check_inference_health()
            return self.worker_client.call('init_rollout', context, published_version)
        def release_rollout(self, state):
            self._check_inference_health()
            return self.worker_client.call('release_rollout',state)
        def releaseRollout(self, state):
            return self.release_rollout(state)
        async def request(self, state, actions, instruction=None):
            self._check_inference_health()
            future = asyncio.get_event_loop().create_future()
            self.requests.append((state, actions, instruction))
            self.promises.append(future)
            return await future
        async def _loop(self):
            while True:
                if not self.requests:
                    await asyncio.sleep(.0005)
                    continue
                request, promise = self.requests.pop(0), self.promises.pop(0)
                token = runtime['batch_begin'](self, [request], [promise])
                try:
                    # Convert only decoded outputs back to merged-env CPU Torch.
                    result = WeaverRemoteOperations(self.worker_client).request(*request)
                    runtime['batch_end'](self, token)
                    if not promise.done():
                        promise.set_result(result)
                except BaseException as error:
                    runtime['batch_end'](self, token, error=error)
                    if not promise.done():
                        promise.set_exception(detached_reply_error(error))
                    if 'Stale WEAVER rollout/cache version' not in str(error):
                        self._fail_pending_requests(error)
                        raise
                await asyncio.sleep(0)
        def apply_rank_weights(self, payload, kind):
            if kind != 'ctrl':
                raise ValueError('Native World inference only accepts World IPC publication')
            result = self.worker_client.call('apply', payload)
            self.capture_model_version = self.model_version = int(result['version'])
            return result
        def receive_and_update_weights(self, *args, **kwargs):
            raise RuntimeError('Online native World requires explicit same-env CUDA IPC ACK publication')
    return WeaverOnlineInference
