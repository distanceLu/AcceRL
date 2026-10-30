"""Actual zero-GPU Ray signature/trace regression with an explicit CPU worker stub.

The production actor factory, install_hooks, profile_rpc and SQLite telemetry are
executed. No WEAVER model, native subprocess, optimizer, CUDA or IPC is tested.
"""
from __future__ import annotations
import argparse, asyncio, inspect, json, os, sqlite3, sys, tempfile, time, uuid
from pathlib import Path
from types import SimpleNamespace


class TestHealthMixin:
    def _check_inference_health(self):
        if getattr(self, 'inference_fatal_error', None):
            raise RuntimeError('Synthetic background failure')

    def _fail_pending_requests(self, error):
        self.inference_fatal_error = error
        for promise in self.promises:
            if not promise.done():
                promise.set_exception(RuntimeError(str(error)))
        self.requests, self.promises = [], []


class TestCommunicationBase:
    pass


class CPUWorkerStub:
    """Explicit test replacement for the isolated CUDA owner, never production."""
    def __init__(self, config, rank, role):
        assert role == 'inference'
        self.version = 0
        self.calls = []
        self.receipt = {'gpu': {'uuid': 'explicit-cpu-stub-no-gpu'},
                        'geometry': {'required_context_frames': 32, 'training_window_frames': 40, 'horizon': 8},
                        'test_stub': True, 'native_worker_started': False}

    def call(self, method, *args, **kwargs):
        self.calls.append({'method': method, 'argument_count': len(args), 'keyword_names': sorted(kwargs)})
        if method == 'init_rollout':
            return {'backend': 'weaver', 'version': self.version, 'initial_reward_score': .25,
                    'initial_terminal': False, 'test_stub': True}
        if method == 'apply':
            self.version = int(args[0]['test_version'])
            return {'version': self.version, 'test_stub': True}
        if method == 'request':
            import numpy as np
            state, actions, instruction = args
            if state['version'] != self.version:
                raise ValueError('Stale WEAVER rollout/cache version: explicit CPU fixture')
            count = len(actions)
            return {'observations': np.zeros((count, 2, 3, 2, 2), dtype=np.float32),
                    'reward_predictions': np.full(count, .75, dtype=np.float32),
                    'terminal_predictions': np.zeros(count, dtype=bool),
                    'served_version': self.version, 'next_state': dict(state),
                    'test_stub': True}
        raise ValueError('Unknown CPU stub method: ' + method)


def process_identity(pid):
    try:
        folder = Path('/proc') / str(pid)
        raw = (folder/'stat').read_text()
        fields = raw[raw.rindex(')')+2:].split()
        return {'pid': int(pid), 'uid': folder.stat().st_uid,
                'birth_ticks': int(fields[19]), 'state': fields[0]}
    except FileNotFoundError:
        return None


def alive_exact(expected):
    actual = process_identity(expected['pid'])
    return bool(actual and actual['state'] not in ('Z', 'X') and all(
        actual[key] == expected[key] for key in ('pid', 'uid', 'birth_ticks')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    for key in list(os.environ):
        if key.startswith(('NSYS', 'NVTX_INJECTION', 'CUDA_INJECTION')) or key == 'LD_PRELOAD':
            os.environ.pop(key)
    sys.path.insert(1, str(source))
    import numpy as np
    import ray
    import torch
    from capture_hooks import install_hooks, profile_rpc, batch_begin, batch_end
    from capture_telemetry import Telemetry
    from weaver_online_bridge import make_weaver_online_inference, make_weaver_online_trainer
    assert not ray.is_initialized() and not torch.cuda.is_initialized()
    assert 'telemetry_meta' not in inspect.signature(make_weaver_online_inference(
        TestHealthMixin, TestCommunicationBase, {}).request).parameters
    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix='irt_', dir='/dev/shm'))
    run_id = 'weaverintegrated_raytelemetry_cpu_' + uuid.uuid4().hex[:12]
    db = root/'collection.sqlite'
    os.environ.update(CUDA_VISIBLE_DEVICES='', ACCERL_RUN_ID=run_id,
        ACCERL_TELEMETRY_DB=str(db), ACCERL_CPU_THREADS='1', ACCERL_NSIGHT_CAPTURE='0')
    os.environ['PYTHONPATH'] = os.pathsep.join((str(Path(__file__).resolve().parent), str(source)))
    runtime = {'_online_process_client': CPUWorkerStub, 'batch_begin': batch_begin, 'batch_end': batch_end}
    base = make_weaver_online_inference(TestHealthMixin, TestCommunicationBase, runtime)

    class ProbeActor(base):
        def inspect_test_state(self):
            self.telemetry.flush(force_gpu=False)
            return {'identity': process_identity(os.getpid()), 'cuda_initialized': torch.cuda.is_initialized(),
                    'pending_metadata': list(self.capture_pending_meta),
                    'served_metadata': dict(self.capture_served_versions),
                    'calls': self.worker_client.calls, 'model_version': self.model_version}

        async def close_test(self):
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self.telemetry.close()

    actor = None
    writer = None
    owned = []
    report = {'scope': 'actual Ray zero-GPU actor registration and production telemetry with explicit CPU worker stub',
              'native_model_loaded': False, 'optimizer_or_ddp_tested': False,
              'cuda_or_ipc_tested': False, 'run_id': run_id, 'python': sys.executable,
              'torch_version': torch.__version__, 'ray_version': ray.__version__}
    try:
        ray.init(address='local', num_cpus=2, num_gpus=0, include_dashboard=False,
                 object_store_memory=256*1024**2, _temp_dir=str(root/'ray'),
                 _node_ip_address='127.0.0.1', log_to_driver=False,
                 runtime_env={'env_vars': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                              'PYTHONPATH': os.environ['PYTHONPATH']}})
        node = ray._private.worker._global_node
        for values in node.all_processes.values():
            for value in values:
                identity = process_identity(value.process.pid)
                if identity:
                    owned.append(identity)
        remote_class = ray.remote(num_cpus=0, num_gpus=0)(ProbeActor)
        from ray._common.signature import flatten_args
        before_signature = remote_class.__ray_metadata__.method_meta.signatures['request']
        try:
            flatten_args(before_signature, ({'version': 0}, [], None), {'telemetry_meta': {}})
        except TypeError as error:
            assert 'telemetry_meta' in str(error)
        else:
            raise AssertionError('Regression fixture must reproduce the old Ray signature rejection')
        namespace = {'WeaverOnlineWorldInferenceActor': remote_class}
        assert install_hooks(namespace) == ['WeaverOnlineWorldInferenceActor']
        modified = remote_class.__ray_metadata__.modified_class
        assert 'telemetry_meta' in inspect.signature(modified.request).parameters
        flatten_args(remote_class.__ray_metadata__.method_meta.signatures['request'],
                     ({'version': 0}, [], None), {'telemetry_meta': {}})
        # Keep production methods unchanged: only request receives injected metadata.
        assert 'telemetry_meta' not in inspect.signature(modified.init_rollout).parameters
        assert 'telemetry_meta' not in inspect.signature(modified.apply_rank_weights).parameters
        trainer = make_weaver_online_trainer({})
        method_signatures = {name: str(inspect.signature(getattr(trainer, name))) for name in
            ('export_rank_weights', 'finish_ipc_weight_publish', 'export_native_snapshot')}
        cfg = json.loads((source/'examples/weaver_integrated_internal.json').read_text())
        actor = remote_class.remote(0, None, {'world_backend_config': cfg}, 8, 300)
        initial = ray.get(actor.inspect_test_state.remote(), timeout=30)
        owned.append(initial['identity'])
        assert not initial['cuda_initialized']
        writer = Telemetry('rollout', 'cpu_fixture', flush_records=1)
        caller = SimpleNamespace(telemetry=writer, capture_role='rollout', wid=3,
                                 capture_episode_id='explicit-cpu-fixture-episode')
        geometry = profile_rpc(caller, 'ctrl_infer', actor, 'get_context_geometry')
        assert geometry['required_context_frames'] == 32
        version0 = profile_rpc(caller, 'ctrl_infer', actor, 'init_rollout', {'fixture': True})
        assert version0['version'] == 0 and caller.capture_last_versions['ctrl'] == 0
        applied = profile_rpc(caller, 'ctrl_infer', actor, 'apply_rank_weights', {'test_version': 7}, 'ctrl')
        assert applied['version'] == 7
        state = profile_rpc(caller, 'ctrl_infer', actor, 'init_rollout', {'fixture': True})
        assert state['version'] == 7 and caller.capture_last_versions['ctrl'] == 7
        context = {'workflow': 'imagined_generation', 'world_outputs': ['observations', 'reward_score']}
        result = profile_rpc(caller, 'ctrl_infer', actor, 'request', state, np.zeros((8, 7), np.float32),
                             'explicit fixture task', _capture_context=context)
        first_id = caller.capture_last_request_id
        assert isinstance(result['observations'], torch.Tensor) and result['served_version'] == 7
        assert caller.capture_last_versions['ctrl'] == 7
        assert caller.capture_last_versions['ctrl_request_id'] == first_id
        stale_failed = False
        try:
            profile_rpc(caller, 'ctrl_infer', actor, 'request', version0, np.zeros((1, 7), np.float32),
                        'explicit stale fixture', _capture_context=context)
        except Exception as error:
            assert 'Stale WEAVER rollout/cache version' in str(error)
            stale_failed = True
        assert stale_failed
        again = profile_rpc(caller, 'ctrl_infer', actor, 'request', state, np.zeros((1, 7), np.float32),
                            'explicit recovered fixture', _capture_context=context)
        assert again['served_version'] == 7
        end = ray.get(actor.inspect_test_state.remote(), timeout=20)
        assert not end['cuda_initialized'] and not end['pending_metadata'] and not end['served_metadata']
        request_calls = [x for x in end['calls'] if x['method'] == 'request']
        assert len(request_calls) == 3 and all(x['argument_count'] == 3 and x['keyword_names'] == [] for x in request_calls)
        writer.flush(force_gpu=False)
        connection = sqlite3.connect(db)
        connection.row_factory = sqlite3.Row
        rows = [dict(x) for x in connection.execute('select kind,actor,actor_id,request_id,model_version,status,details_json from events where run_id=? order by id', (run_id,))]
        connection.close()
        same = [x for x in rows if x['request_id'] == first_id or (x['kind'] in ('batch_dispatch', 'service') and first_id in json.loads(x['details_json'])['request_ids'])]
        kinds = {x['kind'] for x in same}
        assert {'request_submit', 'request_enqueue', 'batch_dispatch', 'service', 'client_wait'} <= kinds
        enqueue = next(x for x in same if x['kind'] == 'request_enqueue')
        info = json.loads(enqueue['details_json'])
        assert info['workflow'] == context['workflow'] and info['world_outputs'] == context['world_outputs']
        assert info['episode_id'] == 'explicit-cpu-fixture-episode' and info['caller_id'] == '3'
        dispatch = next(x for x in same if x['kind'] == 'batch_dispatch')
        assert dispatch['model_version'] == '7'
        wait = next(x for x in same if x['kind'] == 'client_wait')
        assert json.loads(wait['details_json'])['served_versions'] == {'ctrl': 7}
        assert not any('untracked_request' in str(x) for x in rows)
        assert sum(x['kind'] == 'cpu_bridge_placement' for x in rows) == 1
        report.update(passed=True, request_signature_refreshed_before_actor_creation=True,
            registered_actor_key='WeaverOnlineWorldInferenceActor', request_id=first_id,
            trace_join=same, native_stub_argument_counts=request_calls,
            actual_served_version=7, stale_error_then_next_request_succeeded=True,
            pending_and_served_metadata_empty=True, geometry_init_publication_no_extra_keyword=True,
            trainer_signature_review=method_signatures, total_event_count=len(rows),
            actor_identity=end['identity'], cuda_initialized=False)
    finally:
        if actor is not None:
            try:
                ray.get(actor.close_test.remote(), timeout=10)
            finally:
                ray.kill(actor, no_restart=True)
        if writer is not None:
            writer.close()
        ray.shutdown()
        until = time.monotonic() + 8
        while any(alive_exact(x) for x in owned) and time.monotonic() < until:
            time.sleep(.1)
        survivors = [x for x in owned if alive_exact(x)]
        report.update(elapsed_seconds=time.monotonic()-started,
                      cleanup={'private_ray_shutdown': True, 'owned_identities': owned,
                               'survivors': survivors, 'all_owned_exited': not survivors})
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        assert not survivors, survivors
    print(json.dumps({key: report[key] for key in ('passed', 'scope', 'actual_served_version', 'total_event_count', 'cleanup')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
