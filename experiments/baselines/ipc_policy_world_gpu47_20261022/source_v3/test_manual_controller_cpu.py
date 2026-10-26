"""AST-isolated manual controller fixtures; no runner/safety/preflight/GPU imports."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from training_stop_policy import TrainingStopPolicy
from nsight_capture import NsightCaptureWindows
from world_backend_audit import validate_world_zero2_receipts, world_zero2_report
from policy_backend_audit import policy_zero2_report
from test_policy_backend_audit_cpu import engine

SOURCE=Path(__file__).resolve().parent
RUN='w4manual_cpu_unit'


def load(name, function, namespace):
    tree=ast.parse((SOURCE/name).read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==function)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),name,'exec'),namespace)
    return namespace[function]


def env(inherited=None, verified=True, state=None):
    # Extract only the real standard-library cleaner, not the safety module's
    # archived imports/preflight. formal_environment now delegates to it.
    cleaner=load('gpu47_safety.py','cpu_helper_environment',{'os':os})
    namespace={'os':os,'ROOT':Path('/repo'),
               'safe':SimpleNamespace(cpu_helper_environment=cleaner)}
    identity=state if state is not None else {'run_id':RUN,'started_ns':100}
    trusted=Mock(return_value={'NSYS_FRESH_INJECTION':'fresh','CUDA_INJECTION64_PATH':'fresh'})
    if not verified:trusted.side_effect=RuntimeError('untrusted collector')
    with patch.dict(os.environ,{} if inherited is None else inherited,clear=True), \
            patch.dict(sys.modules,{'profile_plan':SimpleNamespace(load_trusted_gpu_injection=trusted)}):
        value=load('runner.py','formal_environment',namespace)(
            identity,Path('/master/formal'),
            {'cards':[{'uuid':str(x)} for x in (4,5,6,7)],'mps_pipe':'/cpu/pipe','mps_log':'/cpu/log'})
    trusted.assert_not_called()  # IPC validation never accepts profiler injection.
    return value


def result(p=27):
    steps={'policy':p,'ctrl':p//5,'reward':p//5}
    policy=[{'pid':100+rank,'actor_id':str(rank),'details':policy_zero2_report(engine(),rank)} for rank in range(4)]
    world=[{'pid':200+rank,'actor_id':str(rank),'details':world_zero2_report(engine(micro=4,gas=64,global_batch=1024),rank)} for rank in range(4)]
    return {'training':{model:{'global_updates':count,'rank_counts_match':True,
                'ranks':{str(rank):{'successful_updates':count,'skipped_steps':0} for rank in range(4)}} for model,count in steps.items()},
        'policy_zero2_verified':policy,'world_zero2_verified':world,
        'fsdp_sharding_verified':[{'pid':300+rank,'actor_id':str(rank),'details':{'model':'reward','local_parameter_elements':10,'global_parameter_elements':100}} for rank in range(4)],
        'reward_features_only_enabled':[{'details':{'skip_lm_head':True}} for _ in range(4)],
        'reward_checkpointing_verified':[{'details':{'vision_block_count':51}} for _ in range(4)],
        'world_memory_policy_verified':[{} for _ in range(4)],'placements':{'complete_expected_placement':True},
        'experiment_completed':[{'ts_ns':2000,'details':{'global_step':p,'completed_epochs':p,'run_started_ns':100,
            'stop_reason':'scheduled_stop','stop_on_target':False,'until_deadline':False,'deadline_unix_seconds':None,
            'stop_policy':{'until_manual_stop':True,'until_deadline':False,'stop_on_target':False,'deadline_unix_seconds':None}}}],
        'checkpoint':{'reason':'final' if p else 'final_before_first_update'},
        'policy_batch_mode_validation':{'valid':True,'run_id':RUN,'steps':p,'mode_receipts':8*p,'global_exposures':576*p},
        'replay':[['policy',0,0,p*576,0,0],['wm',0,0,0,p//5*1024,p//5*1024]],
        'split_wm_cohort_validated':[{}] if p>=5 else []}


class ManualControllerCPU(unittest.TestCase):
    def test_polluted_environment_gets_exact_manual_contract(self):
        inherited={'PATH':'/cpu','ACCERL_RUN_UNTIL_MANUAL_STOP':'0','ACCERL_SOFT_STOP_AT':'1',
            'ACCERL_SHORT_TEST_DEADLINE_UNIX':'2','ACCERL_MEASURE_STOP_AFTER_EPOCHS':'15','ACCERL_STOP_ON_TARGET_85':'1',
            'NSYS_INJECTION':'bad','CUDA_INJECTION64_PATH':'bad','NVTX_INJECTION64_PATH':'bad',
            'NSYS_NVTX_PROFILER_REGISTER_ONLY':'0','ACCERL_NSIGHT_NVTX':'1',
            'ACCERL_NSIGHT_STEPS':'14,15','ACCERL_NSIGHT_SECONDS':'30','ACCERL_NSIGHT_MODE':'timer',
            'LD_PRELOAD':'/cpu/libkeep.so:/tools/nsight/libToolsInjection64.so:/tools/nsys/libinject.so',
            'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE':'20'}
        value=env(inherited)
        self.assertEqual(value['ACCERL_RUN_UNTIL_MANUAL_STOP'],'1')
        self.assertEqual(value['ACCERL_RUN_UNTIL_DEADLINE'],'0')
        self.assertEqual(value['ACCERL_STOP_ON_TARGET_85'],'0')
        self.assertEqual(value['ACCERL_SNAPSHOT_INTERVAL'],'25')
        self.assertEqual(value['ACCERL_NSIGHT_NVTX'],'0')
        self.assertEqual(value['ACCERL_NSIGHT_STEPS'],'')
        self.assertEqual(value['LD_PRELOAD'],'/cpu/libkeep.so')
        self.assertEqual(value['PATH'],'/cpu')
        for key in ('ACCERL_SOFT_STOP_AT','ACCERL_SHORT_TEST_DEADLINE_UNIX','ACCERL_MEASURE_STOP_AFTER_EPOCHS',
                    'NSYS_INJECTION','NSYS_FRESH_INJECTION','CUDA_INJECTION64_PATH','NVTX_INJECTION64_PATH',
                    'NSYS_NVTX_PROFILER_REGISTER_ONLY','ACCERL_NSIGHT_SECONDS','ACCERL_NSIGHT_MODE',
                    'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'):
            self.assertNotIn(key,value)
        self.assertEqual(value['CUDA_VISIBLE_DEVICES'],'4,5,6,7')

    def test_manual_continues_beyond_fifteen_85_and_lr_horizon(self):
        policy=TrainingStopPolicy.from_environment(env())
        args=dict(global_step=50000,train_iters=30000,target_reached=True,requested_stop=False,
                  completed_epochs=50000,measure_stop_epochs=15,now=9999999999)
        self.assertTrue(policy.should_continue(**args))
        self.assertFalse(policy.should_continue(**dict(args,requested_stop=True)))
        self.assertEqual(policy.stop_reason(**dict(args,requested_stop=True)),'scheduled_stop')
        self.assertIsNone(policy.deadline_unix_seconds)

    def test_disabled_capture_opens_no_former_or_late_windows(self):
        value=env();begin,end,marker,timer=(Mock(side_effect=AssertionError('Profiler must remain disabled')) for _ in range(4))
        writer=SimpleNamespace(event=Mock(),flush=Mock())
        with patch.dict(os.environ,value,clear=True):
            capture=NsightCaptureWindows(writer,begin=begin,end=end,timer_factory=timer,marker=marker)
            for step in (1,10,13,14,15,25,30001,50000):
                capture.before_epoch(step)
                capture.before_stage(step,'policy')
                capture.after_stage(step,'policy')
                capture.after_epoch(step)
            capture.close()
        self.assertEqual(capture.steps,())
        self.assertIsNone(capture.active)
        self.assertIsNone(capture.timer)
        for callback in (begin,end,marker,timer):callback.assert_not_called()
        writer.event.assert_not_called()
        policy=TrainingStopPolicy.from_environment(value)
        self.assertTrue(policy.should_continue(global_step=30001,train_iters=30000,target_reached=True,
            requested_stop=False,completed_epochs=30001,measure_stop_epochs=0,now=9999999999))

    def test_trusted_or_untrusted_measurement_never_enables_injection_or_a_stop(self):
        for verified in (True,False):
            with self.subTest(verified=verified):
                state={'run_id':RUN,'started_ns':100,'state':'running','measurement_injection_verified':True}
                value=env({'NSYS_OLD':'old','CUDA_INJECTION64_PATH':'old','NVTX_INJECTION64_PATH':'old',
                    'ACCERL_CONTENTION_COLLECTOR_TOKEN':'old','LD_PRELOAD':'/tools/nsight/libToolsInjection64.so'},verified,state)
                self.assertEqual(value['ACCERL_NSIGHT_NVTX'],'0')
                self.assertEqual(value['ACCERL_NSIGHT_STEPS'],'')
                self.assertFalse(state['measurement_injection_verified'])
                self.assertNotIn('measurement_configuration_error',state)
                self.assertNotIn('error',state)
                self.assertNotIn('stop_requested_ns',state)
                self.assertEqual(state['state'],'running')
                self.assertNotIn('LD_PRELOAD',value)
                self.assertFalse(any(key.startswith(('NSYS_','CUDA_INJECTION','NVTX_INJECTION','ACCERL_CONTENTION')) for key in value))

    def test_manual_gate_and_formal_directory_are_first(self):
        tree=ast.parse((SOURCE/'runner.py').read_text(encoding='utf-8'))
        run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
        condition=compile(ast.Expression(run.body[0].test),'manual_gate','eval')
        for mode in ('preflight','target','restart','handoff'):
            self.assertTrue(eval(condition,{'mode':mode,'stop_on_target':False}))
        self.assertFalse(eval(condition,{'mode':'manual','stop_on_target':False}))
        self.assertTrue(eval(condition,{'mode':'manual','stop_on_target':True}))
        code=ast.unparse(run)
        self.assertIn("safe.MASTER / 'formal_v3'",code)
        self.assertIn("no_resume",code)
        self.assertIn("'bash'",code)
        self.assertNotIn('profile_command',code)
        self.assertNotIn('preflight_passed',code)

    def test_safety_never_reads_mandatory_or_legacy_deadline(self):
        code=(SOURCE/'gpu47_safety.py').read_text(encoding='utf-8')
        tree=ast.parse(code)
        assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEADLINE_UNIX' for t in x.targets))
        self.assertIsNone(ast.literal_eval(assignment.value))
        self.assertNotIn('ACCERL_SHORT_TEST_DEADLINE_UNIX',code)
        pre=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='preflight')
        self.assertNotIn('DEADLINE_UNIX',ast.unparse(pre.body[0]))
        namespace={'os':os}
        with patch.dict(os.environ,{'ACCERL_RUN_ID':'old','ACCERL_SHORT_TEST_DEADLINE_UNIX':'1',
                'ACCERL_CONTENTION_COLLECTOR_TOKEN':'old','NSYS_OLD':'bad','CUDA_INJECTION64_PATH':'bad',
                'NVTX_INJECTION64_PATH':'bad'},clear=True):
            value=load('gpu47_safety.py','cpu_helper_environment',namespace)()
        self.assertEqual(value['CUDA_VISIBLE_DEVICES'],'')
        self.assertFalse(any(key.startswith(('ACCERL_','NSYS_','CUDA_INJECTION','NVTX_INJECTION')) for key in value))

    def test_requested_stop_anchors_900_without_old_time_deadline(self):
        with tempfile.TemporaryDirectory() as td:
            data=Path(td); (data/'training').mkdir()
            state={'run_id':RUN}
            (data/'training/stop.flag').write_text(json.dumps({'reason':'user_requested_stop','run_id':RUN,'uid':1004,'requested_ns':1000_000_000_000}))
            namespace={'time':SimpleNamespace(time=lambda:1001),'json':json,
                       'SaveGraceExpired':type('SaveGraceExpired',(RuntimeError,),{})}
            load('runner.py','observe_stop_request',namespace)
            check=load('runner.py','enforce_stop_window',namespace)
            child=SimpleNamespace(poll=lambda:None)
            check(data,state,child,now=1899)
            self.assertEqual(state['stop_requested_ns'],1000_000_000_000)
            with self.assertRaises(namespace['SaveGraceExpired']):check(data,state,child,now=1900)
            self.assertEqual(state['stop_requested_ns'],1000_000_000_000)

    def test_preflight_mps_caps_are_read_only_exact_100(self):
        cards=[{'index':n,'uuid':'GPU-'+str(n),'used_mib':75,'utilization_percent':0} for n in (4,5,6,7)]
        receipt={'server':{'pid':99,'uid':1004,'birth_ticks':1},
                 'control':{'pid':98,'uid':1004,'birth_ticks':2},
                 'cards':cards,'pipe':'/cpu/pipe','log':'/cpu/log'}
        identities={value['pid']:value for value in (receipt['server'],receipt['control'])}
        responses={'get_server_list':'99','get_default_active_thread_percentage':'100',
                   'get_active_thread_percentage 99':'100.0'}
        control=Mock(side_effect=lambda command:responses[command])
        namespace={'os':SimpleNamespace(getuid=lambda:1004),'mps_receipt':lambda:receipt,
                   'safe':SimpleNamespace(proc_identity=lambda pid:identities[pid],
                       same_process=lambda old,current:old==current,protected=lambda _:True),
                   'control':control,'mps_clients':lambda _:[], 'gpu_snapshot':lambda:cards,
                   'legacy':SimpleNamespace(command_output=Mock(return_value=''))}
        function=load('gpu47_safety.py','preflight',namespace)
        self.assertIsNone(function()['deadline_unix'])
        responses['get_active_thread_percentage 99']='80'
        with self.assertRaises(RuntimeError):function()
        self.assertTrue(all(call.args[0].startswith('get_') for call in control.call_args_list))

    def test_error_grace_cannot_extend_the_user_save_window(self):
        function=load('runner.py','error_wait_budget',{'time':SimpleNamespace(time=lambda:0)})
        self.assertEqual(function({},now=9999999999),120)
        self.assertEqual(function({'stop_requested_ns':1000_000_000_000},now=1890),10)
        self.assertEqual(function({'error_started_ns':1000_000_000_000},now=1110),10)

    def test_successful_update_watchdogs_and_save_phase(self):
        function=load('runner.py','check_update_watchdogs',{'time':SimpleNamespace(time_ns=lambda:0)})
        state={'started_ns':1000_000_000_000}
        function({},state,now_ns=8200_000_000_000)
        with self.assertRaises(RuntimeError):function({},state,now_ns=8201_000_000_000)
        ready={'training':{model:{'ranks':{str(rank):{'successful_updates':1,'last_success_ns':1000_000_000_000} for rank in range(4)}} for model in ('policy','ctrl','reward')}}
        with self.assertRaises(RuntimeError):function(ready,state,now_ns=4601_000_000_000)
        function(ready,dict(state,stop_requested_ns=4000_000_000_000),now_ns=9999999999_000_000_000)

    def test_only_matching_user_stop_and_learning_event_accept_exit0(self):
        validate=load('runner.py','validate_manual_completion',{'validate_world_zero2_receipts':validate_world_zero2_receipts})
        state={'run_id':RUN,'started_ns':100}; flag={'run_id':RUN,'uid':1004,'reason':'user_requested_stop','requested_ns':1000}
        self.assertEqual(validate(result(),state,flag)['steps'],{'policy':27,'ctrl':5,'reward':5})
        for changed in ({},dict(flag,reason='user_deadline_reached'),dict(flag,run_id='old'),dict(flag,requested_ns=3000)):
            with self.assertRaises(RuntimeError):validate(result(),state,changed)
        for field,value in [('stop_reason','training_plan_completed'),('until_deadline',True),('run_started_ns',99)]:
            changed=result();changed['experiment_completed'][0]['details'][field]=value
            with self.assertRaises(RuntimeError):validate(changed,state,flag)

    def test_fill_stop_zero_updates_is_explicit(self):
        validate=load('runner.py','validate_manual_completion',{'validate_world_zero2_receipts':validate_world_zero2_receipts})
        state={'run_id':RUN,'started_ns':100};flag={'run_id':RUN,'uid':1004,'reason':'user_requested_stop','requested_ns':1000}
        value=result(0);value['training']={};value['replay']=[]
        accepted=validate(value,state,flag)
        self.assertTrue(accepted['stopped_before_first_update'])
        self.assertEqual(accepted['checkpoint_reason'],'final_before_first_update')
        value['checkpoint']['reason']='final'
        self.assertEqual(validate(value,state,flag)['checkpoint_reason'],'final')
        value['checkpoint']['reason']='initial'
        with self.assertRaises(RuntimeError):validate(value,state,flag)

    def test_request_stop_identity_and_owned_flag(self):
        with tempfile.TemporaryDirectory() as td:
            master=Path(td);data=master/'formal_v3';(data/'training').mkdir(parents=True)
            identity={'pid':10,'uid':1004,'birth_ticks':123}
            state={'uid':1004,'run_id':RUN,'mode':'manual','until_manual_stop':True,'deadline_unix':None,
                   'stop_on_target':False,'state':'running','supervisor':identity}
            current=dict(identity,state='S')
            def stop(folder,reason):
                (folder/'training/stop.flag').write_text(json.dumps({'run_id':RUN,'uid':1004,'reason':reason,'requested_ns':999}))
            safe=SimpleNamespace(MASTER=master,proc_identity=lambda _:current,same_process=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','birth_ticks')))
            runner=SimpleNamespace(load=lambda _:state,stop=Mock(side_effect=stop))
            function=load('request_stop.py','request',{'os':SimpleNamespace(getuid=lambda:1004),'safe':safe,'runner':runner,'json':json})
            self.assertTrue(function(data,RUN)['stop_requested'])
            current['birth_ticks']=124
            with self.assertRaises(RuntimeError):function(data,RUN)
            with self.assertRaises(RuntimeError):function(master/'other',RUN)

    def test_cpu_weights_audit_accepts_final_and_fill_zero_only_with_manual_identity(self):
        for p,reason in ((0,'final_before_first_update'),(0,'final'),(27,'final')):
            with self.subTest(p=p,reason=reason), tempfile.TemporaryDirectory() as td:
                data=Path(td);directory=data/'training/checkpoints';directory.mkdir(parents=True)
                path=directory/'latest_weights.pt';path.write_bytes(b'fixture')
                steps={'policy':p,'ctrl':p//5,'reward':p//5}
                m={'metadata':{'run_id':RUN,'stop_policy':{'until_manual_stop':True,'until_deadline':False,'deadline_unix_seconds':None,'stop_on_target':False}},
                   'world_size':4,'reason':reason,'steps':steps,'weights_only':True,
                   'tensor_counts':{model:1 for model in steps}}
                (directory/'manifest.json').write_text(json.dumps(dict(m,weights_file=path.name,weights_bytes=7)))
                tensor=SimpleNamespace(device=SimpleNamespace(type='cpu'))
                payload={'manifest':m,'models':{model:{'state_dict':{'p':tensor},'trainable_names':['p'],'full_state':model=='reward'} for model in steps}}
                torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False),load=Mock(return_value=payload),is_tensor=lambda value:value is tensor)
                state={'run_id':RUN,'report':{'training':{model:{'global_updates':count} for model,count in steps.items()}}}
                namespace={'os':SimpleNamespace(environ={}), 'legacy':SimpleNamespace(stamp=lambda:'cpu'),'json':json}
                audit=load('gpu47_safety.py','audit_final_weights',namespace)
                self.assertTrue(audit(data,state,torch_module=torch,persist=False)['valid'])
                torch.load.assert_called_once_with(str(path),map_location='cpu',weights_only=True,mmap=True)
                payload['manifest']['metadata']['run_id']='old'
                self.assertFalse(audit(data,state,torch_module=torch,persist=False)['valid'])


if __name__=='__main__':
    unittest.main(verbosity=2)
