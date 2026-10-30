"""Real controller functions with CPU receipts for frozen World admission."""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_manual_controller_cpu import load, result, RUN, SOURCE
from world_backend_audit import frozen_world_identity, validate_frozen_world_receipts
from world_backends import build_service_plan


CONFIG = json.loads((SOURCE/'examples/weaver_frozen_external.json').read_text())
PLAN = build_service_plan(CONFIG).to_dict()


def fixture(p=27):
    value = result(p)
    value['training'].pop('ctrl')
    value['world_zero2_verified'] = []
    value['world_memory_policy_verified'] = []
    value['replay'][1][4] = 0
    value['experiment_completed'][0]['details']['service_plan'] = PLAN
    value['split_wm_cohort_validated'] = ([{'details': {
        'wm_consumers':['reward'], 'expected_wm_consumers':4}}] if p>=5 else [])
    frozen = frozen_world_identity(CONFIG)
    value['weaver_frozen_verified'] = [{'pid':400+rank, 'actor_id':str(rank),
        'details':{**frozen,'rank':rank,'gpu_uuid':'GPU-'+str(rank),
                   'trainable_parameter_count':0, 'version':0}}
        for rank in range(4)]
    return value


class FrozenControllerCPU(unittest.TestCase):
    def state(self):
        return {'run_id':RUN,'started_ns':100,'service_plan':PLAN,'world_backend_config':CONFIG}

    def validate(self, value):
        check = load('runner.py','validate_manual_completion',{})
        return check(value,self.state(),{'run_id':RUN,'uid':1004,
                     'reason':'user_requested_stop','requested_ns':1000})

    def test_expected_counts_exclude_world_training_keep_three_inference_roles(self):
        self.assertEqual(PLAN['trainer_kinds'],['policy','reward'])
        self.assertEqual(PLAN['inference_kinds'],['policy','ctrl','reward'])
        self.assertEqual(PLAN['expected_gpu_model_processes'],20)
        self.assertEqual(self.validate(fixture())['steps'],{'policy':27,'reward':5})
        self.assertTrue(self.validate(fixture(0))['stopped_before_first_update'])

    def test_no_ghost_world_trainer_or_shared_eight_consumer_barrier(self):
        for mutate in (
            lambda r:r['world_zero2_verified'].append({}),
            lambda r:r['world_memory_policy_verified'].append({}),
            lambda r:r['replay'][1].__setitem__(4,1024),
            lambda r:r['split_wm_cohort_validated'][0]['details'].update(expected_wm_consumers=8),
            lambda r:r['experiment_completed'][0]['details'].update(service_plan={}),
        ):
            bad=fixture();mutate(bad)
            with self.assertRaises(RuntimeError):self.validate(bad)

    def test_world_frozen_receipt_requires_all_four_unique_gpu_processes(self):
        rows=fixture()['weaver_frozen_verified']
        validate_frozen_world_receipts(rows,CONFIG)
        for mutate in (
            lambda r:r.pop(),
            lambda r:r[0]['details'].update(bundle_sha256='0'*64),
            lambda r:r[0]['details'].update(fixed_step=3100),
            lambda r:r[0]['details'].update(world_trainable=True),
            lambda r:r[0].update(pid=r[1]['pid']),
            lambda r:r[0]['details'].update(gpu_uuid=r[1]['details']['gpu_uuid']),
        ):
            bad=copy.deepcopy(rows);mutate(bad)
            with self.assertRaises(RuntimeError):validate_frozen_world_receipts(bad,CONFIG)

    def test_frozen_world_watchdog_is_excluded(self):
        check=load('runner.py','check_update_watchdogs',{})
        ready={'training':{kind:{'ranks':{str(rank):{'successful_updates':1,
            'last_success_ns':9000_000_000_000} for rank in range(4)}} for kind in ('policy','reward')}}
        check(ready,self.state(),now_ns=9001_000_000_000)
        ready['training']['reward']['ranks']={}
        with self.assertRaises(RuntimeError):check(ready,self.state(),now_ns=9001_000_000_000)

    def test_placements_count_gpu_children_and_reject_ghost_world_owner(self):
        rows=[]
        for rank in range(4):
            for index,role in enumerate(('policy_trainer','reward_trainer','policy_infer','ctrl_infer','reward_infer')):
                rows.append({'actor':role,'actor_id':str(rank),'pid':100+rank*5+index,
                    'details':{'role':role,'physical_gpus':[{'physical_index':rank+4,'uuid':'GPU-'+str(rank)}],
                               'physical_gpu_uuid':'GPU-'+str(rank),'cpu_threads':1}})
        safe=SimpleNamespace(query_events=lambda *a,**kw:rows)
        check=load('gpu47_safety.py','placement_evidence',{'safe':safe})
        self.assertTrue(check(None,RUN,service_plan=PLAN)['complete_expected_placement'])
        rows.append({'actor':'ctrl_infer','actor_id':'0','pid':999,
                     'details':{'role':'ctrl_infer','cpu_gpu_bridge':True}})
        self.assertTrue(check(None,RUN,service_plan=PLAN)['complete_expected_placement'])
        rows.append({**rows[0],'actor':'ctrl_trainer','pid':999,
                     'details':{**rows[0]['details'],'role':'ctrl_trainer'}})
        self.assertFalse(check(None,RUN,service_plan=PLAN)['complete_expected_placement'])


if __name__=='__main__':unittest.main()
