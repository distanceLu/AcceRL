"""Deployment must activate both changes after environment cleaning."""
import ast
import copy
import json
import unittest
import test_cost_deployment_cpu as base
import training_entry as wrapper


class CombinedDeploymentTests(unittest.TestCase):
    setUp=base.CostDeploymentTests.setUp
    bind_manifest=base.CostDeploymentTests.bind_manifest
    launch=base.CostDeploymentTests.launch
    frozen=base.CostDeploymentTests.frozen
    def test_real_runner_constants_match_wrapper(self):
        tree=ast.parse((base.CANDIDATE/'runner.py').read_text('utf-8'))
        values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
                if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in ('COST_SCHEDULER_PROFILE','COST_SCHEDULER_RELATIVE')}
        self.assertEqual(values['COST_SCHEDULER_PROFILE'],wrapper.COST_SCHEDULER_PROFILE)
        self.assertEqual(values['COST_SCHEDULER_RELATIVE'],wrapper.COST_SCHEDULER_CONFIG)

    def test_combined_configuration_reaches_runtime(self):
        data,configuration=self.frozen()
        env=self.namespace['cost_scheduler_environment_configuration']({'cost_scheduler':configuration},data)
        from cost_scheduler_runtime import load_settings
        cost=load_settings(env['ACCERL_COST_SCHEDULER_CONFIG'])
        self.assertEqual(cost['training_gangs'],2)
        self.assertEqual(cost['card_inflight'],3)
        self.assertTrue(cost['cycle_feedback']['enabled'])
        self.assertEqual(cost['cycle_feedback']['warmup_policy_updates'],20)

    def test_serial_or_disabled_feedback_is_rejected(self):
        original=json.loads(self.profile.read_text('utf-8'))
        variants=[]
        for key,value in [('training_gangs',1),('card_inflight',2)]:
            bad=copy.deepcopy(original);bad[key]=value;variants.append(bad)
        bad=copy.deepcopy(original);bad['cycle_feedback']['enabled']=False;variants.append(bad)
        for settings in variants:
            self.profile.write_text(json.dumps(settings),encoding='utf-8');self.bind_manifest()
            with self.assertRaisesRegex(ValueError,'Combined parallel/feedback'):
                wrapper.validate_cost_scheduler_admission(self.source)

    def test_wrong_actual_parallel_state_is_rejected(self):
        data,configuration=self.frozen()
        state={'cost_scheduler':configuration,
               'pipeline_config':json.loads((self.source/wrapper.PIPELINE_CONFIG).read_text('utf-8')),
               'pipeline_config_path':str(self.root/'formal_v3/pipeline_config.json')}
        state['cost_scheduler']['settings']['training_gangs']=1
        with self.assertRaisesRegex(ValueError,'did not activate'):
            wrapper.check_cost_scheduler_state(state,self.source,self.root)


if __name__=='__main__':unittest.main()
