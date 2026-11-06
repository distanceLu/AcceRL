import hashlib,json
from pathlib import Path
import sqlite3,tempfile,unittest
from analyze_inference_contention import analyze

NS=1000000000
class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.db=Path(self.temp.name)/'fixture.sqlite'
        self.c=sqlite3.connect(self.db)
        self.c.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,run_id TEXT,ts_ns INTEGER,actor TEXT,actor_id TEXT,pid INTEGER,kind TEXT,workload TEXT,start_ns INTEGER,end_ns INTEGER,actual_batch INTEGER,duration_ns INTEGER,status TEXT,details_json TEXT,request_id TEXT,submitted_ns INTEGER,queue_mean_ns REAL,queue_p95_ns REAL,queue_max_ns INTEGER,peak_bytes INTEGER,model_version TEXT)')
        self.c.execute('CREATE TABLE system_samples(run_id TEXT,ts_ns INTEGER,payload_json TEXT)')
        self.c.execute('CREATE TABLE replay_snapshots(run_id TEXT,ts_ns INTEGER,replay TEXT,actor_id TEXT,resident_count INTEGER,capacity INTEGER)')
        for step,start,end,joint in [(14,1,5,False),(15,5,15,True)]:
            self.event('split_serial_phase_begin',start,details={'policy_step':step,'model':'policy','rank':0})
            self.event('split_serial_phase_end',end-.1,details={'policy_step':step,'model':'policy','rank':0,'trained':True})
            self.event('synchronized_epoch_completed',end,details={'global_step':step,'completed_epochs':step,'trained_ctrl':joint,'trained_reward':joint,'train_seconds':end-start-.5,'sync_seconds':.5})
    def tearDown(self):self.c.close();self.temp.cleanup()
    def event(self,kind,t,work=None,start=None,end=None,batch=None,status=None,details=None,actor='policy_infer',pid=40,request=None,submitted=None,version=None):
        self.c.execute('INSERT INTO events(run_id,ts_ns,actor,actor_id,pid,kind,workload,start_ns,end_ns,actual_batch,duration_ns,status,details_json,request_id,submitted_ns,queue_mean_ns,queue_p95_ns,queue_max_ns,peak_bytes,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            ('r',int(t*NS),actor,'0',pid,kind,work,None if start is None else int(start*NS),None if end is None else int(end*NS),batch,
             None if start is None or end is None else int((end-start)*NS),status,json.dumps(details or {}),request,None if submitted is None else int(submitted*NS),999,999,999,1000,version))
        self.c.commit()
    def group(self,result,mode,work,metric='service'):
        return next(x for x in result['statistics'] if (x['mode'],x['workload'],x['metric'],x['scope'])==(mode,work,metric,'all_processes'))
    def test_completion_throughput_crossing_and_late_gpu_flush(self):
        self.event('service',4,'policy_infer',2,4,2,'ok')
        self.event('service',6,'policy_infer',4,6,3,'ok')
        self.event('service',50,'policy_infer',7,9,1,'ok')
        value=analyze(self.db,'r');p=self.group(value,'policy_only','policy_infer');j=self.group(value,'joint','policy_infer')
        self.assertEqual(value['mode_wall_seconds'],{'policy_only':4,'joint':10})
        self.assertEqual((p['successful_batch_units'],j['successful_batch_units']),(2,4))
        self.assertEqual((p['successful_units_per_mode_wall_second'],j['successful_units_per_mode_wall_second']),(.5,.4))
        self.assertEqual((p['active_service_union_seconds'],j['active_service_union_seconds']),(3,3))
        self.assertEqual(j['cross_epoch_or_inflight_records'],1)
        self.assertEqual(j['wholly_contained_successful_value']['count'],1)
    def test_exact_queue_requests_not_percentile_of_batch_percentiles(self):
        for request,t in [('a',1.2),('b',1.4)]:
            self.event('request_enqueue',t,'policy_infer',start=t,request=request,submitted=t-.1,details={'workflow':'real_rollout'})
        self.event('batch_dispatch',2,'policy_infer',start=2,batch=2,details={'request_ids':['a','b']})
        value=analyze(self.db,'r');queue=self.group(value,'policy_only','policy_infer','actor_queue_wait_ms')
        self.assertAlmostEqual(queue['successful_value']['p50'],700)
        self.assertAlmostEqual(queue['successful_value']['p95'],790)
        self.assertEqual(value['diagnostics']['queue_requests_correlated'],2)
        self.assertEqual(queue['successful_value']['count'],2)
    def test_workflow_partition_does_not_duplicate_requests_or_attribute_batch_duration(self):
        self.event('service',4,'policy_infer',2,4,4,'ok',details={'workflow_counts':{'real_rollout':1,'imagined_action':2,'evaluation':1}})
        value=analyze(self.db,'r');all_requests=self.group(value,'policy_only','policy_infer')['successful_batch_units']
        partitions=[g for g in value['statistics'] if g['metric'].startswith('service_workflow_')]
        self.assertEqual(sum(g['successful_batch_units'] for g in partitions),all_requests)
        self.assertTrue(all(g['successful_value']['count']==0 for g in partitions))
    def test_client_latency_version_lag_and_unknown_do_not_use_episode_id(self):
        self.event('optimizer_update',1.5,'policy_train',batch=576,details={'data_parallel_rank':0,'successful_optimizer_steps':1},version='2')
        self.event('client_wait',4,'policy_infer',2,4,status='ok',details={'method':'request','served_versions':{'policy':1},'episode_id':'real-looking'},actor='rollout')
        value=analyze(self.db,'r');lat=self.group(value,'policy_only','policy_infer','request_rpc_latency_ms_workflow_unknown')
        self.assertEqual(lat['successful_value']['p95'],2000)
        self.assertEqual(self.group(value,'policy_only','policy_infer','latest_trainer_minus_reply_served_version')['successful_value']['p50'],1)
    def test_prefetch_state_time_integral_and_sampled_memory_are_separate(self):
        def state(t,ready,pending,wait,closed=False):
            self.event('cpu_replay_prefetch_state',t,details={'model':'policy','rank':0,'capacity_batches':1,'ready_batches':ready,'pending_fetches':pending,'in_use_batches':0,'empty_waiters':wait,'closed':closed,'observer_errors':0},actor='policy_trainer')
        state(.5,1,0,0);state(3,0,1,1);state(7,1,0,0);state(14,0,0,0,True)
        self.c.execute('INSERT INTO system_samples VALUES(?,?,?)',('r',6*NS,json.dumps({'gpus':[{'physical_gpu':4,'memory_used_bytes':100,'gpu_busy_percent':100}]})));self.c.commit()
        value=analyze(self.db,'r');p=next(x for x in value['prefetch']['owners'] if x['mode']=='policy_only')
        self.assertEqual(p['state_seconds']['ready_batches'],2)
        self.assertEqual(p['state_seconds']['empty_waiters'],2)
        self.assertEqual(p['coverage_seconds'],4)
        self.assertEqual(value['sampled_physical_gpu_peaks'][0]['sampled_peak_used_bytes'],100)
        self.assertIn('SM_Active_occupancy',value['not_measured'])
    def test_profile_excludes_whole_intersecting_epoch_not_partial_batch_samples(self):
        self.event('nsight_capture_started',8,details={'started_ns':8*NS,'policy_step':15})
        self.event('nsight_capture_ended',10,details={'started_ns':8*NS,'close_requested_ns':9*NS,'nvtx_end_returned_ns':10*NS,'policy_step':15})
        self.event('service',4,'policy_infer',2,4,2,'ok');self.event('service',12,'policy_infer',11,12,3,'ok')
        value=analyze(self.db,'r',exclude_profiled=True)
        self.assertEqual(value['mode_wall_seconds']['joint'],0)
        self.assertEqual(value['diagnostics']['whole_epochs_excluded_profile_overlap'],1)
        self.assertEqual(self.group(value,'policy_only','policy_infer')['successful_batch_units'],2)
    def test_budget_and_incomplete_profile_fail_without_modifying_input(self):
        before=hashlib.sha256(self.db.read_bytes()).hexdigest()
        with self.assertRaises(RuntimeError):analyze(self.db,'r',max_rows=1)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(),before)
        self.event('nsight_capture_started',8,details={'started_ns':8*NS})
        with self.assertRaises(ValueError):analyze(self.db,'r',exclude_profiled=True)
    def test_missing_prefetch_is_not_zero_and_episodes_are_separate(self):
        self.event('service',4,'cpu_wm_windows',3,4,1,'ok',actor='rollout')
        self.event('eval_episode_completed',6,actor='evaluation')
        value=analyze(self.db,'r')
        self.assertEqual(value['prefetch']['status'],'not_collected')
        self.assertEqual({x['kind']:x['count'] for x in value['real_episode_production']},{'real_rollout_postprocessing_completed':1,'eval_episode_completed':1})
    def test_bounded_queue_cache_does_not_change_service_throughput(self):
        for i in range(3):self.event('request_enqueue',1.1+i*.1,'policy_infer',start=1.1+i*.1,request=str(i))
        self.event('batch_dispatch',2,'policy_infer',start=2,batch=3,details={'request_ids':['0','1','2']})
        self.event('service',4,'policy_infer',2,4,3,'ok')
        value=analyze(self.db,'r',max_pending=1)
        self.assertEqual(value['diagnostics']['queue_correlation_cache_evictions'],2)
        self.assertEqual(self.group(value,'policy_only','policy_infer')['successful_batch_units'],3)

if __name__=='__main__':unittest.main()
