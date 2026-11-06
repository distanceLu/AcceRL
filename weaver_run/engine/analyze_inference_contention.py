"""Read-only CPU comparison of services during complete P-only/joint epochs.

Uses the existing analyzer's exact schemas/profile markers. Never imports torch,
changes requests, joins request tables, or derives kernel/SM resource shares.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import time

_base_path=Path(__file__).with_name('analyze_short_measurements.py')
if not _base_path.is_file():
    _base_path=Path(__file__).resolve().parents[1]/'candidate/analyze_short_measurements.py'
_spec=importlib.util.spec_from_file_location('contention_base_analyzer',_base_path)
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
INFER=base.INFER
SERVICES=set(base.SERVICE_WORKLOADS)
MODELS={'policy_train':'policy','ctrl_train':'ctrl','reward_train':'reward'}
WORKFLOWS={'real_rollout','evaluation','imagined_action','imagined_bootstrap','imagined_generation','unknown'}

def workflow(value):return value if isinstance(value,str) and value in WORKFLOWS else 'unknown'


class Budget:
    def __init__(self,seconds,max_rows):self.until=time.monotonic()+seconds;self.maximum=max_rows;self.count=0
    def check(self):
        if time.monotonic()>self.until:raise TimeoutError('CPU analysis time budget expired; no partial report returned')
    def rows(self,c,sql,args=()):
        for row in base.rows(c,sql,args):
            self.count+=1
            if self.count>self.maximum:raise RuntimeError('CPU analysis row budget exceeded; narrow epoch range')
            self.check();yield row


class Epochs:
    def __init__(self,items):
        self.items=sorted(items,key=lambda x:x['start_ns']);self.starts=[x['start_ns'] for x in self.items]
        if any(a['end_ns']>b['start_ns'] for a,b in zip(self.items,self.items[1:])):
            raise ValueError('Complete epoch windows overlap; cannot attribute a wall-clock denominator')
        self.wall={mode:sum(x['end_ns']-x['start_ns'] for x in self.items if x['mode']==mode)/1e9
                   for mode in ('policy_only','joint')}
    def at(self,t):
        if t is None:return None
        i=bisect_right(self.starts,t)-1
        return self.items[i] if i>=0 and t<self.items[i]['end_ns'] else None
    def contained(self,start,end):
        epoch=self.at(start)
        return epoch if epoch and end is not None and start<=end<=epoch['end_ns'] else None
    def overlaps(self,start,end):
        i=max(0,bisect_right(self.starts,start)-1)
        while i<len(self.items) and self.items[i]['start_ns']<end:
            item=self.items[i];lo=max(start,item['start_ns']);hi=min(end,item['end_ns'])
            if hi>lo:yield item,lo,hi
            i+=1


class Stats:
    """Exact quantiles on a temporary SQLite file; bounded Python group state."""
    def __init__(self):
        self.c=sqlite3.connect('');self.c.execute('CREATE TABLE q(g TEXT,v REAL,contained INT)')
        self.c.execute('CREATE TABLE intervals(g TEXT,s INTEGER,e INTEGER)')
        self.groups={}
    def group(self,mode,workload,metric,actor=None,pid=None):
        key=json.dumps([mode,workload,metric,actor,pid])
        return key,self.groups.setdefault(key,{'mode':mode,'workload':workload,'metric':metric,
            'actor_id':actor,'pid':pid,'scope':'process' if pid is not None else 'all_processes',
            'units':('successful inference requests' if workload in INFER and (metric=='service' or metric.startswith('service_workflow_'))
                     else 'actual recorded-rank micro sample exposures' if workload in MODELS and metric=='service'
                     else 'local macro batch sizes; not additional exposure' if workload.endswith('_train_update') and metric=='service'
                     else 'successful global optimizer updates' if metric=='global_optimizer_updates'
                     else 'dispatched-request queue observations; service outcome not inferred' if 'queue_wait' in metric
                     else 'observations'),
            'value_unit':'version_steps' if metric=='latest_trainer_minus_reply_served_version' else 'milliseconds',
            'records':0,'successful_records':0,'successful_batch_units':0,'batch_distribution':Counter(),
            'cross_epoch_or_inflight_records':0,'statuses':Counter(),'peak_allocated_bytes':None})
    def add(self,mode,workload,metric,value,*,units=1,status='ok',contained=True,actor=None,pid=None,peak=None):
        for owner,process in ((None,None),(actor,pid)) if pid is not None else ((None,None),):
            key,g=self.group(mode,workload,metric,owner,process)
            g['records']+=1;g['statuses'][str(status)]+=1
            if status=='ok':
                g['successful_records']+=1;g['successful_batch_units']+=units
                g['batch_distribution'][str(units)]+=1
                if value is not None:self.c.execute('INSERT INTO q VALUES(?,?,?)',(key,value,int(contained)))
            if not contained:g['cross_epoch_or_inflight_records']+=1
            if isinstance(peak,(int,float)):g['peak_allocated_bytes']=max(g['peak_allocated_bytes'] or 0,peak)
    def active(self,mode,workload,start,end,actor=None,pid=None):
        for owner,process in ((None,None),(actor,pid)) if pid is not None else ((None,None),):
            key,_=self.group(mode,workload,'service',owner,process)
            self.c.execute('INSERT INTO intervals VALUES(?,?,?)',(key,start,end))
    def quantile(self,key,contained=False):
        clause='g=?'+(' AND contained=1' if contained else '')
        n,lo,hi=self.c.execute('SELECT COUNT(*),MIN(v),MAX(v) FROM q WHERE '+clause,(key,)).fetchone()
        def q(f):
            at=(n-1)*f;a=math.floor(at);b=math.ceil(at)
            left=self.c.execute('SELECT v FROM q WHERE '+clause+' ORDER BY v LIMIT 1 OFFSET ?',(key,a)).fetchone()[0]
            right=self.c.execute('SELECT v FROM q WHERE '+clause+' ORDER BY v LIMIT 1 OFFSET ?',(key,b)).fetchone()[0]
            return left+(right-left)*(at-a)
        return {'count':n,'min':lo,'p50':q(.5) if n else None,'p95':q(.95) if n else None,'max':hi}
    def finish(self,wall,budget):
        self.c.set_progress_handler(lambda:int(time.monotonic()>budget.until),10000)
        self.c.execute('CREATE INDEX q_g_v ON q(g,contained,v)');self.c.execute('CREATE INDEX interval_g_s ON intervals(g,s,e)')
        result=[]
        for key,g in self.groups.items():
            budget.check();active=0;right=None
            for start,end in self.c.execute('SELECT s,e FROM intervals WHERE g=? ORDER BY s,e',(key,)):
                active+=end-max(start,right or start) if right is None or end>right else 0
                right=max(right or end,end)
            g=dict(g,statuses=dict(g['statuses']),batch_distribution=dict(g['batch_distribution']),
                   successful_value=self.quantile(key),wholly_contained_successful_value=self.quantile(key,True),
                   mode_wall_seconds=wall[g['mode']],active_service_union_seconds=active/1e9 if g['metric']=='service' else None)
            g['successful_units_per_mode_wall_second']=g['successful_batch_units']/wall[g['mode']] if wall[g['mode']]>0 else None
            g['successful_units_per_active_service_second']=(g['successful_batch_units']/(active/1e9)
                if active and g['metric']=='service' else None)
            if g['metric']!='service':g['successful_units_per_active_service_second']=None
            result.append(g)
        return sorted(result,key=lambda x:(x['mode'],x['workload'],x['metric'],str(x['pid'])))


def epoch_windows(c,run_id,budget,mask,from_epoch,to_epoch,diagnostics):
    begin={};completed={};phase={};phase_spans=[]
    sql="SELECT id,ts_ns,actor_id,pid,kind,details_json FROM events WHERE run_id=? AND workload IS NULL AND kind IN ('split_serial_phase_begin','split_serial_phase_end','synchronized_epoch_completed') ORDER BY ts_ns,id"
    for row in budget.rows(c,sql,(run_id,)):
        d=base.details(row,diagnostics);step=d.get('policy_step',d.get('global_step'))
        if not isinstance(step,int) or isinstance(step,bool):continue
        if step<from_epoch or (to_epoch is not None and step>to_epoch):continue
        if row['kind']=='synchronized_epoch_completed':
            if step in completed:raise ValueError('Duplicate synchronized epoch marker')
            completed[step]=(row['ts_ns'],d);continue
        model=d.get('model');rank=d.get('rank');key=(step,model,rank,row['pid'])
        if row['kind'].endswith('_begin'):
            begin[step]=min(begin.get(step,row['ts_ns']),row['ts_ns']);phase[key]=row['ts_ns']
        elif key in phase:
            phase_spans.append((step,model,row['actor_id'],row['pid'],phase.pop(key),row['ts_ns'],d.get('trained')))
    windows=[]
    for step,(end,d) in sorted(completed.items()):
        start=begin.get(step)
        if start is None or end<=start:diagnostics['complete_epoch_missing_valid_phase_start']+=1;continue
        ctrl,reward=d.get('trained_ctrl'),d.get('trained_reward')
        if ctrl is False and reward is False:mode='policy_only'
        elif ctrl is True and reward is True:mode='joint'
        else:diagnostics['mixed_or_missing_actual_epoch_schedule']+=1;continue
        if mask.overlap(start,end):diagnostics['whole_epochs_excluded_profile_overlap']+=1;continue
        windows.append({'step':step,'mode':mode,'start_ns':start,'end_ns':end,
            'wall_seconds':(end-start)/1e9,'reported_train_seconds':d.get('train_seconds'),
            'reported_sync_seconds':d.get('sync_seconds')})
    diagnostics['incomplete_epoch_steps']=len(set(begin)-set(completed))
    return Epochs(windows),phase_spans


def analyze(db,run_id,*,from_epoch=1,to_epoch=None,exclude_profiled=False,max_seconds=90,max_rows=2000000,max_pending=100000):
    path=Path(db).resolve();c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=3);c.row_factory=sqlite3.Row
    budget=Budget(max_seconds,max_rows);diagnostics=Counter();stats=Stats()
    try:
        c.execute('PRAGMA query_only=ON');c.execute('BEGIN')
        c.set_progress_handler(lambda:int(time.monotonic()>budget.until),10000)
        missing=base.EVENT_COLUMNS-base.columns(c,'events')
        if missing:raise ValueError('Missing events columns: '+','.join(sorted(missing)))
        profile,incomplete=base.profile_intervals(c,run_id,diagnostics)
        if exclude_profiled and incomplete:raise ValueError('Cannot exclude incomplete or invalid exact profile boundaries')
        mask=base.Mask([item['mask_ns'] for item in profile] if exclude_profiled else [])
        epochs,phases=epoch_windows(c,run_id,budget,mask,from_epoch,to_epoch,diagnostics)
        if not epochs.items:raise ValueError('No eligible complete recorded epochs')
        first,last=epochs.items[0]['start_ns'],epochs.items[-1]['end_ns']
        by_step={item['step']:item for item in epochs.items}
        for step,model,actor,pid,start,end,trained in phases:
            item=by_step.get(step)
            if item and model in ('policy','ctrl','reward') and isinstance(trained,bool):
                stats.add(item['mode'],model+'_train','phase_wall_ms' if trained else 'phase_no_update_wall_ms',
                          (end-start)/1e6,actor=actor,pid=pid)
        for item in epochs.items:
            for field in ('reported_train_seconds','reported_sync_seconds'):
                if isinstance(item[field],(int,float)):stats.add(item['mode'],'driver',field.replace('_seconds','_ms'),item[field]*1000)
        event_columns=base.columns(c,'events')
        optional=['request_id','submitted_ns','queue_mean_ns','queue_p95_ns','queue_max_ns','peak_bytes','model_version']
        selection=base.SELECT_EVENT+','+','.join(name if name in event_columns else 'NULL AS '+name for name in optional)
        pending=OrderedDict();latest_versions={};prefetch={};prefetch_out={};episodes=Counter();sampled_peaks={};rpc_counts=Counter();queue_coverage=Counter()
        def integrate(owner,end):
            prev=prefetch.get(owner)
            if prev is None or prev['closed']:return
            for item,left,right in epochs.overlaps(prev['ts'],end):
                out=prefetch_out.setdefault((item['mode'],owner),{'coverage_ns':0,'state_ns':Counter(),'observer_errors_max':0})
                out['coverage_ns']+=right-left
                for name in ('ready_batches','pending_fetches','in_use_batches','empty_waiters'):
                    out['state_ns'][name]+=(right-left)*prev[name]
                out['observer_errors_max']=max(out['observer_errors_max'],prev['observer_errors'])
        service_workloads=tuple(sorted(SERVICES|{'cpu_wm_windows'}))
        sql='SELECT '+selection+" FROM events WHERE run_id=? AND (ts_ns<=? OR (start_ns<? AND end_ns>?)) AND (kind IN ('request_submit','request_enqueue','batch_dispatch','client_wait','optimizer_update','data_fetch','data_prepare','cpu_replay_prefetch_state','eval_episode_completed') OR (kind='service' AND workload IN ("+','.join('?' for _ in service_workloads)+'))) ORDER BY ts_ns,id'
        for row in budget.rows(c,sql,(run_id,last,last,first)+service_workloads):
            d=base.details(row,diagnostics);kind=row['kind'];work=row['workload'];at=epochs.at(row['end_ns'] if kind in ('service','client_wait','data_fetch','data_prepare') and row['end_ns'] is not None else row['ts_ns'])
            if kind=='optimizer_update' and work in MODELS and d.get('data_parallel_rank')==0 and d.get('successful_optimizer_steps',0)>0:
                try:latest_versions.setdefault(MODELS[work],[]).append((row['ts_ns'],int(row['model_version'])))
                except (TypeError,ValueError):diagnostics['missing_trainer_version']+=1
                if at:stats.add(at['mode'],work,'global_optimizer_updates',None,units=d['successful_optimizer_steps'])
            elif kind=='request_submit' and work in INFER and d.get('method')=='request':
                rpc_counts[(work,'submitted')]+=1
            elif kind=='request_enqueue' and work in INFER:
                key=(row['actor'],row['actor_id'],row['pid'],row['request_id'])
                if row['request_id'] is None:diagnostics['enqueue_without_request_id']+=1;continue
                pending[key]=(row['start_ns'],row['submitted_ns'],workflow(d.get('workflow')))
                if len(pending)>max_pending:pending.popitem(last=False);diagnostics['queue_correlation_cache_evictions']+=1
            elif kind=='batch_dispatch' and work in INFER:
                ids=d.get('request_ids',[])
                if not isinstance(ids,list):ids=[];diagnostics['invalid_dispatch_request_ids']+=1
                diagnostics['dispatched_requests']+=row['actual_batch'] or 0
                dispatch_epoch=epochs.at(row['start_ns'] or row['ts_ns'])
                if dispatch_epoch:queue_coverage[(dispatch_epoch['mode'],work,'dispatched')]+=row['actual_batch'] or 0
                if len(ids)!=(row['actual_batch'] or 0):diagnostics['dispatch_id_count_mismatch']+=1
                for request in ids:
                    record=pending.pop((row['actor'],row['actor_id'],row['pid'],request),None)
                    if record is None:diagnostics['dispatch_without_matching_enqueue']+=1;continue
                    diagnostics['queue_requests_correlated']+=1
                    arrival,submitted,tag=record;end=row['start_ns'] or row['ts_ns'];at=epochs.at(end)
                    if at and isinstance(arrival,int) and end>=arrival:
                        queue_coverage[(at['mode'],work,'correlated')]+=1
                        stats.add(at['mode'],work,'actor_queue_wait_ms',(end-arrival)/1e6,
                            contained=bool(epochs.contained(arrival,end)),actor=row['actor_id'],pid=row['pid'])
                        stats.add(at['mode'],work,'actor_queue_wait_ms_workflow_'+tag,(end-arrival)/1e6,
                            contained=bool(epochs.contained(arrival,end)))
                        if isinstance(submitted,int) and end>=submitted:
                            stats.add(at['mode'],work,'submission_to_dispatch_ms',(end-submitted)/1e6,
                                contained=bool(epochs.contained(submitted,end)),actor=row['actor_id'],pid=row['pid'])
            elif kind=='cpu_replay_prefetch_state':
                owner=(d.get('model'),d.get('rank'),row['actor_id'],row['pid']);integrate(owner,row['ts_ns'])
                fields=('ready_batches','pending_fetches','in_use_batches','empty_waiters')
                if d.get('capacity_batches')==1 and all(type(d.get(k)) is int and d[k] in (0,1) for k in fields) and isinstance(d.get('closed'),bool):
                    prefetch[owner]={**{k:d[k] for k in fields},'ts':row['ts_ns'],'closed':d['closed'],'observer_errors':d.get('observer_errors',0)}
                else:prefetch.pop(owner,None);diagnostics['invalid_prefetch_state']+=1
            elif kind in ('service','client_wait','data_fetch','data_prepare'):
                start,end=row['start_ns'],row['end_ns']
                if not isinstance(start,int) or not isinstance(end,int) or end<start:
                    diagnostics['invalid_interval']+=1;continue
                if kind=='service' and work in SERVICES and row['status']=='ok':
                    for item,left,right in epochs.overlaps(start,end):stats.active(item['mode'],work,left,right,row['actor_id'],row['pid'])
                if kind=='client_wait' and work in INFER and d.get('method')=='request' and end<=last:
                    rpc_counts[(work,'reply_'+str(row['status']))]+=1
                if not at:diagnostics['completed_intervals_outside_eligible_epochs']+=1;continue
                contained=bool(epochs.contained(start,end));duration=row['duration_ns']
                value=duration/1e6 if isinstance(duration,(int,float)) and duration>=0 else None
                if kind=='service' and work in SERVICES:
                    units=row['actual_batch']
                    if type(units) is not int or units<0:diagnostics['invalid_actual_batch']+=1;continue
                    stats.add(at['mode'],work,'service',value,units=units,status=row['status'],contained=contained,
                        actor=row['actor_id'],pid=row['pid'],peak=row['peak_bytes'])
                    if work in INFER:
                        counts=d.get('workflow_counts',{})
                        if (not isinstance(counts,dict) or any(type(v) is not int or v<0 for v in counts.values())
                                or sum(counts.values())!=units):
                            counts={'unknown':units};diagnostics['service_workflow_missing_or_invalid_batches']+=1
                        normalized=Counter()
                        for tag,count in counts.items():normalized[workflow(tag)]+=count
                        for tag,count in normalized.items():
                            if count:stats.add(at['mode'],work,'service_workflow_'+tag,None,units=count,
                                status=row['status'],contained=contained)
                elif kind=='client_wait' and work in INFER and d.get('method')=='request':
                    stats.add(at['mode'],work,'request_rpc_latency_ms',value,status=row['status'],contained=contained,
                        actor=row['actor_id'],pid=row['pid'])
                    stats.add(at['mode'],work,'request_rpc_latency_ms_caller_'+str(row['actor']),value,
                        status=row['status'],contained=contained,actor=row['actor_id'],pid=row['pid'])
                    tag=workflow(d.get('workflow'))
                    stats.add(at['mode'],work,'request_rpc_latency_ms_workflow_'+tag,value,
                        status=row['status'],contained=contained)
                    versions=d.get('served_versions',{})
                    if not isinstance(versions,dict):versions={};diagnostics['invalid_served_versions']+=1
                    for model,served in versions.items():
                        if model in latest_versions and type(served) is int:
                            timeline=latest_versions[model];index=bisect_right(timeline,(end,float('inf')))-1
                            if index<0:diagnostics['reply_lag_no_prior_successful_trainer_version']+=1;continue
                            lag=timeline[index][1]-served
                            if lag<0:diagnostics['negative_reply_version_lag']+=1
                            stats.add(at['mode'],work,'latest_trainer_minus_reply_served_version',lag,status=row['status'],contained=contained)
                elif kind in ('data_fetch','data_prepare'):
                    stats.add(at['mode'],row['actor'],kind+'_ms',value,status=row['status'] or 'ok',contained=contained,
                        actor=row['actor_id'],pid=row['pid'])
                elif kind=='service' and work=='cpu_wm_windows' and row['actor']=='rollout' and row['status']=='ok':
                    episodes[(at['mode'],'real_rollout_postprocessing_completed')]+=1
            elif kind=='eval_episode_completed' and at:
                episodes[(at['mode'],'eval_episode_completed')]+=1
        for owner in list(prefetch):integrate(owner,last)
        queue=[]
        for (mode,owner),out in prefetch_out.items():
            queue.append({'mode':mode,'model':owner[0],'rank':owner[1],'actor_id':owner[2],'pid':owner[3],
                'coverage_seconds':out['coverage_ns']/1e9,'state_seconds':{k:v/1e9 for k,v in out['state_ns'].items()},
                'fraction_of_mode_wall_covered':out['coverage_ns']/(epochs.wall[mode]*1e9) if epochs.wall[mode] else None,
                'state_fraction_of_observed_lifecycle':{k:v/out['coverage_ns'] if out['coverage_ns'] else None for k,v in out['state_ns'].items()},
                'observer_errors_max':out['observer_errors_max'],'right_censored':not prefetch.get(owner,{}).get('closed',False)})
        resident={}
        if {'run_id','ts_ns','replay','actor_id','resident_count','capacity'}<=base.columns(c,'replay_snapshots'):
            for row in budget.rows(c,'SELECT ts_ns,replay,actor_id,resident_count,capacity FROM replay_snapshots WHERE run_id=? AND ts_ns>=? AND ts_ns<?',(run_id,first,last)):
                at=epochs.at(row['ts_ns'])
                if not at:continue
                key=(at['mode'],row['replay'],row['actor_id'])
                out=resident.setdefault(key,{'snapshots':0,'minimum_resident':row['resident_count'],'maximum_resident':row['resident_count'],'capacity':row['capacity']})
                out['snapshots']+=1;out['minimum_resident']=min(out['minimum_resident'],row['resident_count']);out['maximum_resident']=max(out['maximum_resident'],row['resident_count'])
        if {'run_id','ts_ns','payload_json'}<=base.columns(c,'system_samples'):
            for row in budget.rows(c,'SELECT ts_ns,payload_json FROM system_samples WHERE run_id=? AND ts_ns>=? AND ts_ns<? ORDER BY ts_ns',(run_id,first,last)):
                at=epochs.at(row['ts_ns'])
                if not at:continue
                try:payload=json.loads(row['payload_json'])
                except (ValueError,TypeError):diagnostics['invalid_system_sample']+=1;continue
                for gpu in payload.get('gpus',[]):
                    key=(at['mode'],gpu.get('physical_gpu'))
                    out=sampled_peaks.setdefault(key,{'samples':0,'first_sample_ns':row['ts_ns'],'last_sample_ns':row['ts_ns'],'sampled_peak_used_bytes':None,'max_gpu_busy_percent':None});out['samples']+=1;out['last_sample_ns']=row['ts_ns']
                    for source,target in [('memory_used_bytes','sampled_peak_used_bytes'),('gpu_busy_percent','max_gpu_busy_percent')]:
                        if isinstance(gpu.get(source),(int,float)):out[target]=max(out[target] or 0,gpu[source])
        cycles=[]
        for step,item in sorted(by_step.items()):
            members=[by_step.get(n) for n in range(step-4,step+1)]
            if step%5==0 and all(members) and [x['mode'] for x in members]==['policy_only']*4+['joint']:
                cycles.append({'first_epoch':step-4,'last_epoch':step,'wall_seconds':sum(x['wall_seconds'] for x in members),
                               'policy_only_wall_seconds':sum(x['wall_seconds'] for x in members[:4]),'joint_wall_seconds':item['wall_seconds']})
        result={'schema_version':1,'run_id':run_id,'db':str(path),'input_read_only':True,'request_table_joins':False,
            'epoch_window_semantics':'Recorded earliest per-rank phase begin to synchronized epoch completion. Whole selected complete epochs only; gaps/fill/finalization and incomplete tail excluded.',
            'epochs':epochs.items,'mode_wall_seconds':epochs.wall,'profile_windows':profile,'whole_profiled_epochs_excluded':exclude_profiled,
            'eligible_complete_5_to_1_to_1_cycles':cycles,
            'statistics':stats.finish(epochs.wall,budget),'prefetch':{'status':'available' if queue else 'not_collected','owners':queue},
            'sampled_physical_gpu_peaks':[dict(mode=key[0],physical_gpu=key[1],**value) for key,value in sampled_peaks.items()],
            'Replay_resident_snapshots':{'status':'available' if resident else 'not_collected','summaries':[
                dict(mode=key[0],replay=key[1],actor_id=key[2],**value) for key,value in resident.items()]},
            'real_episode_production':[{'mode':mode,'kind':kind,'count':count,'per_mode_wall_hour':count*3600/epochs.wall[mode] if epochs.wall[mode] else None} for (mode,kind),count in episodes.items()],
            'diagnostics':dict(diagnostics,queue_cache_remaining_inflight=len(pending),scanned_rows=budget.count),
            'recorded_request_RPC_up_to_last_epoch':{work:{'submitted':rpc_counts[(work,'submitted')],
                'reply_ok':rpc_counts[(work,'reply_ok')],'reply_error':rpc_counts[(work,'reply_error')],
                'submitted_minus_replies':rpc_counts[(work,'submitted')]-rpc_counts[(work,'reply_ok')]-rpc_counts[(work,'reply_error')]}
                for work in sorted(INFER)},
            'queue_request_coverage_by_mode':[{'mode':mode,'workload':work,'dispatched_requests':queue_coverage[(mode,work,'dispatched')],
                'exact_queue_requests':queue_coverage[(mode,work,'correlated')],
                'fraction':queue_coverage[(mode,work,'correlated')]/queue_coverage[(mode,work,'dispatched')] if queue_coverage[(mode,work,'dispatched')] else None}
                for mode in epochs.wall for work in sorted(INFER)],
            'inference_roles':{'policy_infer':'Real rollout actions, held-out evaluation actions, imagined actions and bootstrap. Exact workflow breakdown requires explicit callsite tags; missing tags stay unknown, never inferred from episode_id.',
                'ctrl_infer':'request predicts imagined future observations. Non-request initialization RPCs are excluded from request latency.',
                'reward_infer':'request scores imagined observations for reward/end prediction; real episode labels come from environment reward.'},
            'semantics':[
                'Service request/exposure counts use successful actual_batch once. Micro, macro-update, client latency, queue and phase records are separate metrics and must never be added together.',
                'service_workflow_* counts partition the original successful inference batch using propagated workflow_counts. No batch duration is attributed to an individual workflow; mixed batches do not create extra requests.',
                'Throughput numerators are completion cohorts. Cross-epoch/inflight batches are retained and labeled; duration quantiles are also provided for whole-epoch-contained cohorts. Source requests are never changed or discarded.',
                'Active throughput uses clipped union of successful service wall intervals in that mode. For cross-boundary batches its completion numerator and active denominator describe different cohorts; use mode-wall completion throughput as the primary rate.',
                'Exact actor queue quantiles correlate request_enqueue to dispatch IDs in a bounded in-flight Python map, not a request-table join. Missing/evicted metadata reduces declared queue coverage, never the successful service throughput numerator.',
                'Latency is request RPC submission to deserialized reply, by reply completion. Model lag is latest observed successful rank0 trainer version at reply minus served version; includes reply time and broadcast lag, not replay sample age.',
                'Real production is rollout cpu_wm_windows successful postprocessing completion, a proxy after environment termination. Eval episodes are separate; no imagined rollout counts are treated as real episodes.',
                'Ready/pending/in-use/empty-wait are overlapping raw CPU one-slot states. Pending is not proof of missing resident Replay; in-use spans the full trainer phase. Fractions must not be summed to100%.',
                'Phase total, data_fetch/data_prepare and driver synchronization are recorded host times, not separable kernel work or precise scheduling wait. Actual take wait is the explicit empty-wait state integral.',
                'Allocated service peaks and sampled physical-card memory peaks have different scopes. Sampled maxima do not guarantee instantaneous headroom, and GPU busy100 is not SM Active or occupancy.',
                'Profiling excludes whole epochs intersecting exact range+export marker intervals; outer profiler overhead and online trajectory perturbation remain. No unprofiled causal or resource-share claim is made.'],
            'not_measured':{'kernel_GPU_share':'requires Nsight real kernels','SM_Active_occupancy':'requires independently enabled hardware metrics',
                'Replay_resident_time_occupancy':'raw prefetch state is distinct; not inferred','exact_environment_termination_rate':'postprocessing completion proxy only'}}
        return result
    finally:stats.c.close();c.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True);p.add_argument('--run-id',required=True);p.add_argument('--output',required=True)
    p.add_argument('--from-epoch',type=int,default=1);p.add_argument('--to-epoch',type=int)
    p.add_argument('--exclude-profiled',action='store_true');p.add_argument('--max-seconds',type=float,default=90)
    p.add_argument('--max-rows',type=int,default=2000000);p.add_argument('--max-pending',type=int,default=100000)
    a=p.parse_args();output=analyze(a.db,a.run_id,from_epoch=a.from_epoch,to_epoch=a.to_epoch,
        exclude_profiled=a.exclude_profiled,max_seconds=a.max_seconds,max_rows=a.max_rows,max_pending=a.max_pending)
    Path(a.output).write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
