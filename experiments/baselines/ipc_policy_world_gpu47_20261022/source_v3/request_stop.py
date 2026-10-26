"""Explicit user boundary stop for this manual run; never sends process signals."""
import argparse
import json
import os
from pathlib import Path

import gpu47_safety as safe
import runner


def request(data, run_id):
    if os.getuid()!=1004:
        raise RuntimeError('Wrong UID')
    if data.resolve()!=(safe.MASTER/'formal_v3').resolve():
        raise RuntimeError('Stop request must target this new master/formal_v3')
    state=runner.load(data)
    if (state.get('uid')!=1004 or state.get('run_id')!=run_id or state.get('mode')!='manual'
        or state.get('until_manual_stop') is not True or state.get('deadline_unix') is not None
        or state.get('stop_on_target') is not False or state.get('state') not in {'starting','running'}):
        raise RuntimeError('Manual state identity or stop policy does not match')
    expected=state.get('supervisor') or {}
    current=safe.proc_identity(expected.get('pid',-1))
    if (expected.get('uid')!=1004 or not expected.get('birth_ticks') or not current or current.get('uid')!=1004
        or current.get('state') in {'Z','X'} or not safe.same_process(expected,current)):
        raise RuntimeError('Supervisor UID/PID/birth identity does not match')
    if not (data/'training').is_dir():
        raise RuntimeError('Training directory not ready')
    runner.stop(data,'user_requested_stop')
    flag=json.loads((data/'training/stop.flag').read_text())
    if flag.get('run_id')!=run_id or flag.get('uid')!=1004 or flag.get('reason')!='user_requested_stop':
        raise RuntimeError('A different stop reason already exists; user stop was not claimed')
    return {'run_id':run_id,'mode':'manual','stop_requested':True,'stop_file':str(data/'training/stop.flag'),
            'requested_ns':flag['requested_ns'],'deadline_unix':None,'process_signals_sent':False,
            'supervisor':{key:expected[key] for key in ('pid','uid','birth_ticks')}}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=safe.MASTER/'formal_v3')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    print(json.dumps(request(args.data,args.run_id),ensure_ascii=False))
