"""Single-entry, idempotent queue submission; never stops an existing job."""
import json,sys
from pathlib import Path
from .panel_common import spec,save


def submit(cli,label,status=False):
    label=label or 'learning-diagnostic-20261109-a'
    cli.require(cli.LABEL.fullmatch(label) is not None,'Invalid diagnostic label')
    root=cli.EXPERIMENTS/label;sid='weaver-diag-'+label
    cli.require(len(sid)<=64,'Submission ID is too long')
    found=cli.lookup(sid)
    if found or status:
        return dict(label=label,submission_id=sid,job_id=found['id'] if found else None,
                    state=found['state'] if found else 'not_submitted',output=str(root),
                    result=json.loads((root/'result.json').read_bytes()) if (root/'result.json').exists() else None)
    # Existing manual training is deliberately allowed. Its GPU lease stays intact.
    rows=cli.queue_rows("select id,cwd,argv_json from jobs where state in ('running','queued','blocked')")
    for row in rows:
        cli.require(str(cli.RUNTIME) not in row['cwd'] and str(cli.RUNTIME) not in row['argv_json'],
                    'Another queued/running job owns the Git runtime; cannot repin it')
    commit=cli.published(cli.CHECKOUT,push=False)
    cli.pin_worktree(cli.CHECKOUT,cli.RUNTIME,commit)
    cli.require(not root.exists(),'Diagnostic directory already exists without a queue receipt; review it before retrying')
    root.mkdir(mode=0o700)
    plan=spec();plan.update(git_commit=commit,output=str(root),submission_id=sid,
        native_python='/mnt/data/lcx1/dong/envs/weaver-libero-20261022/bin/python')
    runner=cli.RUNTIME/'weaver_run/diagnostics/panel_runner.py'
    argv=['/usr/bin/env','PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1',cli.PYTHON,str(runner),'--plan',str(root/'plan.json')]
    plan['job_argv']=argv;save(root/'plan.json',plan)
    try:
        import shutil
        cli.require(shutil.disk_usage(root).free>=20*1024**3,'Diagnostic needs at least 20 GiB free output storage')
        result=cli.cpu([*argv,'--validate-only'],cwd=cli.RUNTIME,timeout=300)
        plan['admission']=json.loads(result.strip().splitlines()[-1]);save(root/'plan.json',plan)
        cli.require(plan['admission']['cuda_initialized'] is False,'Admission must not access GPUs')
        cli.require(cli.published(cli.CHECKOUT)==commit,'Published branch changed during admission')
        cli.revision(cli.RUNTIME,commit)
        job=dict(name='WEAVER frozen WM RM gradients '+label,submission_id=sid,cwd=str(cli.RUNTIME),
                 argv=argv,manual=False,timeout_seconds=plan['max_seconds']+120)
        save(root/'experiment.json',dict(profile='frozen_learning_diagnostic',git_commit=commit,job=job,
            waiting_behind=[r['id'] for r in rows],no_training_stop_requested=True))
        item=cli.create(job,cli.QUEUE)
        save(root/'submission.json',dict(job_id=item['id'],submission_id=sid,git_commit=commit,state=item['state']))
        return dict(job_id=item['id'],submission_id=sid,state=item['state'],git_commit=commit,
                    output=str(root),waiting_behind=[r['id'] for r in rows],gpu_measurements_complete=False)
    except BaseException as e:
        save(root/'submission_error.json',dict(error=str(e)));raise
