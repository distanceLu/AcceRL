"""Backend for the server's single run.sh. All GPU work goes through its existing queue."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
HOME = Path("/mnt/data/lcx1/dong/WEAVER")
EXPERIMENTS = HOME / "experiments"
DATA = HOME.parent / "need_data"
QUEUE = DATA / "experiment_queue_gpu47_20261023"
AGENT = QUEUE / "source_v4_postreboot/agent.py"
PYTHON = "/mnt/data/lcx3/envs/merged-env/bin/python"
PARENT = DATA / "weaver_cycle_restore_gpu47_20261105_v1"
PROFILES = json.loads((HERE / "profiles.json").read_bytes())
CHECKOUT = HERE.parents[1]
RUNTIME = HOME / ".git-runtime"
LEGACY_CLI = HOME / ".launcher/8daa6667c03361e298fc81233978f2fd9b76ee9ed3803726e8af7d218a1aef14/cli.py"
sys.path.insert(0, str(CHECKOUT / "weaver_run"))
from git_version import BRANCH, published, revision, pin_worktree
LABEL = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write((json.dumps(value, indent=2) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def cpu(argv, *, cwd=None, stdin=None, timeout=180, env=None):
    clean = {k:v for k,v in os.environ.items() if not k.startswith(("ACCERL_", "ACCE_", "WEAVER_", "NSYS_", "CUDA_","PYTHON")) and k != "LD_PRELOAD"}
    clean.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                 MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    clean.update(env or {})
    result = subprocess.run(argv, input=stdin, cwd=cwd, env=clean, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr + "\n" + result.stdout)[-4000:])
    return result.stdout


def queue_rows(sql, params=()):
    with sqlite3.connect((QUEUE/"queue.sqlite").as_uri()+"?mode=ro", uri=True, timeout=3) as c:
        c.row_factory=sqlite3.Row
        return [dict(r) for r in c.execute(sql,params)]


def lookup(sid):
    rows=queue_rows("select * from jobs where submission_id=?", (sid,))
    return rows[0] if rows else None


def active_jobs():
    return queue_rows("select id,state,submission_id from jobs where state in ('queued','running','blocked')")


def ready():
    state=json.loads((QUEUE/"state.json").read_bytes())
    require(not state.get("error") and 0 <= time.time_ns()-state["observed_ns"] < 90_000_000_000,
            "队列遥测已过期或有错误，请稍后再试")
    require(not active_jobs() and not state["external_process_count"] and not state["mps"]["clients"],
            "GPU4–7 或队列正被占用，本次没有提交")


def handoff_verified():
    receipt=json.loads((PARENT/"queue_receipt.json").read_bytes())
    require(receipt.get("run_id")=="weaverintegrated_1105_140654_70115c34" and
            receipt.get("job_id")=="283c22feb6d547b5a44dcd8cd2918640" and
            receipt.get("training_acceptance",{}).get("passed") is True and
            receipt.get("all_owned_exited") is True and receipt.get("evaluation",{}).get("skipped") is True,
            "前一轮训练的保存、退出或跳过最终评估记录不匹配")
    for job in (receipt["job_id"],"4e1c37dba14d44108d0efc37493c2fee"):
        require(queue_rows("select state from jobs where id=?",(job,))==[{"state":"succeeded"}],
                "前序训练或 P0 回放队列记录不匹配")


def make_spec(profile, label, root, stage, checkout=RUNTIME):
    require(profile=="restore", "This branch manages the original training profile")
    commit=revision(checkout)
    source=checkout/"weaver_run/engine"
    wrapper=checkout/"weaver_run/reference_site/queue_wrapper/queue_entry.py"
    from launch import expand, settings
    template=json.loads((checkout/"weaver_run/recipe/world_backend_config.json").read_bytes())
    config=expand(template,dict(settings(),weaver=str(checkout/"weaver_run/vendor/weaver")))
    json_write(stage/"git-world.json",config)
    json_write(stage/'visualization_config.json',json.loads((checkout/'weaver_run/recipe/visualization.json').read_bytes()))
    mps=(PARENT/"mps.json").read_bytes()
    require(digest(mps)==PROFILES[profile]["mps_receipt"], "MPS receipt differs")
    (stage/"mps.json").write_bytes(mps)
    argv=["/usr/bin/env","PYTHONNOUSERSITE=1","PYTHONDONTWRITEBYTECODE=1",PYTHON,str(wrapper),
          "--source",str(source),"--root",str(root),"--git-commit",commit,
          "--native-gpu-preflight"]
    job=dict(name="WEAVER "+profile+" "+label,submission_id="weaver-run-"+label,
             cwd=str(source),argv=argv,manual=True,timeout_seconds=0)
    record=dict(schema=2,profile=profile,label=label,root=str(root),source=str(source),
                recipe_baseline="1793355532",real_collection_action_limit=None,
                real_training_action_horizon=150,task_action_horizon=150,
                base_task_action_horizon=220,final_evaluation_action_horizon=220,
                training_horizon_failure_reward_target=0.0,
                training_horizon_failure_labels_valid=True,
                training_horizon_failure_is_terminal=True,
                training_horizon_failure_bootstrap=False,
                stop_policy="explicit_user_stop_or_existing_budget_or_fault_guard",
                git_branch=BRANCH,git_commit=commit,job=job)
    json_write(stage/"experiment.json",record)
    return record


def check_record(record):
    out=cpu([*record["job"]["argv"],"--validate-only"],cwd=record["source"])
    result=json.loads(out.strip().splitlines()[-1])
    validation=result["validation"]
    require(result.get("valid") and validation["gpu_access_performed"] is False
            and validation["git_commit"]==record["git_commit"], "Queue CPU admission failed")


def create(job, queue):
    payload=json.dumps({"op":"create","job":job})
    reply=json.loads(cpu([PYTHON,str(AGENT),"api","--root",str(queue)],stdin=payload,timeout=40))
    require(reply.get("ok"),"队列拒绝了请求")
    item=reply["job"]
    require(all(item[k]==v for k,v in job.items()),"队列回执与请求不一致")
    return item


def recorded(label):
    require(LABEL.fullmatch(label), "实验名只允许 1–40 个小写字母、数字、下划线或连字符")
    root=EXPERIMENTS/label
    require(root.resolve()==root,"输出目录经过重定向")
    r=json.loads((root/"experiment.json").read_bytes())
    require(r["label"]==label and r["root"]==str(root) and r["job"]["submission_id"]=="weaver-run-"+label,
            "实验标识不匹配")
    return r


def latest():
    p=EXPERIMENTS/".last.json"
    return json.loads(p.read_bytes()) if p.is_file() else None


def summary(record, job):
    result=dict(id=record["label"],profile=record["profile"],output=record["root"],
                job=job["id"] if job else None,state=job["state"] if job else "prepared")
    if record.get("git_commit"):
        result["git_commit"]=record["git_commit"]
    p=Path(record["root"])/"formal_v3/runnerstatus.json"
    if p.is_file():
        state=json.loads(p.read_bytes())
        result.update(run=state.get("run_id"),training=state.get("state"))
    return result


def save_submission(root, item):
    receipt={k:item[k] for k in ("id","submission_id","created_ns","manual","timeout_seconds")}
    receipt["manual"]=bool(receipt["manual"])
    path=root/"submission.json"
    if path.exists():
        require(json.loads(path.read_bytes())==receipt,"已保存的提交回执不一致")
    else:
        json_write(path,receipt)


def start(profile,label):
    latest_label=latest()
    if not label and latest_label:
        last=recorded(latest_label)
        prior=lookup(last["job"]["submission_id"])
        if last["profile"]==profile and (prior is None or prior["state"] in ("queued","running")):
            label=latest_label
    label=label or profile+"-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid4().hex[:6]
    require(LABEL.fullmatch(label),"无效的实验名")
    root=EXPERIMENTS/label
    if os.path.lexists(root):
        r=recorded(label)
        require(r["profile"]==profile,"实验名已绑定另一配置")
        prior=lookup(r["job"]["submission_id"])
        if prior:
            require(json.loads(prior["argv_json"])==r["job"]["argv"] and all(prior[k]==r["job"][k]
                    for k in ("cwd","name","manual","timeout_seconds")),
                    "已有队列作业参数不匹配")
            save_submission(root,prior)
            return summary(r,prior)
    else:
        require(lookup("weaver-run-"+label) is None,"该提交编号已有其他记录")
        ready()
        require(CHECKOUT==(HOME/"code").resolve(), "Start new runs from the maintained WEAVER/code checkout.")
        pin_worktree(CHECKOUT, RUNTIME, revision(CHECKOUT))
        with tempfile.TemporaryDirectory(dir=EXPERIMENTS,prefix=".staging-") as staging:
            staged=Path(staging)/"output"
            staged.mkdir(mode=0o700)
            r=make_spec(profile,label,root,staged)
            staged.rename(root)
    if not r.get("git_commit"):
        raise ValueError("Use a new label for a Git-backed experiment")
    ready()
    require(revision(CHECKOUT)==revision(RUNTIME,r["git_commit"]),
            "Prepared experiment belongs to another commit; use a fresh label")
    with tempfile.TemporaryDirectory(dir=EXPERIMENTS,prefix=".verify-") as temp:
        expected=make_spec(profile,label,root,Path(temp))
        require(r==expected and (root/"git-world.json").read_bytes()==(Path(temp)/"git-world.json").read_bytes()
                and digest((root/"mps.json").read_bytes())==PROFILES[profile]["mps_receipt"],
                "准备记录与冻结配置不一致")
    ready()
    handoff_verified()
    print(f"正在校验 {profile}，实验名 {label}，输出 {root}",file=sys.stderr,flush=True)
    check_record(r)
    ready()
    require(published(CHECKOUT, push=True)==r["git_commit"], "Published branch moved during admission")
    revision(RUNTIME,r["git_commit"])
    ready()
    json_write(EXPERIMENTS/".last.json",label)
    item=create(r["job"],QUEUE)
    save_submission(root,item)
    return summary(r,item)


def offline_check(profile):
    with tempfile.TemporaryDirectory(prefix=".cpu-check-",dir=EXPERIMENTS) as d:
        temp=Path(d)
        root=temp/"output"
        root.mkdir()
        r=make_spec(profile,"validation-"+profile,root,root,checkout=CHECKOUT)
        check_record(r)
        # Production agent.py is pinned to its live root. Exercise the same
        # QueueStore.create used by the create API in a private workerless DB.
        private=temp/"queue"
        private.mkdir()
        sys.path.insert(0,str(AGENT.parent))
        from store import QueueStore
        store=QueueStore(private)
        a=store.create(**r["job"])
        b=store.create(**r["job"])
        require(a["id"]==b["id"] and a["state"]=="queued" and all(a[k]==v for k,v in r["job"].items()),
                "隔离队列提交或去重测试失败")
        return dict(profile=profile,cpu_admission="passed",isolated_queue="passed",production_submission=False)


def stop(record,job):
    require(job and record["profile"]!="p0" and job["state"]=="running","只接受本入口正在运行的手动训练")
    require(json.loads(job["argv_json"])==record["job"]["argv"] and job["cwd"]==record["job"]["cwd"],"作业参数不符")
    root=Path(record["root"])
    receipt=json.loads((root/"queue_receipt.json").read_bytes())
    state=json.loads((root/"formal_v3/runnerstatus.json").read_bytes())
    require(receipt["job_id"]==job["id"] and receipt["run_id"]==state["run_id"],"训练身份不符")
    for ident in (json.loads(job["identity_json"]),receipt["supervisor"],state["supervisor"]):
        proc=Path("/proc")/str(ident["pid"])
        stat=(proc/"stat").read_text().rsplit(")",1)[1].split()
        require(proc.stat().st_uid==ident["uid"]==1004 and int(stat[19])==ident["birth_ticks"] and stat[0] not in ("X","Z"),"PID/UID/启动标识不符")
    require(all(receipt["supervisor"][k]==state["supervisor"][k] for k in ("pid","uid","birth_ticks")),"主管身份不符")
    source=RUNTIME/"weaver_run/engine"
    require(str(source)==record["source"] and revision(RUNTIME,record["git_commit"])==state["git_commit"],
            "Running Git revision differs from the stop request")
    return json.loads(cpu([PYTHON,str(source/"request_stop.py"),"--data",str(root/"formal_v3"),"--run-id",state["run_id"]],
                          cwd=source,env={"ACCERL_EXPERIMENT_ROOT":str(root)}))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='view':
        p=argparse.ArgumentParser(prog='./run.sh view',description='CPU-only read-only three-view dashboard')
        p.add_argument('--id',dest='label')
        p.add_argument('--root',type=Path)
        p.add_argument('--port',type=int,default=8765)
        opts=p.parse_args(sys.argv[2:])
        label=opts.label or (latest() if opts.root is None else None)
        require(opts.root is not None or (label and LABEL.fullmatch(label)),'Choose an existing experiment with --id or --root')
        sys.path.insert(0,str(CHECKOUT/'weaver_run/engine'))
        from visualization import serve
        serve(opts.root or EXPERIMENTS/label,opts.port)
        return
    if len(sys.argv)>1 and sys.argv[1] in ('diagnose','diagnose-status'):
        p=argparse.ArgumentParser(prog='./run.sh '+sys.argv[1]);p.add_argument('--id',dest='label')
        opts=p.parse_args(sys.argv[2:])
        require((os.getuid(),socket.gethostname())==(1004,'h20-xyff52ab-0001') and
                (HOME/'.owner').read_text()=='WEAVER server catalog v1\n','Unexpected diagnostic server')
        sys.path.insert(0,str(CHECKOUT))
        from weaver_run.diagnostics.queue_panel import submit
        with (EXPERIMENTS/'.launch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            print(json.dumps(submit(sys.modules[__name__],opts.label,status=sys.argv[1]=='diagnose-status'),indent=2))
        return
    p=argparse.ArgumentParser(prog="./run.sh",description="WEAVER GPU4–7 | 无参数启动 restore；check 只做 CPU 验收",
        epilog="./run.sh  |  ./run.sh list  |  ./run.sh check restore  |  ./run.sh original --id trial01  |  ./run.sh status  |  ./run.sh stop")
    p.add_argument("args",nargs="*",metavar="动作或配置")
    p.add_argument("--id",dest="label",help="自定义实验名；重试时使用相同名字")
    opts=p.parse_args()
    args=opts.args.copy()
    action=args.pop(0) if args and args[0] in ("list","check","start","status","stop") else "start"
    profile=args.pop(0) if args else "restore"
    profile="restore" if profile=="original" else profile
    if action=="check" and profile not in PROFILES and profile!="all":
        os.execv(PYTHON,[PYTHON,"-I","-B",str(LEGACY_CLI),*sys.argv[1:]])
    if args or (profile not in PROFILES and not (action=="check" and profile=="all")):
        p.error("未知配置或多余参数，使用 ./run.sh list")
    if action in ("list","status","stop") and len(opts.args)>1:
        p.error("使用 --id 选择实验")
    if opts.label and (action in ("list","check") or not LABEL.fullmatch(opts.label)):
        p.error("此处不接受该实验名")
    if action=="list":
        for name,s in PROFILES.items():
            print(f"{name:12} {s['description']}")
        return
    require((os.getuid(),socket.gethostname())==(1004,"h20-xyff52ab-0001") and
            HOME.resolve()==HOME and (HOME/".owner").read_text()=="WEAVER server catalog v1\n","服务器工作区身份不符")
    EXPERIMENTS.mkdir(mode=0o700,exist_ok=True)
    require(EXPERIMENTS.resolve()==EXPERIMENTS and EXPERIMENTS.stat().st_uid==1004,"实验目录身份不符")
    if action=="check":
        for name in (PROFILES if profile=="all" else [profile]):
            print(json.dumps(offline_check(name),ensure_ascii=False),flush=True)
        return
    if action=="start":
        with (EXPERIMENTS/".launch.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result=start(profile,opts.label)
    else:
        label=opts.label or latest()
        if not label and action=="status":
            result={"state":"该入口尚无提交记录"}
        else:
            require(bool(label),"没有可停止的训练")
            r=recorded(label)
            if not r.get("git_commit"):
                os.execv(PYTHON,[PYTHON,"-I","-B",str(LEGACY_CLI),action,"--id",label])
            if action=="stop" and CHECKOUT!=RUNTIME:
                pinned=RUNTIME/"weaver_run/reference_site/cli.py"
                revision(RUNTIME,r["git_commit"])
                os.execv(PYTHON,[PYTHON,"-I","-B",str(pinned),action,"--id",label])
            j=lookup(r["job"]["submission_id"])
            result=stop(r,j) if action=="stop" else summary(r,j)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as error:
        sys.exit(f"run.sh: {error}")
