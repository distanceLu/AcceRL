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
    return queue_rows("select id,state,submission_id from jobs where state in ('queued','running')")


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


def paths(profile):
    spec=PROFILES[profile]
    source=Path(spec["source"])
    wrapper=Path(spec["wrapper"]) if spec.get("wrapper") else HERE/"wrappers"/profile
    require(source.resolve(strict=True)==source and digest((source/"source_manifest.json").read_bytes())==spec["manifest"],
            "冻结源码路径或清单不匹配")
    for name,expected in spec.get("wrapper_hashes",{}).items():
        asset=wrapper/name
        require(asset.resolve(strict=True)==asset and digest(asset.read_bytes())==expected,
                f"启动包装器文件变化: {asset}")
    return spec,source,wrapper


def replay_program(bundle, root, sid):
    """Only rebind the replay's physical bundle, new output and queue identity."""
    original=(bundle/"model_replay.py").read_bytes()
    index=(bundle/"bundle.json").read_bytes()
    require(digest(index)==bundle.name and digest(original)==json.loads(index)["model_replay.py"], "P0 程序指纹不符")
    text=original.decode("utf8")
    for old,new in [
        ("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(bundle)!r})"),
        ("ROOT = Path('/mnt/data/lcx1/dong/need_data/weaver_p0_replay_gpu47_20261106_v1')", f"ROOT = Path({str(root)!r})"),
        ("queue=ROOT.parent/'experiment_queue_gpu47_20261023/queue.sqlite'", f"queue=Path({str(QUEUE/'queue.sqlite')!r})"),
        ("'weaver-p0-request-replay-17-gpu47-20261106-v1'",repr(sid)),
    ]:
        require(text.count(old)==1, f"P0 绑定位置不唯一: {old}")
        text=text.replace(old,new)
    compile(text,"replay.py","exec")
    return text.encode()


def make_spec(profile, label, root, stage):
    spec,source,wrapper=paths(profile)
    sid="weaver-run-"+label
    require(root.is_absolute() and root.resolve()==root, "输出必须使用独立实路径")
    mps=(PARENT/"mps.json").read_bytes()
    require(digest(mps)==spec["mps_receipt"], "MPS 回执有变化")
    (stage/"mps.json").write_bytes(mps)
    if profile=="p0":
        program=replay_program(source.parent,root,sid)
        (stage/"replay.py").write_bytes(program)
        argv=[PYTHON,str(root/"replay.py")]
    else:
        entry="profile_queue_entry.py" if profile=="efficiency" else "queue_entry.py"
        argv=[PYTHON,str(wrapper/entry),"--source",str(source),"--root",str(root),"--manifest-sha256",spec["manifest"]]
        if profile!="efficiency":
            argv+=["--native-gpu-preflight","--skip-final-evaluation"]
    argv=["/usr/bin/env","PYTHONNOUSERSITE=1","PYTHONDONTWRITEBYTECODE=1",*argv]
    job=dict(name="WEAVER "+profile+" "+label,submission_id=sid,cwd=str(source),argv=argv,
             manual=profile!="p0",timeout_seconds=3900 if profile=="p0" else 0)
    record=dict(schema=1,profile=profile,label=label,root=str(root),source=str(source),manifest_sha256=spec["manifest"],
                release=str(HERE),job=job)
    if profile=="p0":
        record["replay_sha256"]=digest(program)
    json_write(stage/"experiment.json",record)
    return record


def check_record(record):
    argv=record["job"]["argv"]
    profile=record["profile"]
    if profile=="p0":
        require(digest((Path(record["root"])/"replay.py").read_bytes())==record["replay_sha256"],"P0 回放程序校验失败")
    if profile=="efficiency":
        # The private collector has no validate-only switch. Build its real
        # target command without starting Nsight, then validate its inner job.
        spec,source,wrapper=paths(profile)
        root=Path(record["root"])
        check="import json,sys;from pathlib import Path;import cycle_profile as p;print(json.dumps(p.profile_command(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])))"
        command=json.loads(cpu([PYTHON,"-c",check,str(source),str(root),str(wrapper),spec["manifest"]],cwd=source))
        target=command.index(str(wrapper/"profile_target_entry.py"))
        require(os.access(command[0],os.X_OK) and "--capture-range=nvtx" in command and "--skip-final-evaluation" in command,
                "Nsight 采样计划不匹配")
        argv=[PYTHON,str(wrapper/"queue_entry.py"),*command[target+1:]]
    out=cpu([*argv,"--cpu-check" if profile=="p0" else "--validate-only"],cwd=record["source"])
    result=json.loads(out.strip().splitlines()[-1])
    if profile=="p0":
        require(result["cases"]==17 and result["arms_admitted"] and result["cuda_initialized"] is False,"P0 CPU 准入失败")
    else:
        require(result.get("valid") and result["validation"]["gpu_access_performed"] is False,"训练 CPU 准入失败")


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
        with tempfile.TemporaryDirectory(dir=EXPERIMENTS,prefix=".staging-") as staging:
            staged=Path(staging)/"output"
            staged.mkdir(mode=0o700)
            r=make_spec(profile,label,root,staged)
            staged.rename(root)
    require(r["release"]==str(HERE), "未提交的准备记录属于先前启动器版本，请使用新实验名")
    with tempfile.TemporaryDirectory(dir=EXPERIMENTS,prefix=".verify-") as temp:
        expected=make_spec(profile,label,root,Path(temp))
        require(r==expected and digest((root/"mps.json").read_bytes())==PROFILES[profile]["mps_receipt"],
                "准备记录与冻结配置不一致")
    json_write(EXPERIMENTS/".last.json",label)
    ready()
    handoff_verified()
    print(f"正在校验 {profile}，实验名 {label}，输出 {root}",file=sys.stderr,flush=True)
    check_record(r)
    ready()
    item=create(r["job"],QUEUE)
    save_submission(root,item)
    return summary(r,item)


def offline_check(profile):
    with tempfile.TemporaryDirectory(prefix=".cpu-check-",dir=EXPERIMENTS) as d:
        temp=Path(d)
        root=temp/"output"
        root.mkdir()
        r=make_spec(profile,"validation-"+profile,root,root)
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
    spec,source,_=paths(record["profile"])
    members=json.loads((source/"source_manifest.json").read_bytes())["files"]
    require(source==Path(record["source"]) and spec["manifest"]==record["manifest_sha256"] and
            digest((source/"request_stop.py").read_bytes())==members["request_stop.py"],"停止程序校验失败")
    return json.loads(cpu([PYTHON,str(source/"request_stop.py"),"--data",str(root/"formal_v3"),"--run-id",state["run_id"]],
                          cwd=source,env={"ACCERL_EXPERIMENT_ROOT":str(root)}))


def main():
    p=argparse.ArgumentParser(prog="./run.sh",description="WEAVER GPU4–7 | 无参数启动 restore；check 只做 CPU 验收",
        epilog="./run.sh  |  ./run.sh list  |  ./run.sh check restore  |  ./run.sh fixed --id trial01  |  ./run.sh status  |  ./run.sh stop")
    p.add_argument("args",nargs="*",metavar="动作或配置")
    p.add_argument("--id",dest="label",help="自定义实验名；重试时使用相同名字")
    opts=p.parse_args()
    args=opts.args.copy()
    action=args.pop(0) if args and args[0] in ("list","check","start","status","stop") else "start"
    profile=args.pop(0) if args else "restore"
    profile="restore" if profile=="original" else profile
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
            j=lookup(r["job"]["submission_id"])
            result=stop(r,j) if action=="stop" else summary(r,j)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as error:
        sys.exit(f"run.sh: {error}")
