"""Single-node entry for the archived original95 training recipe (Linux, four GPUs)."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE = HERE / "engine"
RECIPE = HERE / "recipe"
from git_version import BRANCH, published
REFERENCE_SERVER = "h20-xyff52ab-0001"
SITE_ENTRY = Path("/mnt/data/lcx1/dong/WEAVER/run.sh")
NATIVE_DIR = Path("/mnt/data/lcx1/dong/envs/weaver-libero-20261022")
SITE_ASSETS = Path("/mnt/data/lcx1/dong/need_data/weaver_libero_offline_20261022/micro8_formal_v1/inference_bundle")
SITE_POLICY = Path("/mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt")
SITE_ENCODERS = Path("/mnt/data/lcx1/dong/checkpoints/weaver_public_20261022/bundled_encoders")


def read(p):
    return json.loads(p.read_bytes())


def put(p, value):
    with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:
        f.write((json.dumps(value,indent=2)+"\n").encode())
        f.flush()
        os.fsync(f.fileno())
    Path(f.name).replace(p)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checksum(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def settings():
    host=socket.gethostname()==REFERENCE_SERVER and not os.getenv("WEAVER_ASSETS")
    assets=Path(os.getenv("WEAVER_ASSETS",ROOT/"assets")).expanduser().absolute()
    defaults={"policy":SITE_POLICY if host else assets/"openvla",
              "distilled":SITE_POLICY.parent/"20251225_113851_distill_checkpoint_latest.pt" if host else assets/"policy_distilled.pt",
              "bundle":SITE_ASSETS if host else assets/"native3200",
              "image":SITE_ENCODERS/"image" if host else assets/"image_encoder",
              "task":SITE_ENCODERS/"task" if host else assets/"task_encoder",
              "native":NATIVE_DIR/"bin/python" if socket.gethostname()==REFERENCE_SERVER else ROOT/".venv-native/bin/python"}
    variables={"policy":"WEAVER_POLICY","distilled":"WEAVER_DISTILLED","bundle":"WEAVER_BUNDLE",
               "image":"WEAVER_IMAGE_ENCODER","task":"WEAVER_TASK_ENCODER","native":"WEAVER_NATIVE_PYTHON"}
    # Keep the venv interpreter entry rather than resolving its symlink to the base Python.
    return {k:str(Path(os.getenv(variables[k],v)).expanduser().absolute()) for k,v in defaults.items()}


def gpu_inventory():
    out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True,timeout=15)
    return dict(line.strip().split(", ",1) for line in out.splitlines() if line.strip())


def allocated(check):
    inherited=os.getenv("CUDA_VISIBLE_DEVICES")
    ids=os.getenv("WEAVER_GPU_IDS") or inherited
    if not ids and check:
        ids="4,5,6,7" if socket.gethostname()==REFERENCE_SERVER else "0,1,2,3"
    require(ids,"Set CUDA_VISIBLE_DEVICES or WEAVER_GPU_IDS to the four allocated physical cards.")
    inventory=gpu_inventory()
    by_uuid={val:key for key,val in inventory.items()}
    chosen=[by_uuid.get(v.strip(),v.strip()) for v in ids.split(",")]
    require(len(chosen)==len(set(chosen))==4 and all(v in inventory for v in chosen),"Need four distinct available physical GPUs")
    if inherited is not None and not check:
        allowed={by_uuid.get(v.strip(),v.strip()) for v in inherited.split(",")}
        require(set(chosen)<=allowed,"Selected GPUs exceed CUDA_VISIBLE_DEVICES allocation")
    return chosen, [inventory[i] for i in chosen]


def mps_preflight(pipe, cards):
    require(pipe and Path(pipe).is_dir() and Path(pipe).is_absolute(),"Set CUDA_MPS_PIPE_DIRECTORY for your existing MPS service.")
    def control(cmd):
        return subprocess.check_output(["nvidia-cuda-mps-control"],input=cmd+"\n",text=True,timeout=12).strip()
    servers=control("get_server_list").split()
    require(len(servers)==1 and servers[0].isdigit(),"Expected one existing MPS server on this pipe")
    server=servers[0]
    require(Path("/proc",server).stat().st_uid==os.getuid(),"MPS server has a different owner")
    require(not control("get_client_list "+server),"MPS service already has clients")
    for cmd in ("get_default_active_thread_percentage","get_active_thread_percentage "+server):
        require(float(control(cmd))==100.0,"This profile requires a 100% MPS cap")
    out=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True,timeout=15)
    for line in out.splitlines():
        gpu,pid=(v.strip() for v in line.split(","))
        require(gpu not in cards or pid==server,"An allocated GPU is already being used")
    usage=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"],text=True,timeout=15)
    for line in usage.splitlines():
        gpu,used,utilization=(v.strip() for v in line.split(","))
        if gpu in cards:
            require(int(used)<=1024 and int(utilization)==0,"Allocated GPU is busy: "+gpu)
    mask=next((x.partition(b"=")[2].decode() for x in (Path("/proc")/server/"environ").read_bytes().split(b"\0")
               if x.startswith(b"CUDA_VISIBLE_DEVICES=")),None)
    if mask is not None:
        inventory=gpu_inventory()
        allowed={inventory.get(i.strip(),i.strip()) for i in mask.split(",")}
        require(set(cards)<=allowed,"MPS server's GPU mask differs from the allocation")


def expand(data, values):
    if isinstance(data,dict):
        return {k:expand(v,values) for k,v in data.items()}
    if isinstance(data,list):
        return [expand(v,values) for v in data]
    if isinstance(data,str) and data.startswith("@"):
        if data.startswith("@inactive_"):
            return str(ROOT/"assets"/"unused_ctrl_backend"/data[1:])
        return values[data[1:]]
    return data


def exec_managed_driver(overrides=()):
    """Start the real trainer after the site supervisor has set its run identity and paths."""
    values=dict(settings(), ids="4,5,6,7", world=os.environ["ACCERL_WORLD_BACKEND_CONFIG"],
                pipeline=os.environ["ACCERL_PIPELINE_CONFIG"], checkpoints=os.environ["ACCERL_FINAL_CHECKPOINT"],
                agent=str(ROOT/"envs/config/agent.yaml"),trainer=str(ROOT/"envs/config/trainer.yaml"))
    argv=expand(read(RECIPE/"argv.json"),values) + list(overrides)
    env=dict(os.environ)
    env.update(read(RECIPE/"env.json"))
    # TensorBoard consumers on this host already watch the original runs directory.
    os.chdir(SITE_ENTRY.parent.parent/"AcceRL-scheduler")
    os.execve(sys.executable,[sys.executable,str(ENGINE/"train.py"),*argv],env)


def prepare(work, ids, run_id):
    put(work/'visualization_config.json',read(RECIPE/'visualization.json'))
    host=settings()
    missing=[k+"="+v for k,v in host.items() if not Path(v).exists()]
    require(not missing,"Missing local resources: "+", ".join(missing)+". See .weaver.env.example.")
    require(Path(host["native"]).is_file() and os.access(host["native"],os.X_OK),"Invalid native Python")
    values=dict(host,ids=",".join(ids),weaver=str(HERE/"vendor/weaver"),world=str(work/"world.json"),
                pipeline=str(RECIPE/"pipeline_config.json"),checkpoints=str(work/"checkpoints"),
                agent=str(ROOT/"envs/config/agent.yaml"),trainer=str(ROOT/"envs/config/trainer.yaml"))
    put(work/"world.json", expand(read(RECIPE/"world_backend_config.json"),values))
    argv=expand(read(RECIPE/"argv.json"),values)
    env={k:v for k,v in os.environ.items() if not k.startswith(("ACCERL_","ACCE_CAPTURE_","NSYS_","CUDA_INJECTION","NVTX_INJECTION","EXPERIMENT_QUEUE_"))}
    for key in ("RAY_ADDRESS","SOFT_STOP_AT","MEASURE_STOP_AFTER_EPOCHS","SHORT_TEST_DEADLINE",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE","CUDA_MPS_PINNED_DEVICE_MEM_LIMIT","CUDA_MPS_CLIENT_PRIORITY"):
        env.pop(key,None)
    env.update(read(RECIPE/"env.json"))
    pythonpath=[str(ENGINE),str(ROOT),str(ROOT/"rl/ctrl_world"),str(HERE/"vendor/ctrl")]
    if os.getenv("WEAVER_LIBERO_ROOT"):
        libero=Path(os.environ["WEAVER_LIBERO_ROOT"]).resolve()
        pythonpath.append(str(libero))
        if not env.get("LIBERO_CONFIG_PATH"):
            package=libero/"libero/libero"
            require((package/"bddl_files").is_dir() and (package/"init_files").is_dir(),"Incomplete LIBERO task assets")
            location=work/"libero-config"
            location.mkdir()
            # JSON is valid YAML, including paths containing spaces.
            put(location/"config.yaml",dict(benchmark_root=str(package),bddl_files=str(package/"bddl_files"),
                init_states=str(package/"init_files"),datasets=str(libero/"libero/datasets"),assets=str(package/"assets")))
            env["LIBERO_CONFIG_PATH"]=str(location)
    if not env.get("LIBERO_CONFIG_PATH") and not (Path.home()/".libero/config.yaml").is_file():
        raise ValueError("Set WEAVER_LIBERO_ROOT to your complete LIBERO checkout, or LIBERO_CONFIG_PATH to your existing config directory")
    env.update(PYTHONPATH=os.pathsep.join(pythonpath),PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",
               ACCERL_RUN_ID=run_id,ACCERL_ALLOCATED_GPU_IDS=values["ids"],ACCERL_RUN_STARTED_NS=str(time.time_ns()),
               ACCERL_EXPERIMENT_ROOT=str(work),ACCERL_DATA_DIR=str(work),ACCERL_FINAL_CHECKPOINT=values["checkpoints"],
               ACCERL_REPO_ROOT=str(ROOT),ACCERL_STOP_FILE=str(work/"stop.flag"),ACCERL_TELEMETRY_DB=str(work/"collection.sqlite"),
               ACCERL_PIPELINE_CONFIG=values["pipeline"],ACCERL_WORLD_BACKEND_CONFIG=values["world"],
               ACCERL_COST_SCHEDULER_CONFIG=str(RECIPE/"cost_scheduler_config.json"),
               ACCERL_RAY_TMP=f"/dev/shm/wr_{os.getuid()}_{uuid.uuid4().hex[:10]}",ACCERL_RAY_SPILL=str(work/"ray_spill"))
    return [sys.executable,str(ENGINE/"train.py"),*argv],env,host


def check(work, cmd, env, host):
    safe=dict(env,CUDA_VISIBLE_DEVICES="")
    for name,h in read(RECIPE/"bundle-checksums.json").items():
        require(checksum(Path(host["bundle"])/name)==h,"native3200 bundle mismatch: "+name)
    for name,h in read(RECIPE/"policy-checksums.json").items():
        p=Path(host["distilled"]) if name=="@distilled" else Path(host["policy"])/name
        require(checksum(p)==h,"Initial policy asset mismatch: "+name)
    result=subprocess.run([sys.executable,str(ENGINE/"inspect_world_backend.py"),"--config",str(work/"world.json"),"--check-local"],
                          cwd=work,env=safe,text=True,capture_output=True,timeout=150)
    require(result.returncode==0,"Native WEAVER admission failed: "+(result.stdout+result.stderr)[-2000:])
    subprocess.run([sys.executable,str(HERE/"check_runtime.py"),*cmd[2:]],cwd=work,env=safe,check=True,timeout=180)
    subprocess.run([host["native"],"-c","import sys;sys.path.insert(0,sys.argv[1]);from weaver.wm import model,encoders;import torch;assert not torch.cuda.is_initialized()",
                    str(HERE/"vendor/weaver")],cwd=work,env=safe,check=True,timeout=90)


def alive(pid,ticks):
    try:
        p=Path("/proc")/str(pid)
        fields=(p/"stat").read_text().rsplit(")",1)[1].split()
        return p.stat().st_uid==os.getuid() and fields[0] not in ("Z","X") and int(fields[19])==ticks
    except (FileNotFoundError,ProcessLookupError):
        return False


def status(work):
    s=read(work/"status.json")
    if s["state"]=="running" and (s["boot"]!=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                                  or not alive(s["pid"],s["birth_ticks"])):
        s=dict(s,state="exited_without_final_receipt")
    return s


def stop(work):
    s=status(work)
    require(s["state"]=="running" and s["boot"]==Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"No live matching run")
    put(work/"stop.flag",dict(reason="user_requested_stop",run_id=s["run_id"],uid=os.getuid(),requested_ns=time.time_ns()))
    return dict(run_id=s["run_id"],stop_requested=True)


def supervise(work, cmd, env, log, run_id):
    child=subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s=dict(state="running",run_id=run_id,pid=child.pid,started_ns=time.time_ns(),
           boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
           birth_ticks=int(Path(f"/proc/{child.pid}/stat").read_text().rsplit(")",1)[1].split()[19]))
    put(work/"status.json",s)
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_: stop(work) if child.poll() is None else None)
    code=child.wait()
    result=dict(s,state="failed",exit_code=code,finished_ns=time.time_ns())
    if code==0:
        try:
            directory=work/"checkpoints"
            manifest=read(directory/"manifest.json")
            world=manifest["metadata"]["online_world"]
            policy_weights=directory/"latest_weights.pt"
            world_weights=Path(world["weights_file"]).resolve()
            require(manifest["reason"] in ("final","final_before_first_update")
                    and world["reason"]==manifest["reason"]
                    and manifest["metadata"]["run_id"]==world["run_id"]==run_id
                    and world["policy_step"]==manifest["steps"]["policy"]
                    and world["successful_updates"]==world["policy_step"]//5,
                    "Final snapshot boundary mismatch")
            require(world_weights.is_relative_to(directory/"checkpoints_world")
                    and world_weights.stat().st_size==world["weights_bytes"]
                    and checksum(world_weights)==world["weights_sha256"]
                    and policy_weights.stat().st_size==manifest["weights_bytes"]>0,
                    "Final snapshot files failed validation")
            result.update(state="finished",policy_step=world["policy_step"],world_updates=world["successful_updates"])
        except (OSError,KeyError,ValueError) as error:
            result["save_error"]=str(error)
    put(work/"status.json",result)
    ok=result["state"]=="finished"
    return 0 if ok else (code or 1)


def main():
    parser=argparse.ArgumentParser(prog="./run.sh")
    parser.add_argument("action",choices=["original","restore","start","check","status","stop","diagnose","diagnose-status","view"],default="original",nargs="?")
    parser.add_argument("--id",help="Run label. Use the same label to check or stop the run.")
    parser.add_argument("--root",type=Path,help="Existing output directory (view only)")
    parser.add_argument("--port",type=int,default=8765,help="Loopback viewer port (view only)")
    args=parser.parse_args()
    host=socket.gethostname()==REFERENCE_SERVER
    if args.action=='view':
        sys.path.insert(0,str(ENGINE))
        from visualization import serve
        if args.root:
            work=args.root
        else:
            outputs=Path('/mnt/data/lcx1/dong/WEAVER/experiments') if host else Path(os.getenv('WEAVER_OUTPUTS',ROOT/'.weaver-runs'))
            label=args.id or read(outputs/('.last.json' if host else 'last.json'))
            require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',label),'Invalid run label')
            work=outputs/label
        serve(work,args.port)
        return 0
    require(args.root is None and args.port==8765,'--root/--port are only for view')
    if host and args.action!="check":
        op="original" if args.action in ("original","restore","start") else args.action
        cli=HERE/"reference_site/cli.py"
        os.execv(sys.executable,[sys.executable,"-I","-B",str(cli),op,*(["--id",args.id] if args.id else [])])
    require(not args.action.startswith('diagnose'),'This saved-checkpoint diagnostic uses the reference site queue and assets')
    if args.action=="check":
        with tempfile.TemporaryDirectory(prefix="weaver-check-") as d:
            ids,_=allocated(True)
            cmd,env,paths=prepare(Path(d),ids,"weaverintegrated_cpu_check")
            check(Path(d),cmd,env,paths)
        print("CHECK PASSED: clone imports, actual parsed training arguments, native bundle and CPU environment.")
        return 0
    outputs=Path(os.getenv("WEAVER_OUTPUTS",ROOT/".weaver-runs")).expanduser().resolve()
    outputs.mkdir(exist_ok=True,parents=True)
    label=args.id
    if not label:
        if args.action in ("status","stop"):
            require((outputs/"last.json").exists(),"No run has started in WEAVER_OUTPUTS.")
        label=(read(outputs/"last.json") if args.action in ("status","stop") else
               datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6])
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",label),"Invalid run label")
    work=outputs/label
    require(work.resolve()==work,"Output path must be canonical")
    if args.action in ("status","stop"):
        print(json.dumps(status(work) if args.action=="status" else stop(work),indent=2))
        return 0
    git_commit=published(ROOT)
    ids,cards=allocated(False)
    work.mkdir(mode=0o700)
    run_id="weaverintegrated_"+label
    put(work/"status.json",dict(state="checking",run_id=run_id))
    try:
        cmd,env,paths=prepare(work,ids,run_id)
        check(work,cmd,env,paths)
        mps_preflight(env.get("CUDA_MPS_PIPE_DIRECTORY"),cards)
        # Advisory locks only for this user's explicitly allocated GPUs. Keep through exit.
        from contextlib import ExitStack
        with ExitStack() as locks:
            for gpu in sorted(cards):
                file=locks.enter_context(open(f"/dev/shm/weaver-{os.getuid()}-{gpu}.lock","a"))
                fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            mps_preflight(env.get("CUDA_MPS_PIPE_DIRECTORY"),cards)
            env["CUDA_VISIBLE_DEVICES"]=",".join(cards)
            require(published(ROOT)==git_commit,"Revision changed during admission")
            put(work/"command.json",dict(git_branch=BRANCH,git_commit=git_commit,argv=cmd,gpus=cards,run_id=run_id,recipe="original95",weights_only=True))
            put(outputs/"last.json",label)
            print(f"Starting {run_id}; log: {work/'train.log'}",flush=True)
            with (work/"train.log").open("xb") as log:
                return supervise(work,cmd,env,log,run_id)
    except Exception as error:
        previous=read(work/"status.json")
        if previous["state"]=="checking":
            put(work/"status.json",dict(previous,state="failed",error=str(error)))
        raise


if __name__=="__main__":
    try:
        sys.exit(main())
    except (ValueError,OSError,subprocess.SubprocessError) as e:
        sys.exit(f"run.sh: {e}")
