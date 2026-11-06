"""CPU parent regressions plus the new opt-in trajectory protocol."""
import json
import validate_jit_candidate as jit

def main():
    jit.parent.NEW_MODULES += ('test_trajectory_sampling_cpu',)
    code=jit.main()
    path=jit.parent.ROOT/'validation/jit_cpu_validation.json'
    if path.exists():
        report=json.loads(path.read_text('utf8'))
        report.update(protocol='weaver_jit_trajectory_cpu_v1',gpu_deployed=False,
            sampling_profile='examples/trajectory_01_imagined.json')
        (path.parent/'trajectory_cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return code

if __name__=='__main__':raise SystemExit(main())
