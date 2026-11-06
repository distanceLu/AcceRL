"""Parent + trajectory regressions and bounded adaptive/parallel protocol tests."""
import json
import validate_trajectory_candidate as trajectory


def main():
    parent = trajectory.jit.parent
    parent.NEW_MODULES += ('test_cycle_parallel_cpu',)
    code = trajectory.main()
    path = parent.ROOT/'validation/trajectory_cpu_validation.json'
    if path.exists():
        report=json.loads(path.read_text('utf8'))
        report.update(protocol='weaver_cycle_parallel_cpu_v1',gpu_deployed=False,
            scope='CPU parent learning regressions, actual asyncio broker and cycle feedback. No new Ray/NCCL/CUDA or concurrent memory validation.')
        (path.parent/'cycle_parallel_cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return code


if __name__=='__main__':raise SystemExit(main())
