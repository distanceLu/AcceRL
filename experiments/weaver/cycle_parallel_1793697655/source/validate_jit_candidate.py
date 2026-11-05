"""CPU acceptance with parent regressions, actual JIT runtime and profile checks."""
import json
from pathlib import Path
import sys

import validate_cost_scheduler_candidate as parent


def main():
    original=parent.validate_profiles
    profiles=[]
    def validate_profiles():
        result=original()
        from fresh_config import FreshConfig
        from world_backends import WorldBackendConfig
        base=json.loads((parent.ROOT/'examples/fresh_04_real.json').read_text('utf8'))
        expected={'00_shared_exact':('off','optimizer_window'),
                  '01_trainer_exact':('trainer_separate','optimizer_window'),
                  '02_trainer_ema':('trainer_separate','ema_zscore'),
                  '03_fused_ema':('fused','ema_zscore'),'04_fused_rms':('fused','ema_rms')}
        for name,(mode,norm) in expected.items():
            raw=json.loads((parent.ROOT/f'examples/jit_{name}.json').read_text('utf8'))
            config=WorldBackendConfig.from_mapping(raw);config.validate_admission()
            fresh=FreshConfig.from_world_config(config)
            assert (fresh.jit_gae_mode,fresh.advantage_norm)==(mode,norm)
            for key in ('jit_gae_mode','advantage_norm','advantage_ema_beta','advantage_scale_floor','jit_context_micro'):
                raw['backend_options']['fresh'].pop(key)
            assert raw==base,'JIT profile changed an unrelated learning rule'
            profiles.append(dict(profile=name,switches=fresh.to_dict()))
        return result
    parent.validate_profiles=validate_profiles
    parent.NEW_MODULES=parent.NEW_MODULES+('test_jit_gae_cpu','test_jit_parent_cpu')
    exit_code=parent.main()
    source=parent.ROOT/'validation/cost_cpu_validation.json'
    if source.exists():
        report=json.loads(source.read_text('utf8'))
        report.update(protocol='weaver_jit_gae_cpu_acceptance_v1',jit_profiles=profiles,
            scope='Local CPU parent regressions and actual JIT runtime with toy engine; real CPU Gloo has a separate receipt. No Ray/OpenVLA/ZeRO2/CUDA validation.',
            cuda_deployed=False)
        (source.parent/'jit_cpu_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return exit_code


if __name__=='__main__':raise SystemExit(main())
