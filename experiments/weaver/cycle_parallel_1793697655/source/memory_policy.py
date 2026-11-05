"""Per-instance memory choices; batch sizes, loss and optimizer math unchanged."""
from types import MethodType


def _vision_checkpoint_forward(self, x):
    import torch
    from torch.utils.checkpoint import checkpoint
    original = self._accerl_original_forward
    if self.training and torch.is_grad_enabled():
        return checkpoint(original, x, use_reentrant=False, preserve_rng_state=True)
    return original(x)


def install_reward_checkpointing(reward_model):
    from transformers import LlamaForCausalLM
    from peft import PeftModel
    if type(reward_model).__name__ != 'RewardModel':
        raise TypeError('Checkpointing is restricted to this Reward model')
    vla = reward_model.vla
    if isinstance(vla, PeftModel):
        vla = vla.get_base_model()
    language = vla.language_model
    core = language.get_base_model() if isinstance(language, PeftModel) else language
    if not isinstance(core, LlamaForCausalLM) or not getattr(core, '_accerl_reward_features_only', False):
        raise TypeError('Reward checkpointing requires the verified hidden-only Llama path')
    core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        'use_reentrant': False, 'preserve_rng_state': True})
    if not core.model.gradient_checkpointing:
        raise RuntimeError('Reward decoder checkpointing did not enable')
    # timm 0.9.10 get_intermediate_layers bypasses forward_features and its
    # grad_checkpointing flag. Wrap each original Block.forward instance so
    # the actual Prismatic intermediate-feature path is checkpointed too.
    vision_blocks = []
    vision = getattr(vla, 'vision_backbone', None)
    if vision is not None:
        from timm.models.vision_transformer import Block
        for tower_name in ('featurizer', 'fused_featurizer'):
            tower = getattr(vision, tower_name, None)
            if tower is None:
                continue
            for index, block in enumerate(tower.blocks):
                if not isinstance(block, Block):
                    raise TypeError('Unknown Reward vision block')
                if not hasattr(block, '_accerl_original_forward'):
                    block._accerl_original_forward = block.forward
                    block.forward = MethodType(_vision_checkpoint_forward, block)
                vision_blocks.append(f'{tower_name}.blocks.{index}')
    return {'scope': 'reward_training_instance_only', 'backend': 'native_hf_decoder_and_instance_timm_blocks',
            'use_reentrant': False, 'preserve_rng_state': True,
            'layers': len(core.model.layers), 'vision_blocks': vision_blocks,
            'vision_block_count': len(vision_blocks),
            'module_names_and_state_keys_unchanged': True}
