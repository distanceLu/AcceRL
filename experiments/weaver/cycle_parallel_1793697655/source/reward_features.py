"""Reward-instance-only decoder features, retaining the original VLA call path.

Pinned to the deployed Transformers 4.40.1 Llama interface. No global library
patch, replacement module, parameter rename, input reconstruction, or detach.
"""
from types import MethodType


def _features_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                      past_key_values=None, inputs_embeds=None, labels=None,
                      use_cache=None, output_attentions=None, output_hidden_states=None,
                      return_dict=None, cache_position=None):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if labels is not None or past_key_values is not None or return_dict is False:
        raise ValueError("Reward features require a full sequence, no LM loss, and named outputs")
    # Call through __call__: decoder/FSDP/adapter hooks must still execute.
    output = self.model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=None, inputs_embeds=inputs_embeds, use_cache=False,
        output_attentions=output_attentions, output_hidden_states=False,
        return_dict=True, cache_position=cache_position,
    )
    # The unchanged RewardModel reads only hidden_states[-1]. Avoid retaining
    # the intermediate-layer tuple and avoid both lm_head and FP32 vocab logits.
    return CausalLMOutputWithPast(
        loss=None, logits=None, past_key_values=None,
        hidden_states=(output.last_hidden_state,), attentions=output.attentions,
    )


def install_reward_features_only(reward_model):
    """Specialize only this Reward model's CausalLM instance before wrapping."""
    from transformers import LlamaForCausalLM
    from peft import PeftModel

    if type(reward_model).__name__ != "RewardModel":
        raise TypeError("Hidden-only specialization is restricted to RewardModel")
    vla = reward_model.vla
    if isinstance(vla, PeftModel):
        vla = vla.get_base_model()
    language = vla.language_model
    core = language.get_base_model() if isinstance(language, PeftModel) else language
    if not isinstance(core, LlamaForCausalLM):
        raise TypeError("The deployed Reward path requires LlamaForCausalLM")
    if getattr(core, "_accerl_reward_features_only", False):
        return core._accerl_reward_features_metadata
    metadata = {"scope": "this_reward_instance_only", "causal_lm": type(core).__name__,
                "decoder": type(core.model).__name__, "skip_lm_head": True,
                "use_cache": False, "retain_intermediate_hidden_states": False,
                "multimodal_preparation_unchanged": True, "decoder_called_with_hooks": True}
    core.forward = MethodType(_features_forward, core)
    core._accerl_reward_features_only = True
    core._accerl_reward_features_metadata = metadata
    return metadata
