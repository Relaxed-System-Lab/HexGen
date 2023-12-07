# Adapted from https://github.com/AFDWang/Hetu-Galvatron/blob/00b8abc168a2deb26861e4672b922acf50321980/galvatron/models/llama/LlamaModel_sequential.py#L27
import torch.nn as nn
import torch
try:
    from llama_inference import inference
except ImportError:
    pass

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None
    
try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm_parallel_residual
except ImportError:
    dropout_add_rms_norm_parallel_residual = None

from collections import namedtuple, OrderedDict
import sys

from llama.arguments import get_kv_cache, set_kv_cache

class LlamaEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        attrs = ['embeddings', 'process_group', 'sequence_parallel']
        for key in attrs:
            setattr(self, key, getattr(model, key))
    
    def label(self):
        return [0,0]
    
    def forward(self, input_ids, position_ids=None, inference_params=None):
        embedding_kwargs = ({'combine_batch_seqlen_dim': True}
                            if self.process_group is not None and self.sequence_parallel else {})
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, **embedding_kwargs)
        input_ids = input_ids.clone()
        return hidden_states, input_ids



class LlamaLayers_(nn.Module):
    def __init__(self, model, layer_idx_start, layer_idx_end, pp_indices):
        super().__init__()
        model = model.transformer
        self.layers = model.layers[layer_idx_start:layer_idx_end]
        self.layer_idx = layer_idx_start
        self.pp_indices = pp_indices
        attrs = ['prenorm', 'parallel_block', 'process_group', 'sequence_parallel']
        for key in attrs:
            setattr(self, key, getattr(model, key))
        
    def label(self):
        return [1,self.layer_idx]

    def forward(self, hidden_states, input_ids, residual=None, position_ids=None, inference_params=None):
        kv_cache_dict = get_kv_cache()
        if self.layer_idx in self.pp_indices and kv_cache_dict is None:
            mixer_kwargs = ({'seqlen': hidden_states.shape[1]}
                            if self.process_group is not None and self.sequence_parallel else {})
            if inference_params is not None:
                mixer_kwargs['inference_params'] = inference_params
        for layer in self.layers:
            if self.prenorm:
                if self.layer_idx in self.pp_indices and kv_cache_dict is None:
                    hidden_states, residual, kv_cache_dict = layer(hidden_states, residual, mixer_kwargs=mixer_kwargs)
                else:
                    kv_cache_dict['inference_params'].sequence_len_offset=inference_params.sequence_len_offset
                    hidden_states, residual, kv_cache_dict = layer(hidden_states, residual, mixer_kwargs=kv_cache_dict)
                set_kv_cache(kv_cache_dict)
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            
        input_ids = input_ids.clone()
        return hidden_states, input_ids, residual

class LlamaPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        self.drop_f = model.drop_f
        self.ln_f = model.ln_f
        attrs = ['fused_dropout_add_ln', 'drop_f', 'parallel_block', 'ln_f', 'prenorm', 'residual_in_fp32']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def label(self):
        return [2,0]

    def forward(self, hidden_states, input_ids, residual=None, position_ids=None, inference_params=None):
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                if not self.parallel_block:
                    residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                if not self.parallel_block:
                    fused_add_norm_fn = (dropout_add_rms_norm if isinstance(self.ln_f, RMSNorm)
                                         else dropout_add_layer_norm)
                    hidden_states = fused_add_norm_fn(
                        hidden_states, residual, self.ln_f.weight, self.ln_f.bias,
                        self.drop_f.p if self.training else 0.0, self.ln_f.eps, prenorm=False,
                        residual_in_fp32=self.residual_in_fp32
                    )
        input_ids = input_ids.clone()
        return hidden_states, input_ids

class LlamaCls_(nn.Module):
    def __init__(self, model):
        super().__init__()
        attrs = ['lm_head', 'config', 'project_out']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def label(self):
        return [3,0]

    def forward(self, hidden_states, input_ids, position_ids=None, inference_params=None):
        
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # # During inference, we want the full logit for sampling
        if isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
            lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
            lm_logits = rearrange(lm_logits, '(n b) s d -> b s (n d)', b=hidden_states.shape[0])
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)
