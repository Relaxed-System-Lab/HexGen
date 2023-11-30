# Copyright (c) 2023, Tri Dao.

import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from transformers import GPT2Config, LlamaConfig

def config_from_meta_checkpoint(checkpoint_path: Union[str, os.PathLike], model_name: str) -> LlamaConfig:
    """Load a LlamaConfig from a checkpoint path."""
    with open(Path(checkpoint_path) / model_name / 'params.json') as f:
        params = json.load(f)
    config = LlamaConfig(hidden_size=params['dim'], intermediate_size=None,
                         num_attention_heads=params['n_heads'],
                         num_hidden_layers=params['n_layers'],
                         rms_norm_eps=params['norm_eps'])
    return config


def config_from_hf_checkpoint(checkpoint_path: Union[str, os.PathLike], model_name: str) -> LlamaConfig:
    return LlamaConfig.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf' / "config.json")


def config_from_checkpoint(
    checkpoint_path: Union[str, os.PathLike], model_name: str, checkpoint_format="meta"
) -> LlamaConfig:
    if checkpoint_format == "meta":
        return config_from_meta_checkpoint(checkpoint_path, model_name)
    else:
        return config_from_hf_checkpoint(checkpoint_path, model_name)


def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config:
    return GPT2Config(
        vocab_size=llama_config.vocab_size,
        # n_positions=llama_config.max_position_embeddings,
        n_positions=0,
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function='swiglu',  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
    )
    
def overwrite_configs_and_args(config, args):
    overwrite_config = {'use_cache': False,
                        'use_flash_attn': args.use_flash_attn,
                        'fused_bias_fc': True,
                        'sequence_parallel': False}
    for key, val in overwrite_config.items():
        setattr(config, key, val)
    
    if args.overwrite_config:
        overwrite_config = {'hidden_size': args.hidden_size,
                            'max_position_embeddings': args.seq_length,
                            'num_hidden_layers': args.num_hidden_layers,
                            'vocab_size': args.vocab_size}
        for key, val in overwrite_config.items():
            setattr(config, key, val)
    else:
        args.hidden_size = config.hidden_size
        args.seq_length = config.max_position_embeddings
        args.max_position_embeddings = config.max_position_embeddings
        args.num_hidden_layers = config.num_hidden_layers
        args.vocab_size = config.vocab_size
