import torch
from torch import nn
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
import time
import os
import sys
from arguments import add_arguments
sys.path.insert(0, '..')
sys.path.insert(0, '../hexgen_core')
sys.path.insert(0, '../../third_party/megatron')
sys.path.insert(0, './modules')
from hexgen_core import decode
from megatron.initialize import initialize_megatron
from megatron import get_args
from torch.utils.data.distributed import DistributedSampler
from hybrid_parallel_model_dist import get_hybrid_parallel_configs, construct_hybrid_parallel_model, overwrite_megatron_args
from typing import Tuple, List
from llama_config_utils import llama_config_to_gpt2_config, config_from_checkpoint, overwrite_configs_and_args
from transformers import GPT2Config, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from hexgen_core.models.gpt import GPTLMHeadModel, shard_state_dict_tp, create_mixer_cls, create_mlp_cls
from hexgen_core import gen_hetero_groups
from load_model_parameters_utils.load_model_parameters import load_model_parameters
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def forward_step_func(inputs, inference_params, position_ids, model):
    if isinstance(inputs, (Tuple, List)):
        outputs = model(*inputs, position_ids=position_ids, inference_params=inference_params)
    else:
        outputs = model(inputs, position_ids=position_ids, inference_params=inference_params)
    return outputs

def create_model(args):
    if 'benchmark' in os.path.abspath('..'):
        os.chdir("../../hexgen/llama")    

    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    llama_config = config_from_checkpoint('./llama-config/', args.model_size)
    config = llama_config_to_gpt2_config(llama_config)
    overwrite_configs_and_args(config, args)
    overwrite_megatron_args(config, args)
    
    hybrid_parallel_configs = get_hybrid_parallel_configs(args)
    
    # Generate hetero groups with respect to given config
    hetero_groups = gen_hetero_groups(hetero_config=args.hetero_config, pp_partition=args.pp_partition, layer_num=args.num_hidden_layers)
    
    if local_rank == 0:
        print("Creating Model...")

    # Init model on meta device
    mixed_precision = {'fp32': torch.float, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.mixed_precision]
    gpt_model = GPTLMHeadModel(config, device='meta' if args.initialize_on_meta else 'cpu', dtype=mixed_precision)
    from flash_attn.models.gpt import create_mixer_cls, create_mlp_cls
    factory_kwargs = {'device': 'meta' if args.initialize_on_meta else 'cpu', 'dtype': mixed_precision}
    for i in range(config.num_hidden_layers):
        layer = gpt_model.transformer.layers[i]
        setattr(layer, 'mixer', create_mixer_cls(config, layer_idx=i, process_group=hetero_groups['current_tp_group'], **factory_kwargs)(config.hidden_size))
        setattr(layer, 'mlp', create_mlp_cls(config, layer_idx=i, process_group=hetero_groups['current_tp_group'], **factory_kwargs)(config.hidden_size))
    
    # Construct hybrid parallel model
    model = construct_hybrid_parallel_model(model=gpt_model, 
                                            model_config=config, 
                                            inference_args=args, 
                                            hybrid_parallel_configs=hybrid_parallel_configs,
                                            pp_partition=args.pp_partition,
                                            device=device,
                                            hetero_config=args.hetero_config)
    
    # Load model checkpoints with respect to hetero_config
    tp_ranks_whole_model = hetero_groups['tp_ranks_whole_model']
    tp_group_list = hetero_groups['tp_rank_groups']
    state_dicts_path = "./load_model_parameters_utils/"
    load_model_parameters(model, config, state_dicts_path, tp_ranks_whole_model, tp_group_list, rank)

    if rank == 0:
        print('Model configures:')
        print(config)
    time.sleep(rank * 0.1)

    # Initialize the tokenizer for the GPT model.
    tokenizer = LlamaTokenizer.from_pretrained("../../../Llama-2-7b-chat-hf/") 

    return model, tokenizer, hetero_groups['pp_rank_groups']

def inference(model, tokenizer, pp_groups, model_msg, args):
    # current rank
    rank = torch.distributed.get_rank()

    # Tokenize the provided prompt text.
    prompt_text = model_msg['prompt']
    max_length = model_msg['max_new_tokens']
    temperature = model_msg['temperature']
    top_k = model_msg['top_k']
    top_p = model_msg['top_p']
    
    max_length += len(prompt_text)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").cuda()
    input_ids_shape = [[-1, len(input_ids[0]), args.hidden_size], [-1, len(input_ids[0])], [-1, len(input_ids[0]), args.hidden_size]]

    torch.cuda.synchronize()
    start = time.time()
    output = decode(input_ids, input_ids_shape, model, forward_step_func, max_length, pp_last_stage_rank=pp_groups[0][-1], 
                    temperature=temperature, top_k=top_k, top_p=top_p, timing=True).sequences
    torch.cuda.synchronize()
    end = time.time()
    infer_time = end - start
    
    decoded_text = tokenizer.decode(output[0])
    if rank == pp_groups[0][-1]:
        print(decoded_text)
   
    return decoded_text, infer_time


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_arguments)
    args = get_args()
    set_seed()

    model, tokenizer, pp_groups = create_model(args)

    model_msg = {
        'prompt': "Do you like yourself ?",
        'max_new_tokens': 128, 
        'temperature': 0.2,
        'top_k': 20, 
        'top_p': 0.9, 

    }

    inference(model, tokenizer, pp_groups, model_msg, args)
