import os
import sys
sys.path.insert(0, '..')
import torch
from torch import nn
import numpy as np
from typing import Any
from hexgen_core.heterogeneous_pipeline import PipelineParallel, PipeSequential
from hexgen_core.modules.block import Block
import torch.distributed as dist
import math

def construct_hybrid_parallel_model(
        model: Any,
        model_config: Any,
        inference_args: Any,
        hybrid_parallel_configs: Any,
        pp_partition: Any,
        device: Any,
        hetero_config: Any) -> Any:

    """
    Constructs a hybrid parallel model for inferencing large-scale models Llama.
    This function integrates various parallelism techniques, including tensor parallelism (TP), 
    and pipeline parallelism (PP) to efficiently distribute the model across multiple devices and nodes.

    Args:
        model: The base model architecture.
        model_config: Configuration parameters specific to the model architecture.
        inference_args: Arguments specific to the inference process, such as sequence length and hidden size.
        hybrid_parallel_configs: Configuration parameters for hybrid parallelism, including degrees of TP and PP.
        pp_partition: Information about the partitioning of the model for pipeline parallelism.
        device: The device to deploy the model on (e.g., CPU, GPU).
        hetero_config: Configuration for heterogeneous settings in the model, accommodating different computational capabilities.

    The function performs the following steps:
    1. Extracts and sets up the required configurations for different types of parallelism.
    2. Initializes the configurations for TP and PP across the entire model.
    3. Generates heterogeneous groups and communication lists for effective model parallelism.
    4. Constructs the model layers using specialized Llama modules.
    5. Defines the output tensor shapes, data types, and sizes for each model layer under the parallelism setup.
    6. Assembles the final hybrid parallel model ready for training.

    Returns:
        A PipelineParallel model that integrates TP and PP for efficient large-scale model inference.
    """
    
    world_size = torch.distributed.get_world_size()
    gpt_model = model
    config = model_config
    args = inference_args
    hp_configs = hybrid_parallel_configs
    pp_deg = hp_configs['pp_deg']
    tp_sizes_enc = hp_configs['tp_sizes_enc']
    tp_consecutive_flags = hp_configs['tp_consecutive_flags']
    dp_types_enc = hp_configs['dp_types_enc']
    pp_ranks_enc = hp_configs['pp_ranks_enc']
    tp_sizes_whole_model = [1] + tp_sizes_enc + [1, 1]
    dp_types_whole_model = [0] + dp_types_enc + [0, 0]
    pp_ranks_whole_model = [0] + pp_ranks_enc + [pp_deg - 1] * 2
    tp_consecutive_whole_model = [1] + tp_consecutive_flags + [1, 1]

    from hexgen_core import gen_hetero_groups
    hetero_groups = gen_hetero_groups(hetero_config=hetero_config, pp_partition=pp_partition, layer_num=config.n_layer)
    tp_group_list, pp_groups, pp_ranks_whole_model = hetero_groups['tp_rank_groups'], hetero_groups['pp_rank_groups'], hetero_groups['pp_ranks_whole_model']

    from hexgen_core import generate_send_recv_lists
    send_list, recv_list, send_empty_list, recv_empty_list = generate_send_recv_lists(pp_groups, pp_groups[0])
    p2p_lists = [send_list, recv_list, send_empty_list, recv_empty_list]
    def format_list(lst):
        return '\n'.join(['    - ' + str(item) for item in lst])

    separator = '=' * 80
    tp_separator = '-' * 80
    pp_separator = '-' * 80

    if dist.get_rank() == 0:
        print(separator)
        print('Heterogeneous Parallel Configuration'.center(80))
        print(separator)
        print('Tensor Parallel Groups'.center(80))
        print(tp_separator)
        for tp_group in tp_group_list:
            print('TP Group:', tp_group)
        print('Overall TP Groups:\n', format_list(tp_group_list))
        print(separator)
        print('Pipeline Parallel Groups'.center(80))
        print(pp_separator)
        for pp_group in pp_groups:
            print('PP Group:', pp_group)
        print('Overall PP Groups:\n', format_list(pp_groups))
        print(separator)
        print('Pipeline Parallel Layer to Stage Mapping'.center(80))
        print(pp_separator)
        print(pp_ranks_whole_model)
        print(separator)
        print('Tensor Parallel Layer to Degree Mapping'.center(80))
        print(pp_separator)
        print(hetero_groups['tp_ranks_whole_model'])
        print(separator)
        print('P2p Lists'.center(80))
        print(pp_separator)
        print('Send List:', send_list)
        print('Recv List:', recv_list)
        print('Send Data Option:', {k: [not item for item in v] for k, v in send_empty_list.items()})
        print('Recv Data Option:', {k: [not item for item in v] for k, v in recv_empty_list.items()})
        print(separator)

    # Generate pp_indices: like 24 layers anad 5 layers per stage: [0,5,10,15,20]
    # num_layers_per_stage = math.ceil(config.num_hidden_layers / len(pp_groups[0]))
    # pp_indices = [i for i in range(0, config.num_hidden_layers, num_layers_per_stage)]
    
    def find_first_indices(lst):
        indices = {}
        result = []
        for i, value in enumerate(lst):
            if value not in indices:
                indices[value] = i
                # Subtract 1 from the index for all values except for 0
                result.append(i if value == 0 else i-1)
        return result
    pp_indices = find_first_indices(pp_ranks_whole_model)

    if dist.get_rank() == 0:
        print('The First PP Layer Index of Each Stage'.center(80))
        print(pp_separator)
        print('List:', pp_indices)
        print(separator)

    from Llamamodel_pipeline import LlamaEmbeddings_, LlamaLayers_, LlamaPreNorm_, LlamaCls_
    model = PipeSequential()
    model.add_module('embeddings', LlamaEmbeddings_(gpt_model))
    for i in range(config.num_hidden_layers):
        enc = LlamaLayers_(gpt_model, i, i + 1, pp_indices)
        model.add_module('layer_%d'%i, enc)
    model.add_module('prenorm', LlamaPreNorm_(gpt_model))
    model.add_module('cls', LlamaCls_(gpt_model))

    seq_len, hidden_size = args.seq_length, args.hidden_size
    layer_output_tensor_shapes = [None] + [[[-1,seq_len,hidden_size], [-1,seq_len]]] * config.num_hidden_layers + [None] * 2
    mixed_precision = {'fp32': torch.float, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.mixed_precision]
    layer_output_tensor_dtypes = [None] + [[mixed_precision, torch.long]] * config.num_hidden_layers + [None] * 2
    layer_dp_sizes = [world_size // pp_deg // tp_size for tp_size in tp_sizes_whole_model]
    
    # Some hints
    # pp_ranks_whole_model = [0] + [0,1,2] + [2,2]
    # pp_groups = [[0,2,4],[1,3]]
    # pp_groups = [[0,2,3],[1,2,4],[0,2,3],[0,2,3],[1,2,4]]
    # broadcast_group = dist.new_group([3,4])

    hp_model = PipelineParallel(
                model = model, 
                model_ranks = pp_ranks_whole_model, 
                layer_output_tensor_shapes = layer_output_tensor_shapes, 
                layer_output_tensor_dtypes = layer_output_tensor_dtypes,
                layer_dp_sizes = layer_dp_sizes, 
                p2p_lists = p2p_lists,
                process_group = hetero_groups['current_pp_group'],
                broadcast_group = hetero_groups['current_tp_group'],
                broadcast_group_list = hetero_groups['current_tp_rank_group'],
                nproc_per_node=8,
                info=False,
                show_process=False)

    module_types = ['embed'] + ['gpt_dec']*config.num_hidden_layers + ['norm', 'cls']
    hp_model.wrap_pipeline_modules_data_parallel(
            dp_types_whole_model,
            hetero_groups['process_groups_whole_model'],
            module_types=module_types,
            mixed_precision=mixed_precision,
            wrap_block_name=[Block])
    return hp_model

def get_hybrid_parallel_configs(args):
    pp_deg = 1 
    num_layers = args.num_hidden_layers
    global_tp_deg = 1
    global_tp_consec = 1 

    tp_sizes_enc = [global_tp_deg if global_tp_deg > 0 else 1] * num_layers
    tp_consecutive_flags = [global_tp_consec if global_tp_consec in [0, 1] else 1] * num_layers
    dp_types_enc = [0] * num_layers

    avg_num_layers = num_layers // pp_deg
    pp_ranks_enc = [i for i in range(pp_deg) for _ in range(avg_num_layers)]

    return {
        'pp_deg': pp_deg,
        'tp_sizes_enc': tp_sizes_enc,
        'tp_consecutive_flags': tp_consecutive_flags,
        'dp_types_enc': dp_types_enc,
        'pp_ranks_enc': pp_ranks_enc
    }

def overwrite_megatron_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.max_position_embeddings = config.max_position_embeddings
    args.use_cpu_initialization = True
