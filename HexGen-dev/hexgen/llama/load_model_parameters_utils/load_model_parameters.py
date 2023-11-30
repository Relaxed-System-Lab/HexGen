import torch
from hexgen_core.models.gpt import shard_state_dict_tp

def load_weights(
    layer, state_dict,
    layer_idx, config
):
    """Loads all layer weights from the state dictionary.
    
    Args:
        layer: The layer object to load weights into.
        state_dict (dict): The nested dictionary containing all state data.
        layer_idx (int): The index of the layer.
        config: The configuration object.
    Returns:
        layer: The layer object with loaded weights.
    """
    
    # Construct the base string for key access
    base_str = f'transformer.layers.{layer_idx}'

    # Load mixer weights
    layer.mixer.Wqkv.weight.data.copy_(state_dict[f'{base_str}.mixer.Wqkv.weight'])
    layer.mixer.out_proj.weight.data.copy_(state_dict[f'{base_str}.mixer.out_proj.weight'])
    
    # Load mlp weights
    layer.mlp.fc1.weight.data.copy_(state_dict[f'{base_str}.mlp.fc1.weight'])
    
    if config.activation_function in ["glu", "swiglu", "geglu"]:
        layer.mlp.fc2.weight.data.copy_(state_dict[f'{base_str}.mlp.fc2.weight'])
    
    # Load normalization layer weights and biases
    layer.norm1.weight.data.copy_(state_dict[f'{base_str}.norm1.weight'])
    layer.norm2.weight.data.copy_(state_dict[f'{base_str}.norm2.weight'])
    
    # Set dropout probabilities to 0.0
    layer.dropout1.p = 0.0
    layer.dropout2.p = 0.0
    layer.mixer.inner_attn.drop.p = 0.0
    layer.mixer.inner_cross_attn.drop.p = 0.0
    
    return layer


def load_model_parameters(model, config, state_dicts_path, tp_ranks_whole_model, tp_group_list, rank):
    
    """
    Loads and applies specific model parameters from external files to different components of a given model.

    Parameters:
    - model (PyTorch Model): The neural network model whose parameters are to be updated.
    - config (dict): A configuration dictionary containing model-specific settings.
    - tp_ranks_whole_model (list): A list of tensor parallel (TP) ranks for the entire model. This is used to determine the layer sizes for sharding.
    - tp_group_list (list of lists): A nested list where each sublist represents a tensor parallel group, used to compute the TP rank.
    - rank (int): The rank of the current process in a distributed training setup, used to index into the TP rank list.

    This function iterates through each module of the model's current stage. Depending on the type of module (identified by its label), it loads different parameters:
    - For embedding layers, it loads embedding weights.
    - For transformer layers, it loads layer-specific weights and applies rotary position embeddings.
    - For the final layer normalization and language model head, it loads their respective weights.

    Note:
    - The function assumes the existence of specific .pt files in 'separate_state_dicts' directory for each type of parameter.
    - Functions 'shard_state_dict_tp' and 'load_weights' need to be defined and available in the scope.
    - The actual structure and labeling of the model components must align with the logic used in the conditional statements.
    """
    
    layer_tp_sizes = tp_ranks_whole_model[1:-2]
    tp_rank = [i for sub_array in tp_group_list for i in range(len(sub_array))]

    for module in model.model_cur_stage:
        if module.label()[0] == 0:
            embed_data = torch.load(f'{state_dicts_path}/separate_state_dicts/embeddings.pt')
            module.embeddings.word_embeddings.weight.data.copy_(embed_data)
        elif module.label()[0] == 1:
            layer = module.layers[0]
            layer_state_dict = torch.load(f'{state_dicts_path}/separate_state_dicts/layer_{module.label()[1]}.pt')
            buffer_data = torch.load(f'{state_dicts_path}/inv_freq.pt')
            layer.mixer.rotary_emb.inv_freq.copy_(buffer_data)
            layer_state_dict = shard_state_dict_tp(layer_state_dict, config, layer_tp_sizes[module.label()[1]], tp_rank[rank])
            load_weights(layer, layer_state_dict, module.label()[1], config)
        elif module.label()[0] == 2:
            ln_f_data = torch.load(f'{state_dicts_path}/separate_state_dicts/ln_f.pt')
            module.ln_f.weight.data.copy_(ln_f_data)
        else:
            lm_head_data = torch.load(f'{state_dicts_path}/separate_state_dicts/lm_head.pt')
            module.lm_head.weight.data.copy_(lm_head_data)
