import os
import torch
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from llama_config_utils import llama_config_to_gpt2_config, config_from_checkpoint, overwrite_configs_and_args
from transformers import LlamaForCausalLM, LlamaTokenizer
from remap_state_dict import remap_state_dict_hf_llama

def load_remapped_state_dict(config, checkpoint_path):
    
    """
    Loads and remaps the state dictionary of a pretrained Llama model.

    Parameters:
    - config (dict): Configuration dictionary for the Llama model.

    Returns:
    - dict: Remapped state dictionary suitable for the specific configuration.
    """
    
    state_dict = remap_state_dict_hf_llama(LlamaForCausalLM.from_pretrained(f"{checkpoint_path}").state_dict(), config)
    return state_dict

def save_model_components(config_path, checkpoint_name, checkpoint_path, num_layers, save_dir):
    
    """
    Save specific components and each transformer layer of a model's state dictionary to separate files.

    Args:
    config_path (str): Path to the configuration directory.
    checkpoint_name (str): Name of the model checkpoint.
    num_layers (int): Number of transformer layers in the model.
    save_dir (str): Directory path where the state dictionaries will be saved.

    This function performs the following steps:
    1. Load the configuration and state dictionary for the model.
    2. Save specific components of the state dictionary (embeddings, layer normalization, and language model head).
    3. Iterate over each transformer layer and save its state dictionary separately.
    """

    # Configuration and state dictionary loading
    llama_config = config_from_checkpoint(config_path, checkpoint_name)
    config = llama_config_to_gpt2_config(llama_config)
    state_dict = load_remapped_state_dict(config, checkpoint_path)

    # Saving specific components of the state dictionary to separate files
    torch.save(state_dict['transformer.embeddings.word_embeddings.weight'], f'{save_dir}/embeddings.pt')
    torch.save(state_dict['transformer.ln_f.weight'], f'{save_dir}/ln_f.pt')
    torch.save(state_dict['lm_head.weight'], f'{save_dir}/lm_head.pt')

    # Save the state dictionary of each transformer layer separately
    for idx in range(num_layers):
        layer_key_prefix = f'transformer.layers.{idx}'
        layer_state_dict = {key: value for key, value in state_dict.items() if key.startswith(layer_key_prefix)}
        torch.save(layer_state_dict, f'{save_dir}/layer_{idx}.pt')

def main():
    # Generate model separate state_dicts
    if not os.path.exists("./separate_state_dicts"):
        os.mkdir("./separate_state_dicts")

    save_model_components(
        config_path='../llama-config/',
        checkpoint_name='llama-7b',
        checkpoint_path='../../../../Llama-2-7b-chat-hf/',
        num_layers=32,
        save_dir='./separate_state_dicts/'
    )

if __name__ == "__main__":
    main()
