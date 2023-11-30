## Load Model Parameters for LlaMA 7b

### Overview
This guide provides instructions on how to load model parameters for Llama-7b. It works very similarly for other version of Llama models. Here, we will focus on creating separate state dictionaries for each component and layer of the model.

### Customize Parameters
If you need to specify custom paths, you can manually edit the `create_separate_state_dicts_llama_7b.py` script. Locate the `save_model_components` function call and adjust the paths as needed. For example:

```python
save_model_components(
    config_path='../llama-config/',
    checkpoint_name='llama-7b',
    checkpoint_path='/path/to/Llama-2-7b-chat-hf/',
    num_layers=32,
    save_dir='./separate_state_dicts/'
)
```

Here, your sole requirement is to specify the `checkpoint_path`, as the other parameters have been pre-defined and supplied for your convenience. You can download the model checkpoints from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

### Run the Script
To create the separate state dictionaries for the Llama-7b model, run the following command in the terminal:

```bash
python3 create_separate_state_dicts_llama_7b.py
```

This script will automatically generate and save the state dictionaries in the appropriate directory.

### Verify the Output
After running the script, you should find the separate state dictionaries saved in the designated folder. Verify that all the expected files are present and correctly named.

### Modifying the Inference Script
In the `llama_inference.py` file, add the following code snippet to load the parameters for Llama-7b. Adjust the paths as per your setup:

```python
# Load model checkpoints with respect to hetero_config
tp_ranks_whole_model = hetero_groups['tp_ranks_whole_model']
tp_group_list = hetero_groups['tp_rank_groups']
state_dicts_path = "./load_model_parameters_utils/"
load_model_parameters(model, config, state_dicts_path, tp_ranks_whole_model, tp_group_list, rank)
```
