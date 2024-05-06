HexGen presents a versatile framework capable of facilitating Llama-2 inference, integrating **hybrid model parallelism** along with **an automated mechanism for request dispatch**. Key features include:
- Comprehensive support for hybrid pipeline parallelism as well as tensor parallelism.
- Ocf, a seamlessly integrated subsystem, is dedicated to coordinating and efficiently dispatching requests.

----------

## Content

- [Building Environment](#building-environment)
    - [Establish A Personal Head Node Coordinator](#establish-a-personal-head-node-coordinator)
    - [Incorporate Additional Worker Nodes](#incorporate-additional-worker-nodes)
- [Loading Model Parameters for Llama Models](#loading-model-parameters-for-llama-models)
    - [Create Separate Model State Dicts](#create-separate-model-state-dicts)
    - [Load Model Parameters](#load-model-parameters)
    - [Load Model Parameters for Llama2-70b](#load-model-parameters-for-llama2-70b)
- [Starting HexGen](#starting-hexgen)
    - [Activating Head Node Coordinator](#activating-head-node-coordinator)
    - [Activating Worker Nodes](#activating-worker-nodes)
    - [Activating Independent Inference Process](#activating-independent-inference-process)
- [Asymmetric Parallel Group Support in HexGen](#asymmetric-parallel-group-support-in-hexgen)
    - [Tensor Model Parallelism and Pipeline Parallelism](#tensor-model-parallelism-and-pipeline-parallelism)
    - [Asymmetric Parallel Group Support](#asymmetric-parallel-group-support)
- [Performance Results](#performance-results)
- [Acknowledgements](#acknowledgements)


## Building Environment

HexGen stipulates the utilization of CUDA version 11.8 and Python version 3.11 or above. The assembly of HexGen is designed to be efficient and accessible:

### Only Establish A Personal Head Node Coordinator

```bash
make hexgen-head
```

### Incorporate Additional Worker Nodes

```bash 
make hexgen
```

## Loading Model Parameters for Llama Models

Navigate to the `hexgen/llama/load_model_parameters_utils` directory. Here, you will initiate the process of setting up parameters for the model.

### Create Separate Model State Dicts

For scenarios where specific custom paths are required, modifications to the `create_separate_state_dicts_llama_7b.py` script are necessary. In this script, locate the function call to `save_model_components`. You can then alter the paths according to your specific requirements. For instance:

```python
save_model_components(
    config_path='../llama-config/',
    checkpoint_name='llama-70b',
    checkpoint_path='/path/to/Llama-2-70b-chat-hf/',
    num_layers=80,
    save_dir='./separate_state_dicts/'
)
```

Here, your sole requirement is to specify the `checkpoint_path`, as the other parameters have been pre-defined and supplied for your convenience. You can download the model checkpoints from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

A recommended way to download is:

```bash
huggingface-cli download --resume-download meta-llama/Llama-2-7b-chat-hf --local-dir Llama-2-7b-chat-hf --token <your token>
```

To create the separate state dictionaries for the Llama-7b model, run the following command in the terminal:

```bash
python3 create_separate_state_dicts_llama_7b.py
```

This script will automatically generate and save the state dictionaries in the appropriate directory.

### Load Model Parameters

In the `llama_inference.py` file, add the following code snippet to load the parameters for Llama-7b. Adjust the paths as per your setup:

```python
# Load model checkpoints with respect to hetero_config
tp_ranks_whole_model = hetero_groups['tp_ranks_whole_model']
tp_group_list = hetero_groups['tp_rank_groups']
state_dicts_path = "./load_model_parameters_utils/"
load_model_parameters(model, config, state_dicts_path, tp_ranks_whole_model, tp_group_list, rank)
```

### Load Model Parameters for Llama2-70b


The Llama2-70b model features eight key and value heads, differing from the 7b and 13b models. To accommodate this configuration, it is necessary to manually specify the appropriate settings in the `hexgen/llama/llama_config_utils.py` file:

```python
def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config:
    return GPT2Config(
        ...
        n_inner=28672,
        n_head_kv=8,
    )
```

## Starting HexGen

### Activating Head Node Coordinator

HexGen can be launched in head node coordinator modes by:

```bash
bash scripts/run_head.sh
```

### Activating Worker Nodes

HexGen can be launched in worker modes by a similar command, except that you should modify the file `./third_party/ocf/ocf-core/config/cfg.yaml`, the p2p addr should be as similar format as `"/ip4/{Pubilc_IP}/tcp/43905/p2p/{Peer_ID}"`, you could replace `{Public_IP}` as your own head coordinator's IP address and `{Peer_ID}` as its peer ID:

```bash
bash scripts/run_worker.sh
```

### Activating Independent Inference Process

To initiate an independent inference process without involving the coordinator, navigate to the `hexgen/llama` directory and execute the following command:

```bash
bash scripts/run_llama_inference.sh
```

You have the flexibility to customize various inputs to tailor your inference task according to your specific requirements. The `model_msg` object can be adjusted with different parameters, as shown in the example below:

```python
model_msg = {
    'prompt': "Do you like yourself ?",  # Define your own prompt here
    'max_new_tokens': 128,               # Set the maximum number of new tokens
    'temperature': 0.2,                  # Adjust the randomness in response generation
    'top_k': 20,                         # Specify the number of highest probability vocabulary tokens to keep for top-k sampling
    'top_p': 0.9,                        # Set the cumulative probability threshold for top-p (nucleus) sampling
}
```

## Asymmetric Parallel Group Support in HexGen

### Tensor Model Parallelism and Pipeline Parallelism

Two methods are used to distribute the workload of training large deep learning models across multiple computing units.

- **Tensor Model Parallelism** splits the model's layers or components across different processors.
- **Pipeline Parallelism** divides the process into different stages, with each stage being processed on a different set of processors.

### Asymmetric Parallel Group Support

HexGen introduces a novel approach with its Asymmetric Parallel Group Support, driven by two critical parameters: `--hetero_config` and `--pp_partition`.

- `--hetero_config`: This parameter allows for the specification of varying TP degrees for each pipeline stage. For instance, a setting like `4 2 2` configures a three-stage pipeline with respective TP degrees of 4, 2, and 2, showing HexGen's adaptability.
- `--pp_partition`: This parameter complements `--hetero_config` by managing the distribution of model layers across the pipeline stages. A combination like `40 20 20` with a `hetero_config` of `4 2 2` signifies an optimized layer distribution, illustrating HexGen's capability for precision tuning according to model needs and hardware constraints.

HexGen can be launched with asymmetric parallel group by:

```python
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 9996 llama_inference.py \
--model_size llama-70b \
--hetero_config 2 2 4 \
--pp_partition 20 20 40 \
```

## Performance Results
Experimental findings provide a detailed comparison of cost-performance trade-offs across different configurations and implementations, with a focus on achieving Service Level Objectives (SLO) in various environments:

- HexGen with Asymmetric Parallel Group Support (Full Budget, Heterogeneous Setting): This configuration can achieve up to 2.3× lower latency and can handle peak request rates up to 4× higher than FlashAttention in a homogeneous setting, demonstrating significant performance enhancement.
- HexGen (Half Budget, Heterogeneous Setting): Even with a halved budget, HexGen can still slightly outperform FlashAttention in a homogeneous environment, showcasing its ability to efficiently utilize heterogeneous GPUs.
- Asymmetric vs. Symmetric Parallelism in HexGen (Full Budget, Heterogeneous Setting): The integration of asymmetric parallelism into HexGen can lead to up to 1.8× improvement in meeting lower latency deadlines and can manage peak traffic rates up to 2× higher than its symmetric parallelism counterpart.
