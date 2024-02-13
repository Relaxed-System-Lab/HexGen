## Run HexGen worker

1. HexGen worker can be launched on single machine or multi machines. To start it, simply run the corresponding scripts. Remember to properly specify port, uesd devices and master addr. An example is shown below.

```bash
# for single machine
bash scripts/run_llama.sh
# for multi mahcines
bash scripts/run_cross_node.sh
```

2. Besides for common parameters, HexGen worker requires you to
    - Set `--head_node` as the format of `'http://<ip>:<port>'`, you could replace the IP address as your own head coordinator.
    - Set `--model_name` as the name you wish users to call, for example `--model_name "Llama-2-70b-chat-hf"`.
    - Set `--group_id` uniquely, in case starting multiple services on a single nood. 

3. After started up, a HexGen worker uses coroutine techniques to hang there and wait for incoming requests. When sending requests, it is forced to add a suffix `_0` to the previous declared `--model_name`, i.e. call `"Llama-2-70b-chat-hf_0"` for inference request.

4. If you have multiple service on a single head node, it will dispatch requests by roubin robin method, i.e. the first request goes to the first model replica, the second request goes to the second model replica...

5. If you run into cases that different machines have different number of GPUs and `torch.distributed.launch` doesn't work, please refer to `../../hexgen/llama/scripts/run_llama_p0.sh`.

6. It is also supported to run multiple instances on a single worker, just differentiate them by `group_id`.

## HexGen Asymmetric Parallel Implementation

The source code for the asymmetric parallel implementation is detailed in `hexgen/hexgen_core/heterogeneous_pipeline.py`. You can enable the HexGen asymmetric parallel on heterogeneous clusters as follows:

1. Switch to the hexgen llama folder:

```bash
cd hexgen/llama
```

2. Rewrite the `run_llama_p{gpu_index}.sh` file to specify the master address, world size, rank, and CUDA visible device as needed:

```bash
# Modify the master IP below before execution
export MASTER_ADDR='xxx.xxx.xxx.xx'
export WORLD_SIZE={world_size}
export RANK={gpu_index}
export CUDA_VISIBLE_DEVICES={gpu_index}
```

Note that the `RANK` and `CUDA_VISIBLE_DEVICES` should be different on different GPUs.

3. Specify the `hetero_config` and `pp_partition`; the length of these two variables should be the same:

```bash
--hetero_config {tp_degree_of_pp_stage_1} {tp_degree_of_pp_stage_2} {tp_degree_of_pp_stage_3} ... \
--pp_partition {layer_number_of_pp_stage_1} {layer_number_of_pp_stage_2} {layer_number_of_pp_stage_3} ... \
```

Each element of the `hetero_config` represents a pipeline stage with certain tensor model parallel degree.
Note that the pipeline partition should be tuned with respect to the memory limit of each device to avoid an Out-Of-Memory problem.

4. Execute the following scripts on the respective machines:

```base
# On machine A
bash scripts/run_llama_p{gpu_index}.sh
...
# On machine B
bash scripts/run_llama_p{gpu_index}.sh
...
# On machine C
bash scripts/run_llama_p{gpu_index}.sh
...
# On machine D
...
```

5. The execution of the following scripts serves as an example: we use 4 A6000 GPUs, 2 A5000 GPUs, and 2 A4000 GPUs to implement an asymmetric parallel setup:

On each device, we set the environment variables for the master address, port, and world size by exporting them:

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
# Modify the master IP below before execution
export MASTER_ADDR='xxx.xxx.xxx.xx'
export MASTER_PORT=9991
export WORLD_SIZE=8
```

Specify `CUDA_VISIBLE_DEVICES` and `RANK` for each GPU:

```bash
# On device 1 with 4 A6000 GPUs.
export RANK=0
CUDA_VISIBLE_DEVICES=0 python3 llama_inference.py \
export RANK=1
CUDA_VISIBLE_DEVICES=1 python3 llama_inference.py \
export RANK=2
CUDA_VISIBLE_DEVICES=2 python3 llama_inference.py \
export RANK=3
CUDA_VISIBLE_DEVICES=3 python3 llama_inference.py \
# On device 2 with 2 A5000 GPUs.
export RANK=4
CUDA_VISIBLE_DEVICES=0 python3 llama_inference.py \
export RANK=5
CUDA_VISIBLE_DEVICES=1 python3 llama_inference.py \
# On device 3 with 2 A4000 GPUs.
export RANK=6
CUDA_VISIBLE_DEVICES=0 python3 llama_inference.py \
export RANK=7
CUDA_VISIBLE_DEVICES=1 python3 llama_inference.py \
```

Specify the model size:

```bash
--model_size llama-70b \
```

Specify parallel strategy and pipeline layer distribution:

```bash
--hetero_config 4 2 2 \
--pp_partition 44 20 16 \
```

The upper case utilizes an **asymmetric parallel setup**, which employs 4 A6000 GPUs to serve the first stage of the pipeline with a tensor parallel degree of 4, and uses 2 A5000 GPUs and 2 A4000 GPUs to serve the subsequent stages, each with a tensor parallel degree of 2. Note that the 2 A4000 GPUs, with a 16 GB memory limit, can only support 16 Llama2-70B layers. Therefore, we adjust the number of layers for the last stage to be 16 (**uneven layer distribution**).



