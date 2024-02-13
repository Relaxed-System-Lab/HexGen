export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
# Modify the master IP below before execution
export MASTER_ADDR='xxx.xxx.xxx.xx'
export MASTER_PORT=9991
export WORLD_SIZE=6
export RANK=4

CUDA_VISIBLE_DEVICES=0 python3 llama_inference.py \
--model_size llama-7b \
--use-flash-attn \
--hetero_config 4 2 \
--pp_partition 20 12 \
