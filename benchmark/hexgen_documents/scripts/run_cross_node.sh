export PORT=9991
export DEVICES=0,1,2,3

export NUM_NODES=2
export NUM_GPUS_PER_NODE=2
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export MASTER_ADDR=198.176.96.63
export NODE_RANK=0
# export NODE_RANK=1

CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$PORT --node_rank=$NODE_RANK _llama_worker.py \
--model_size llama-7b \
--use-flash-attn \
--hetero_config 1 2 1 \
--pp_partition 8 16 8 \
--model_name "Llama-2-7b-chat-hf" \
--head_node 'http://198.176.96.165:8092' \
--group_id 0
