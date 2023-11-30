# petals model
MODEL_PORT=$1
DEVICE=$2
INITIAL_PEERS=$3
BLOCKS=$4
VISIBLE_DEVICE='"device='$DEVICE',"'
sudo docker run -p $MODEL_PORT:$MODEL_PORT --ipc host --gpus=''$VISIBLE_DEVICE'' --volume petals-cache:/cache \
--rm learningathome/petals:main python -m petals.cli.run_server meta-llama/Llama-2-70b-chat-hf  \
--initial_peers $INITIAL_PEERS --token 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 --num_blocks $BLOCKS --quant_type none
