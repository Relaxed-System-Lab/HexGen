

# tune i, peers, blocks mannually at each time.

# selecting which gpu, for example, from 0-7 if your machine has 8 GPUs
i=3
# corresponding coordinator's addr 
# change this in real settings
INITIAL_PEERS=/ip4/xxx.xxx.xxx/tcp/8995/p2p/QmbWW8jCpqLsAFjYMVLyhYWoMLet3iV8JZkJUmzK1b1eF4
# how many layers you want to serve on this gpu
BLOCKS=10
SESSION_NAME="model_$i"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd FastGen-enhance-spec_decoding/" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_DEVICE=$i" C-m
tmux send-keys -t $SESSION_NAME "export INITIAL_PEERS=$INITIAL_PEERS" C-m
tmux send-keys -t $SESSION_NAME "export MODEL_PORT=\`expr \$CUDA_DEVICE + 59551\`" C-m
tmux send-keys -t $SESSION_NAME "bash petals_model.sh \$MODEL_PORT \$CUDA_DEVICE \$INITIAL_PEERS $BLOCKS" C-m

tmux attach -t model_3
tmux attach -t session_3