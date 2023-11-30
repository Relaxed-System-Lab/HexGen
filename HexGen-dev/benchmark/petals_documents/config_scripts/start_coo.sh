#!/bin/bash

INITIAL_PEERS=$1


for i in {2..8}; do
    SESSION_NAME="session_$i"
    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "export ID=$i" C-m
    tmux send-keys -t $SESSION_NAME "export COOR_PORT=\`expr \$ID + 8991\`" C-m
    tmux send-keys -t $SESSION_NAME "cd FastGen-enhance-spec_decoding/" C-m
    tmux send-keys -t $SESSION_NAME "bash petals_coordinator.sh \$ID \$COOR_PORT $INITIAL_PEERS" C-m
done

echo "7 tmux sessions have been initialized."
