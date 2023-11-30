# 
i=1
SESSION_NAME="session_$i"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "export ID=$i" C-m
tmux send-keys -t $SESSION_NAME "export COOR_PORT=\`expr \$ID + 9991\`" C-m
tmux send-keys -t $SESSION_NAME "cd FastGen-enhance-spec_decoding/" C-m
tmux send-keys -t $SESSION_NAME "bash petals_coordinator.sh \$ID \$COOR_PORT" C-m
tmux attach -t session_$i