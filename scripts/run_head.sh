# run coordinator
SESSION_NAME="coordinator"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd ./third_party/ocf/src/ocf-core/" C-m
tmux send-keys -t $SESSION_NAME "build/core start --config config/cfg_standalone.yaml" C-m