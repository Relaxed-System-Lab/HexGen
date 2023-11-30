VAR=$1
PORT_COO=$2
MASTER_PORT=9991

if [[ $VAR -eq 1 ]]
then
  # the first coordinator
  sudo docker run -p $PORT_COO:$PORT_COO --ipc host --gpus all --volume petals-cache:/cache \
  --rm learningathome/petals:main python -m petals.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/$PORT_COO --identity_path bootstrap1.id
else
  # petals coordinator
  INITIAL_PEERS=$3
  sudo docker run -p $PORT_COO:$PORT_COO --ipc host --gpus all --volume petals-cache:/cache \
  --rm learningathome/petals:main python -m petals.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/$PORT_COO --identity_path bootstrap1.id --initial_peers $INITIAL_PEERS
fi






/ip4/192.168.99.2/tcp/9992/p2p/QmaPtUnaY39LxQ8bXM4h1APeegQJo1BJzrkXWvVMCpShnp