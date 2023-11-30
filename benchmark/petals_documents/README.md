## Config petals step by step

1. We need to check cuda version, it must be 12.1. If not, cuda have to be installed as follows:

```bash
# delete old version
sudo su
dpkg -l | grep -iE "Cuda|nvidia"  | awk {'print $2'} | xargs apt-get -y remove
dpkg -l | grep -iE "Cuda|nvidia"  | awk {'print $2'} | xargs apt-get -y purge
exit

# this is the only way that works on FluidStack, other frameworks depend
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# after reinstalling, reboot
sudo reboot
```
2. Python version must be 3.11, I install miniconda to overwrite the version of python and for future use

```bash
# for python 3.11
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/software/miniconda3
rm -f Miniconda3-latest-Linux-x86_64.sh
echo "export PATH=${HOME}/software/miniconda3/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# checking
conda --version
```

3. Then install petals and its dependencies. 
  - If you don't use python version of 3.11, your pip tools will be broken and even pip list is not working.  
  - Even if you reinstall pip, the version of huggingface_hub, tokenizer, transformers will be conflicted.

```bash
# install petals
pip install git+https://github.com/bigscience-workshop/petals

# in case the mentioned error really occured 

# reinstall pip in case pip broken
sudo apt remove python3-pip 

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

# reinstall some dependencied in case huggingface-hub version not correct
pip uninstall transformers tokenizers huggingface-hub
pip install transformers tokenizers huggingface-hub
```

4. If you have enough budget of RAM, skip this step

```bash
# for more RAM
sudo rm -rf ~/.cache/pip
```

5. After reinstalling cuda-12.1, you need to install nvidia-docker to run docker image of petals. On FluidStack, however, the normal installing may doesn't work, so you need to restart docker manually

```bash
# install nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# If the above commands doesn't work, run the following for docker gpus not found problem
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

6. Hopefully we can run petals now. We need a coordinator and corresponding worker for every **GPU**. 
  - What's more, except for the head coordinator, other coordinators must cnnect to the first coordinator by specifying the **--initial-peers** argument. 
  - In this step, we start the first coordinator. After doing so, you will get some log and could remember the `--initial-peers` info, it looks like: /ip4/192.168.99.2/tcp/9991/p2p/QmNqNz5moz6J9P6U5waHo65ZQcb79LoRJLYXREJoikS3t8
  - Just use the ip for docker is enough, if it doesn't work, change the ip as public ip. This happen frequently when doing cross machine inference.

```bash
# the first coordinator
PORT_COO=9991
MASTER_PORT=9991
sudo docker run -p $PORT_COO:$PORT_COO --ipc host --gpus all --volume petals-cache:/cache \
--rm learningathome/petals:main python -m petals.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/$PORT_COO --identity_path bootstrap1.id

```

7. After starting the first coordinator, start the others like this. 
  - Just be aware that on the same machine, docker needs different ports for running a image, that's why you have to tune `PORT_COO` each time. 
  - Also, tune the `INITIAL_PEERS` variable as the first coordinator's address. Once you finished the step 6th, you will find it naturally by the logs.

```bash
# petals coordinator
PORT_COO=9998
MASTER_PORT=9991
INITIAL_PEERS=/ip4/192.168.99.2/tcp/9991/p2p/QmNqNz5moz6J9P6U5waHo65ZQcb79LoRJLYXREJoikS3t8
sudo docker run -p $PORT_COO:$PORT_COO --ipc host --gpus all --volume petals-cache:/cache \
--rm learningathome/petals:main python -m petals.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/$PORT_COO --identity_path bootstrap1.id --initial_peers $INITIAL_PEERS
```

8. Then we can start workers for the above coordinators.
  -  Workers also need to specify the `--initial_peers` argument to connect to the coordinator. But unlike coordinators, **every worker connect to its corresponding coordinator, not all to the first coordinator!!**. 
  - You might want to use multiple GPU on the same machine. The correct way is to specify visible devices for the docker command. 
  - Also be aware to tune the `--num_blocks` varibable to control how many layers will be on this worker. 
  - Choose correct `--quant_type`, i.e. the correct quantization you desire. The default config uses `fp16` but you could also tune it.

```bash
# petals model

# change the port manually for docker
MODEL_PORT=28816
# strictly follow this format for docker 
DEVICE='"device=0,"'
# this p2p address varies every time you start a worker
INITIAL_PEERS=/ip4/192.168.99.7/tcp/8992/p2p/QmVYGHDiK6NtMqTNcU266iiUaSoGviCCV8FDdAuRwgFN8a
sudo docker run -p $MODEL_PORT:$MODEL_PORT --ipc host --gpus=''$DEVICE'' --volume petals-cache:/cache \
--rm learningathome/petals:main python -m petals.cli.run_server meta-llama/Llama-2-70b-chat-hf  \
--initial_peers $INITIAL_PEERS --token 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 --num_blocks 10 --quant_type none 
```

# Reminds

1. It's usual that we want to use many GPUs at the same time and you will find it very laberous. Hopefully I have further encapsulated the commands in bash scripts to relieve the pain. 
  - `start_coo_head.sh` starts the head coordinator for petals
  - `start_coo.sh` starts the rest of coordinators by loop
  - `start_model.sh` provides a clue of how to start workers quickly. **But don't run it directly!!**

2. It's better to use tmux to finish multiple start simultaneously. Opening too many terminals in MacOS will crash the system and make your efforts in vain. An example from `start_coo_head.sh` is as followed.

```bash 
# petals_coordinator.sh essentially does the same thing as step 6 and 7
i=1
SESSION_NAME="session_$i"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "export ID=$i" C-m
tmux send-keys -t $SESSION_NAME "export COOR_PORT=\`expr \$ID + 9991\`" C-m
tmux send-keys -t $SESSION_NAME "cd FastGen-enhance-spec_decoding/" C-m
tmux send-keys -t $SESSION_NAME "bash petals_coordinator.sh \$ID \$COOR_PORT" C-m
tmux attach -t session_$i
```

3. If you want to integrate with ocf, run `pip install -r requirements.txt` in this dir to install dependencies. Also, install go, the version must by 1.20. Hopefully you can copy the following commands.

```bash 
# go install
wget -c https://dl.google.com/go/go1.20.linux-amd64.tar.gz -O - | sudo tar -xz -C /usr/local
export PATH=$PATH:/usr/local/go/bin
```

4. Then, don't forget to upload ocf and FastGen. And remember to tune the `--initial_peers` argument in `start_petals.sh`. **It only needs to be the head coordinator's p2p addr**. No need to specify all the coordinator's p2p addr!

5. Coping every time in this file is painful. You could also refer to the bash scripts for the commands! The file name's `config_petals.sh`.

6. From time to time, you may fail for some reason. But ports may not be released. Then you can run the following command to prevent port already allocated error.

```bash 
sudo docker stop $(sudo docker ps -a -q)
```

7. Caching the `--initial_peers` argument for every coordinator maybe a good choice.

8. At last, prepare very large disk in advance!

# Run petals with HexGen

1. After launching `petals`, you could also start a petals worker by HexGen
```bash
# xxx means the --initial_peers for the head coordinator of petals
bash scripts/run_petals.sh xxx
```
