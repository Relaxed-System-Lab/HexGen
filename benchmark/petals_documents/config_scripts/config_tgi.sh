# nvidia 12.1 install
sudo su
dpkg -l | grep -iE "Cuda|nvidia"  | awk {'print $2'} | xargs apt-get -y remove
dpkg -l | grep -iE "Cuda|nvidia"  | awk {'print $2'} | xargs apt-get -y purge
exit

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

sudo reboot


# install nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# for docker gpus not found
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# go install
wget -c https://dl.google.com/go/go1.20.linux-amd64.tar.gz -O - | sudo tar -xz -C /usr/local
export PATH=$PATH:/usr/local/go/bin
# source ~/.profile

# ocf start
cd ocf-enhance-bind_wallet/src/ocf-core
git init
make build
build/core start --config config/cfg.yaml




model=meta-llama/Llama-2-70b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=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

sudo docker run --gpus '"device=0,1,2,3"' --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 9090:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model --num-shard 4 --dtype float16 --cuda-memory-fraction 0.45