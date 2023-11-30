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

# update python to 3.11
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/software/miniconda3
rm -f Miniconda3-latest-Linux-x86_64.sh
echo "export PATH=${HOME}/software/miniconda3/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
conda --version


# install petals
pip install git+https://github.com/bigscience-workshop/petals

# for more RAM
sudo rm -rf ~/.cache/pip

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


# scp ocf and FastGen

# start coordinator first 
export ID=1
export COOR_PORT=`expr $ID + 10005`
cd FastGen-enhance-spec_decoding/
bash petals_coordinator.sh $ID $COOR_PORT 

# start coordinator after
export ID=3
tmux new -s pt_$ID
export COOR_PORT=`expr $ID + 10005`
cd FastGen-enhance-spec_decoding/
bash petals_coordinator.sh $ID $COOR_PORT /ip4/192.168.99.2/tcp/9991/p2p/QmY7XSYgdnRy4nJU27SJ7VJx7HAYWNGhSm9ZamepCucZMJ


# start model
cd FastGen-enhance-spec_decoding/
export CUDA_DEVICE=0
export INITIAL_PEERS=/ip4/192.168.99.2/tcp/10006/p2p/QmWGUaPYJrGm44HGWHBfqRqgWnHKoS8mSytJhBbe3MbpCx
export MODEL_PORT=`expr $CUDA_DEVICE + 17551`
bash petals_model.sh $MODEL_PORT $CUDA_DEVICE $INITIAL_PEERS 10

# in case fail and ports already allocated
sudo docker stop $(sudo docker ps -a -q)
