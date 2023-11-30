## Config tgi step by step

1. Install nvidia-docker

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

2. Run the docker command, tune the quantization, data type, num shards(how many GPUs to be used), cuda memory fraction, port (If planning to start multiple instances within one machine). Refer to `launcher.md` listed in tgi's repo for more details. Following command is an example.

```bash
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your cli READ token>

docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model --num-shard 4 --dtype float16 --cuda-memory-fraction 0.45
```

3. Run the worker, this step assumes you have started work coordinator.

```bash
bash scripts/run_tgi.sh
```
