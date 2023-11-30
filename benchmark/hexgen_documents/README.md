## Run HexGen worker

1. HexGen worker can be launched on single machine or multi machines. To start it, simply run the corresponding scripts. Remember to properly specify port, uesd devices and master addr. An example is shown below.

```bash
# for single machine
bash scripts/run_llama.sh
# for multi mahcines
bash scripts/run_cross_node.sh
```

2. Besides for common parameters, HexGen worker requires you to
    - Set `--head_node` as the format of `'http://198.176.96.165:8092'`, you could replace the IP address as your own head coordinator.
    - Set `--model_name` as the name you wish users to call, for example `--model_name "Llama-2-7b-chat-hf"`.
    - Set `--group_id` uniquely, in case starting multiple services on a single nood. 

3. After started up, a HexGen worker uses coroutine techniques to hang there and wait for incoming requests. When sending requests, it is forced to add a suffix `_0` to the previous declared `--model_name`, i.e. call `"Llama-2-7b-chat-hf_0"` for inference request.

4. If you have multiple service on a single head node, it will dispatch requests by roubin robin method, i.e. the first request goes to the first model replica, the second request goes to the second model replica...

5. If you run into cases that different machines have different number of GPUs and `torch.distributed.launch` doesn't work, please refer to `../../hexgen/llama/scripts/run_llama_p0.sh`.

6. It is also supported to run multiple instances on a single worker, just differentiate them by `group_id`.
