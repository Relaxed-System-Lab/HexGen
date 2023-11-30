## Init Inference Tasks

To initiate an independent inference process without involving the coordinator, execute the following command:

```bash
bash scripts/run_llama_inference.sh
```

HexGen supports multi-process scenarios without relying on `torch.distributed.launch` for initialization. This is achieved by manually starting HexGen on each process across different machines. For instance, in a setup with 6 processes—4 on one machine and 2 on another—specific environment variables are exported for automatic detection by HexGen. The setup can be executed as follows:

```bash
# on machine A
bash scripts/run_llama_p0.sh
bash scripts/run_llama_p1.sh
bash scripts/run_llama_p2.sh
bash scripts/run_llama_p3.sh
# on machine B
bash scripts/run_llama_p4.sh
bash scripts/run_llama_p5.sh
```

Exercise caution with the `CUDA_VISIBLE_DEVICES` setting, as handling 6 processes on a single machine differs from managing them across multiple machines.

You have the flexibility to customize various inputs to tailor your inference task according to your specific requirements. The `model_msg` object can be adjusted with different parameters, as shown in the example below:

```python
model_msg = {
    'prompt': "Do you like yourself ?",  # Define your own prompt here
    'max_new_tokens': 128,               # Set the maximum number of new tokens
    'temperature': 0.2,                  # Adjust the randomness in response generation
    'top_k': 20,                         # Specify the number of highest probability vocabulary tokens to keep for top-k sampling
    'top_p': 0.9,                        # Set the cumulative probability threshold for top-p (nucleus) sampling
}
```
