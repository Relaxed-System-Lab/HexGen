import torch
import time
import argparse
from loguru import logger
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../utils")
from utils._base import InferenceWorker
from text_generation import Client


class TGIWorker(InferenceWorker):
    def __init__(self, model_name, tgi_addr) -> None:
        super().__init__(f"{model_name}")

        self.tgi_addr = tgi_addr

    async def handle_requests(self, msg):
        prompts = msg.get('prompt', '')
        max_new_tokens = msg.get('max_new_tokens', 128)
        temperature = msg.get('temperature', 0.9)
        top_k = msg.get('top_k', 50)
        top_p = msg.get('top_p', 0.9)

        client = Client(self.tgi_addr)

        torch.cuda.synchronize()
        start = time.time()
        outputs = client.generate(prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p).generated_text
        end = time.time()

        return outputs, end - start

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="tgi_0")
    parser.add_argument("--tgi-addr", type=str, default="http://127.0.0.1:8080")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    worker = TGIWorker(args.model_name, args.tgi_addr)
    worker.start()