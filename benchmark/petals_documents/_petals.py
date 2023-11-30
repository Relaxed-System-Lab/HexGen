import torch
import time
import argparse
from loguru import logger
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSequenceClassification
from transformers import TextGenerationPipeline, TextClassificationPipeline
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../utils")
from utils._base import InferenceWorker
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

pipeline_mapping = {
    'text-generation': TextGenerationPipeline,
    'text-classification': TextClassificationPipeline,
}

model_mapping = {
    'text-generation': AutoModelForCausalLM,
    'text-classification': AutoModelForSequenceClassification,
}

dtype_mapping = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

class PetalsWorker(InferenceWorker):
    def __init__(self, model_name, init_peers, token, id) -> None:
        super().__init__(f"{model_name}_{id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if init_peers == 'default':
            self.model = AutoDistributedModelForCausalLM.from_pretrained(model_name, token=token)
        else:
            self.model = AutoDistributedModelForCausalLM.from_pretrained(model_name, initial_peers=[init_peers], token=token)

    async def handle_requests(self, msg):
        prompts = msg.get('prompt', '')
        max_new_tokens = msg.get('max_new_tokens', 128)
        temperature = msg.get('temperature', 0.9)
        top_k = msg.get('top_k', 50)
        top_p = msg.get('top_p', 0.9)
        # if prompt is str:
        if isinstance(prompts, str):
            prompts = [prompts]

        print(prompts)
        outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            torch.cuda.synchronize()
            start = time.time()
            output = self.model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            output = self.tokenizer.decode(output[0])
            end = time.time()
            outputs.append(output)
        return outputs[0], end - start

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="openlm-research/open_llama_7b")
    parser.add_argument("--init-peers", type=str, default="default")
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()
    logger.info(f"args: {args}")
    worker = PetalsWorker(args.model_name, args.init_peers, args.token, args.id)
    worker.start()