import torch
import argparse
from loguru import logger
import sys
sys.path.insert(0, "..")
sys.path.insert(0, '../utils')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../hexgen')
sys.path.insert(0, '../../hexgen/hexgen_core')
sys.path.insert(0, '../../hexgen/llama')
sys.path.insert(0, '../../hexgen/llama/modules')
sys.path.insert(0, '../../hexgen/llama/llama-config')
sys.path.insert(0, '../../third_party/megatron')
from _base_rank_based import InferenceWorker
from llama.arguments import add_arguments, clear_kv_cache
from llama.llama_inference import inference, create_model, set_seed
from megatron.initialize import initialize_megatron
from megatron import get_args
from threading import Thread
from multiprocessing import Process

class LlamaWorker(InferenceWorker):
    def __init__(self, model_name, head_node, args, ) -> None:
        self.head_node = head_node

        self.args = args
        self.rank = args.rank
        self.world_size = args.world_size
        self.model, self.tokenizer, self.pp_groups = create_model(args)

        super().__init__(model_name, head_node, self.rank, self.rank, args=args)

    async def handle_requests(self, msg):

        model_msg = self.parse_msg(msg)

        if self.rank == 0:
            threads = []
            for rank in range(self.rank + 1, self.world_size):
                threads.append(Thread(target=self.send_request, args=(msg, rank)))

            for t in threads:
                t.start()

            print(f"On {self.rank}, Start inference")
            outputs, infer_time = inference(self.model, self.tokenizer, self.pp_groups, model_msg, self.args)
            
        else:
            print(f"On {self.rank}, Start inference")
            outputs, infer_time = inference(self.model, self.tokenizer, self.pp_groups, model_msg, self.args)

        clear_kv_cache()
        return outputs, infer_time   

    def get_rank(self):
        return self.rank


if __name__=="__main__":

    initialize_megatron(extra_args_provider=add_arguments)
    args = get_args()

    model_name = args.model_name
    head_node = args.head_node

    set_seed()
    
    logger.info(f"Creating Decentralized-LLM-inference Worker, {args.rank}, with world size of {args.world_size}")
    worker = LlamaWorker(model_name=model_name, head_node=head_node, args=args)
    worker.start()

