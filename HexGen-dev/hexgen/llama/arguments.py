import argparse

def add_arguments(parser):
    group = parser.add_argument_group(title='hexgen arguments')

    # hetro parallelism arguments
    group.add_argument(
        "--local-rank", type=int, default=-1, help="Local rank.",
    )
    parser.add_argument(
        "--model_size", type=str, default='llama-7b', help="Model size.", choices=['llama-7b', 'llama-13b', 'llama-30b', 'llama-70b']
    )
    parser.add_argument(
        "--overwrite_config", type=int, default=0, help="Whether to overwrite model config"
    )
    group.add_argument(
        "--initialize_on_meta", type=int, default=1, help="Whether to initialize parameters on meta device.", choices=[0, 1]
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    group.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    group.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default='fp16', help="Mixed precision option.", choices=['fp32', 'fp16', 'bf16'],
    )
    parser.add_argument(
        "--hetero_config", type=int, nargs='+', default=0, help="Give and execute heterogeneous configuration",
    )
    parser.add_argument(
        "--pp_partition", type=int, nargs='+', default=0, help="Give and execute pipeline configuration",
    )

    # coordinator arguments
    parser.add_argument(
        "--model_name", type=str, default="Llama-2-7b-chat-hf", help="Assign the desired name for a worker"
    )
    parser.add_argument(
        "--head_node", type=str, default='http://198.176.96.165:8092', help="Head node of coordinator"
    )
    parser.add_argument(
        "--priority", type=int, default=0, help="To be implemented",
    )
    parser.add_argument(
        "--group_id", type=int, default=0, help="To differentiate workers on a single node",
    )
    return parser


_KV_CACHE_DICT = None

def get_kv_cache():
    global _KV_CACHE_DICT
    return _KV_CACHE_DICT

def set_kv_cache(kv_cache_dict):
    global _KV_CACHE_DICT
    _KV_CACHE_DICT = kv_cache_dict

def clear_kv_cache():
    global _KV_CACHE_DICT
    _KV_CACHE_DICT = None
