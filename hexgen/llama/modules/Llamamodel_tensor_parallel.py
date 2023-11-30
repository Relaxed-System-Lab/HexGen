import torch
from torch import nn
from torch import Tensor, device
from typing import Tuple
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron import get_args
from megatron.model.enums import AttnMaskType, AttnType
from megatron.model import MegatronModule
from megatron.core import mpu, tensor_parallel
import torch.nn.functional as F

class LlamaParallelMLP(MegatronModule):
    def __init__(self, init_method, 
                output_layer_init_method, 
                act_func = 'silu', 
                bias = False, 
                dropout_prob = 0.0, 
                hidden_size=None, 
                intermediate_size=None,
                tp_group = None,
                ):
        super().__init__()
        args = get_args()
        self.bias = bias

        hidden_size = args.hidden_size
        intermediate_size = int(8 * hidden_size / 3)
        intermediate_size = args.multiple_of * ((intermediate_size + args.multiple_of - 1) // args.multiple_of)

        self.w1 = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            tp_group=tp_group
        )

        self.w2 = tensor_parallel.RowParallelLinear(
            intermediate_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            tp_group=tp_group
        )

        self.w3 = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            tp_group=tp_group
        )
        
        assert act_func == 'silu'
        self.activation_func = F.silu

    def forward(self, hidden_states):
        return self.w2(self.activation_func(self.w1(hidden_states)[0]) * self.w3(hidden_states)[0])[0]
    
class LlamaMLP_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = LlamaParallelMLP(init_method, scaled_init_method, tp_group = self.tp_group)
    
    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return hidden_states
