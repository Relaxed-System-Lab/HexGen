import torch.nn as nn
import torch

def param_init_fn(module):
    module.to_empty(device=torch.device("cuda"))
    for m in module.modules():
        if callable(getattr(m, 'reset_parameters', None)):
            m.reset_parameters()

def param_init_fn_(module: nn.Module):
    for submodule in module.modules():
        # Handle parameters
        for param_name, param in submodule.named_parameters(recurse=False):
            if param.is_meta:
                materialized_param = nn.Parameter(
                    torch.empty_like(param, device=torch.device("cuda"))
                )
                nn.init.uniform_(materialized_param)
                setattr(submodule, param_name, materialized_param)
        # Handle buffers
        for buffer_name, buffer in submodule.named_buffers(recurse=False):
            if buffer.is_meta:
                materialized_buffer = torch.empty_like(buffer, device=torch.device("cuda"))
                # No need to apply nn.init.uniform_ unless you specifically want to for buffers.
                setattr(submodule, buffer_name, materialized_buffer)

def wrap_modules_data_parallel(module_list, dp_types, dp_groups, module_types, pp_devices=None, mixed_precision=torch.bfloat16, default_process_group=None, wrap_block_name=None):
    assert len(module_list) == len(dp_types)
    assert len(module_list) == len(dp_groups)
    
    process_group = default_process_group if default_process_group is not None else dp_groups[0]
    pp_on = True if process_group.size < torch.distributed.get_world_size() else False
    
    if pp_devices is not None:
        assert len(pp_devices) == len(module_list)
    for i in range(len(module_list)):
        pp_device = None if pp_devices is None else pp_devices[i]
        param_init_fn_(module_list[i])
        module_list[i].process_group = process_group.group
        module_list[i] = module_list[i].to(pp_device)
    return module_list
