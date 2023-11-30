from typing import Optional, List, Union
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp._common_utils import _FSDPState, TrainingState
from torch.distributed.fsdp.flat_param import FlatParamHandle
from torch.distributed.fsdp._runtime_utils import _post_backward_hook
import functools, weakref

def pre_pipeline_forward(num_microbatches, idx, model):
    if num_microbatches > 1 and idx == 0:
        delete_ddp_backward_hook(model)
        
def post_pipeline_forward(num_microbatches, idx, model, checkpoint_list):
    if num_microbatches > 1:
        if isinstance(model, FSDP):
            model = model._fsdp_wrapped_module
        assert(len(model)==len(checkpoint_list))
        for module, checkpoint in zip(model, checkpoint_list):
            if not checkpoint:
                if idx == num_microbatches - 1:
                    delete_fsdp_post_backward_hook(module, save_acc_grad=True, release_param=True)
                else:
                    delete_fsdp_post_backward_hook(module)
            else:
                if idx == num_microbatches - 1:
                    rewrite_fsdp_forward_no_post_backward(module)
                    
def pre_pipeline_backward(num_microbatches, idx, model, checkpoint_list):
    if num_microbatches > 1:
        if isinstance(model, FSDP):
            model = model._fsdp_wrapped_module
        assert(len(model)==len(checkpoint_list))
        if idx == num_microbatches - 1:
            register_ddp_backward_hook(model)
            for module, checkpoint in zip(model, checkpoint_list):
                if not checkpoint:
                    register_fsdp_post_backward_hook(module)
                else:
                    recover_fsdp_forward_with_post_backward(module)
    
def _register_post_backward_hooks_handle(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> None:
    if not torch.is_grad_enabled():
        return
    flat_param = handle.flat_param
    already_registered = hasattr(flat_param, "_post_backward_hook_state")
    if already_registered or not flat_param.requires_grad:
        return
    # Get the `AccumulateGrad` object
    acc_grad = handle.acc_grad  # type: ignore[union-attr]
    assert acc_grad is not None
    hook_handle = acc_grad.register_hook(
        functools.partial(_post_backward_hook, state, handle)
    )
    flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]

def delete_fsdp_post_backward_hook(model, save_acc_grad=False, release_param=True):
    for m in model.modules():
        if isinstance(m, FSDP):
            for handle in m._handles:
                flat_param = handle.flat_param
                if flat_param.requires_grad:
                    if hasattr(flat_param, "_post_backward_hook_state"):
                        if save_acc_grad:
                            handle.acc_grad = flat_param._post_backward_hook_state[0]
                        flat_param._post_backward_hook_state[1].remove()
                        delattr(flat_param, "_post_backward_hook_state") # whether to reduce-scatter and release grad
                    flat_param._post_backward_called = False
            if not release_param and m._is_root:
                m._post_backward_callback_queued = True # whether to release params, trades off an allgather between param memory

def register_fsdp_post_backward_hook(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            for handle in m._handles:
                _register_post_backward_hooks_handle(m, handle)
            if m._is_root:
                m.training_state = TrainingState.IDLE
            m._post_backward_callback_queued = False # need to wait for post backward

def delete_ddp_backward_hook(model):
    for m in model.modules():
        # For DDP module, we need to disable gradient sync for accumulation, 
        #   and set sync manually before backward of the last microbatch.
        if isinstance(m, DDP):
            m.require_backward_grad_sync = False

def register_ddp_backward_hook(model):
    for m in model.modules():
        # For DDP module, we need to disable gradient sync for accumulation, 
        #   and set sync manually before backward of the last microbatch.
        if isinstance(m, DDP):
            m.require_forward_param_sync = True
            m.reducer.prepare_for_backward([])

def forward_delete_backward_hook(original_forward, weak_self, *args, **kwargs):
    module = weak_self()
    output = original_forward(module, *args, **kwargs)
    with torch.no_grad():
        delete_fsdp_post_backward_hook(module, save_acc_grad=True, release_param=False)
    return output

def rewrite_fsdp_forward_no_post_backward(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            m.original_forward = m.forward
            m.forward = functools.partial(forward_delete_backward_hook, type(m).forward, weakref.ref(m))

def recover_fsdp_forward_with_post_backward(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            m.forward = m.original_forward

def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]

def chunk_batch(inputs, chunks):
    if inputs is None:
        return inputs

    batches = [[] for _ in range(chunks)]
    # Actual number of chunks produced
    num_chunks = -1
    for input in inputs:
        if torch.is_tensor(input):
            # Chunk only tensors.
            tensors = input.chunk(chunks)

            # Validate number of chunks equal across all inputs.
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(f'Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}')
            num_chunks = len(tensors)

            for i, tensor in enumerate(tensors):
                batches[i].append(tensor)
        else:
            # Replicate non-tensors or tensors wrapped with 'NoChunk'.
            for i in range(chunks):
                batches[i].append(input)

    # Truncate to actual number of chunks
    batches = batches[:num_chunks]

    return batches

def unwrap_model(model, module_instances=(DistributedDataParallel,)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model