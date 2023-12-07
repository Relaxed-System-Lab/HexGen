# Adapted from https://github.com/AFDWang/Hetu-Galvatron/blob/main/galvatron/core/pipeline/pipeline.py#L23
import torch
import torch.nn as nn
import numpy as np
from hexgen_core.utils import unwrap_model, chunk_batch
from typing import Union, Optional, Tuple, List
import operator
import copy
from torch import Tensor
import sys
sys.path.insert(0, '..')
from hexgen_core import wrap_modules_data_parallel 
import torch.distributed as dist
from typing import Any
from hexgen_core.utils import *

Shape = Union[List[int], torch.Size]


class PipelineParallel(nn.Module):
    def __init__(
        self,
        model: Any,
        model_ranks: Any,
        layer_output_tensor_shapes: Any,
        layer_output_tensor_dtypes: Any = None,
        layer_dp_sizes: Any = None,
        chunks: Any = 1,
        p2p_lists: Any = None,
        process_group: Any = None,
        broadcast_group: Any = None,
        broadcast_group_list: Any = None,
        nproc_per_node: Any = None,
        inference_params: Any = None,
        position_ids: Any = None,
        info: Any = False,
        show_process: Any = False):
        super().__init__()
        self.rank = dist.get_rank()
        self.send_list, self.recv_list, self.send_empty_list, self.recv_empty_list = p2p_lists[0], p2p_lists[1], p2p_lists[2], p2p_lists[3]
        self.broadcast_group = broadcast_group
        self.broadcast_group_list = broadcast_group_list
        self.total_model_len = len(model)
        self.show_process = show_process
        assert(len(model) == len(model_ranks))
        assert(len(model) == len(layer_output_tensor_shapes))
        layer_output_tensor_dtypes = self.get_default_tensor_dtype(layer_output_tensor_shapes) if layer_output_tensor_dtypes is None else layer_output_tensor_dtypes
        self.check_tensor_dtype(layer_output_tensor_shapes, layer_output_tensor_dtypes)

        if layer_dp_sizes is None:
            layer_dp_sizes = [1] * len(model)
        assert(len(model) == len(layer_dp_sizes))
        self.world_size = torch.distributed.get_world_size()
        self.global_rank = torch.distributed.get_rank()
        self.device_count = nproc_per_node if nproc_per_node is not None and nproc_per_node <= torch.cuda.device_count() else torch.cuda.device_count()
        self.local_rank = self.global_rank % self.device_count
        
        self.pp_global_ranks = [i for i in range(self.world_size)] if process_group is None else sorted(list(set(list(process_group))))
        assert(self.global_rank in self.pp_global_ranks)
        self.group = torch.distributed.new_group(process_group)
        self.group_size = torch.distributed.get_world_size(self.group)
        self.group_rank = torch.distributed.get_rank(self.group)

        self.stage_start_idx, cnt = model_ranks.index(self.group_rank), model_ranks.count(self.group_rank)
        self.stage_end_idx = self.stage_start_idx + cnt
        self.model_cur_stage = model[self.stage_start_idx:self.stage_end_idx]
        self.chunks = int(chunks)
        assert(self.chunks >= 1)
        self.stage_input_tensor_shape = [None] if self.is_pipeline_first_stage() else layer_output_tensor_shapes[self.stage_start_idx - 1]
        self.stage_output_tensor_shape = [None] if self.is_pipeline_last_stage() else layer_output_tensor_shapes[self.stage_end_idx - 1]
        self.stage_input_tensor_dtype = [None] if self.is_pipeline_first_stage() else layer_output_tensor_dtypes[self.stage_start_idx - 1]
        self.stage_output_tensor_dtype = [None] if self.is_pipeline_last_stage() else layer_output_tensor_dtypes[self.stage_end_idx - 1]
        self.dp_size_prev_stage = None if self.is_pipeline_first_stage() else layer_dp_sizes[self.stage_start_idx - 1]
        self.dp_size_cur_stage = None if self.is_pipeline_last_stage() else layer_dp_sizes[self.stage_end_idx - 1]

        self.dp_size_input = layer_dp_sizes[0]
        self.info = info
        self.chunk_warning = True
        
        self.checkpoint_flags_stage = None

    def check_tensor_dtype(self, layer_output_tensor_shapes, layer_output_tensor_dtypes):
        assert(len(layer_output_tensor_shapes) == len(layer_output_tensor_dtypes))
        for i in range(len(layer_output_tensor_shapes)):
            if layer_output_tensor_shapes[i] is not None:
                assert(len(layer_output_tensor_shapes[i]) == len(layer_output_tensor_dtypes[i]))

    def get_default_tensor_dtype(self, layer_output_tensor_shapes):
        layer_output_tensor_dtypes = []
        for tensor_shape in layer_output_tensor_shapes:
            if tensor_shape is None:
                layer_output_tensor_dtypes.append(None)
            else:
                layer_output_tensor_dtypes.append([torch.float] * len(tensor_shape))
        return layer_output_tensor_dtypes

    def wrap_pipeline_modules_data_parallel(self, dp_types, dp_groups, module_types, mixed_precision=torch.bfloat16, wrap_block_name=None):
        assert(self.total_model_len == len(dp_types))
        assert(self.total_model_len == len(dp_groups))
        assert(self.total_model_len == len(module_types))
        dp_types_cur_stage = dp_types[self.stage_start_idx:self.stage_end_idx]
        module_types_cur_stage = module_types[self.stage_start_idx:self.stage_end_idx]
        dp_groups_cur_stage = dp_groups[self.stage_start_idx:self.stage_end_idx]
        pp_devices_cur_stage = [self.local_rank]*(self.stage_end_idx-self.stage_start_idx)
        default_process_group = dp_groups[0]
        self.model_cur_stage = wrap_modules_data_parallel(self.model_cur_stage, dp_types_cur_stage, dp_groups_cur_stage, module_types=module_types_cur_stage, pp_devices=pp_devices_cur_stage, mixed_precision=mixed_precision, default_process_group=default_process_group, wrap_block_name=wrap_block_name)

    def update_tensor_shape(self, microbatches, dp_size_input, dp_size, template_tensor_shape):
        # Update tensor_shape with correct microbatch_size
        tensor_shape, tensor_shape_last = copy.deepcopy(template_tensor_shape), copy.deepcopy(template_tensor_shape)
        microbatch_size = microbatches[0][0][0].shape[0] * dp_size_input // dp_size
        microbatch_size_last = microbatches[0][-1][0].shape[0] * dp_size_input // dp_size
        for i in range(len(tensor_shape)):
            tensor_shape[i][0] = microbatch_size
            tensor_shape_last[i][0] = microbatch_size_last
        return tensor_shape, tensor_shape_last

    def gpipe_forward(
        self,
        forward_step_func, 
        batch,
        stage_tensor_shape,
        inference_params=None,
        position_ids=None,
        forward_only=False,
        ):

        """
        Performs the forward pass of the pipeline parallelism model using the GPipe strategy.

        Args:
        forward_step_func: The function to be used for the forward step.
        batch: Input batch of data to be processed.
        stage_tensor_shape: The shape of the tensor at this stage of the pipeline.
        inference_params: Additional parameters for inference (optional).
        position_ids: Position IDs for the input batch (optional).
        forward_only: A flag indicating whether to perform only the forward pass without storing tensors for backward pass.

        The method involves:
        1. Chunking the input batch into microbatches.
        2. Computing and updating tensor shapes for each microbatch.
        3. Iteratively processing each microbatch through the model.
        4. Handling tensor communication between stages of the pipeline.
        5. Optionally storing input and output tensors for backward pass.

        Returns:
            The output tensor from the final microbatch of the forward pass.
        """

        model = self.model_cur_stage

        # Chunk input batch into microbatches
        microbatches = [chunk_batch(batch[0], self.chunks), chunk_batch(batch[1], self.chunks)]
        self.real_chunks = len(microbatches[0])
        if self.chunks != self.real_chunks and self.chunk_warning:
            if self.global_rank == 0:
                print('\nWarning from PipelineParallel Module: Real chunks is %d !'%self.real_chunks, 'Microbatch sizes is', [m[0].shape[0] for m in microbatches[0]])
                print()
            self.chunk_warning = False
        self.num_microbatches = self.real_chunks

        # Compute tensor shapes for all microbatches, note that the last microbatch may have different microbatch_size, thus different shape!
        batch_size = batch[0][0].shape[0] * self.dp_size_input

        # Update stage_input_tensor_shape with correct microbatch_size
        if self.is_pipeline_first_stage():
            self.stage_input_tensor_shape = self.stage_input_tensor_shape_last = [None]
        else:
            self.stage_input_tensor_shape, self.stage_input_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_prev_stage, stage_tensor_shape)

        # Update stage_output_tensor_shape with correct microbatch_size
        if self.is_pipeline_last_stage():
            self.stage_output_tensor_shape = self.stage_output_tensor_shape_last = [None]
        else:
            self.stage_output_tensor_shape, self.stage_output_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_cur_stage, stage_tensor_shape)
        

        self.input_tensors = []
        self.output_tensors = []
        losses_reduced = []
        
        if self.info:
            print('rank %d'%self.global_rank, 'start forward')

        # Run forward passes.
        for i in range(self.num_microbatches):
            recv_tensor_shapes = self.stage_input_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes = self.stage_output_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            
            input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes, dtypes=recv_tensor_dtypes)
            
            cur_microbatch = [microbatches[0][i], microbatches[1][i]]

            pre_pipeline_forward(self.num_microbatches, i, self.model_cur_stage)

            output_tensor = self.forward_step(
                forward_step_func,
                cur_microbatch,
                model,
                input_tensor,
                inference_params,
                position_ids,
                losses_reduced,
            )

            post_pipeline_forward(self.num_microbatches, i, self.model_cur_stage, self.checkpoint_flags_stage)

            self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes, dtypes=send_tensor_dtypes)

            if not forward_only:
                self.input_tensors.append(input_tensor)
                self.output_tensors.append(output_tensor)

        if self.info:
            print('rank %d'%self.global_rank, 'finish forward')
        return output_tensor

    def forward_step(self, forward_step_func, batch, model, input_tensor, inference_params, position_ids, losses_reduced, loss_stage=False):
        """Forward step for passed-in model.

        If first stage, input tensor is obtained from data_iterator, otherwise
        passed-in input_tensor is used.

        Returns output tensor."""

        unwrapped_model = unwrap_model(model)
        if not isinstance(input_tensor, (list, tuple)):
            input_tensor = [input_tensor]

        for x in input_tensor:
            if x is not None and x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                x.requires_grad = True
        
        if input_tensor[0] is None:
            output_tensor = forward_step_func(batch[0], inference_params, position_ids, model)
        else:
            # Sync input hidden states within each tensor parallel group
            if len(self.broadcast_group_list) > 1:
                for i in range(len(input_tensor)):
                    if self.show_process:
                        print('Rank', self.broadcast_group_list[0], 'Start Broadcasting Tensor', i, 'with Shape:', input_tensor[i].shape, 'to Rank Group:', self.broadcast_group_list)
                    dist.broadcast(tensor=input_tensor[i], src=self.broadcast_group_list[0], group=self.broadcast_group)
            output_tensor = forward_step_func(input_tensor, inference_params, position_ids, model)
        
        if self.is_pipeline_last_stage():
            if not isinstance(output_tensor, (list, tuple)):
                output_tensor = [output_tensor]
        
        if not isinstance(output_tensor, (list, tuple)):
            return [output_tensor]
        return list(output_tensor)
    
    def get_pipeline_model_parallel_first_rank(self):
        return self.pp_global_ranks[0]


    def get_pipeline_model_parallel_last_rank(self):
        last_rank_local = self.group_size - 1
        return self.pp_global_ranks[last_rank_local]


    def get_pipeline_model_parallel_next_rank(self):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline + 1) % world_size]


    def get_pipeline_model_parallel_prev_rank(self):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline - 1) % world_size]

    def is_pipeline_first_stage(self):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        return self.group_rank == 0


    def is_pipeline_last_stage(self):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        return self.group_rank == (self.group_size - 1)

    def gen_p2p_ops(
            self, 
            rank,
            ops,
            rank_recv_list=False,
            rank_send_list=False,
            tensor_recv_prev=False,
            tensor_send_next=False,
            send_next=False, 
            empty_list=False,
        ):
        if send_next:
            for i in range(len(rank_send_list)):
                if empty_list[i]:
                    send_next_op = torch.distributed.P2POp(
                                        torch.distributed.isend,
                                        torch.zeros([]).cuda(),
                                        rank_send_list[i],
                                    )
                else:
                    if self.show_process:
                        print('Rank', self.rank, 'Start Sending Data with Shape:', tensor_send_next.shape, 'to Rank', rank_send_list[i])
                    send_next_op = torch.distributed.P2POp(
                                        torch.distributed.isend,
                                        tensor_send_next,
                                        rank_send_list[i],
                                    )
                ops.append(send_next_op)
        else:
            for i in range(len(rank_recv_list)):
                if empty_list[i]:
                    recv_prev_op = torch.distributed.P2POp(
                                        torch.distributed.irecv,
                                        torch.zeros([]).cuda(),
                                        rank_recv_list[i],
                                    )
                else:
                    if self.show_process:
                        print('Rank', self.rank, 'Start Recving Data with Shape:', tensor_recv_prev.shape, 'from Rank', rank_recv_list[i])
                    recv_prev_op = torch.distributed.P2POp(
                                        torch.distributed.irecv,
                                        tensor_recv_prev,
                                        rank_recv_list[i],
                                    )
                ops.append(recv_prev_op)
        return ops

    def _run_p2pops(
            self,
            tensor_send_prev: Union[torch.Tensor, None],
            tensor_send_next: Union[torch.Tensor, None],
            tensor_recv_prev: Union[torch.Tensor, None],
            tensor_recv_next: Union[torch.Tensor, None],
    ):
        ops = []
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_send_prev,
                self.get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            ops = self.gen_p2p_ops(
                        rank=self.rank, 
                        ops=ops, 
                        rank_recv_list=self.recv_list[self.rank], 
                        tensor_recv_prev=tensor_recv_prev, 
                        send_next=False, 
                        empty_list=self.recv_empty_list[self.rank],
                  )
        if tensor_send_next is not None:
            ops = self.gen_p2p_ops(
                        rank=self.rank, 
                        ops=ops, 
                        rank_send_list=self.send_list[self.rank], 
                        tensor_send_next=tensor_send_next, 
                        send_next=True, 
                        empty_list=self.send_empty_list[self.rank],
                  )
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_next,
                self.get_pipeline_model_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    def _communicate(
        self,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Optional[Shape] = None,
        override_scatter_gather_tensors_in_pipeline: bool = False,
        dtype_: Optional[torch.dtype] = None,
        *,
        scatter_gather_tensors_in_pipeline: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        fp32_residual_connection: bool = False,
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Base function for communication of tensors between stages.

        dtype logic: If none of ``dtype_``, ``params_dtype``, ``fp32_residual_connection`` is specified,
        torch.float32 is used.

        See https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/arguments.py#L145-L159
        for the details of arguments of ``dtype_``, ``params_dtype``, ``fp32_residual_connection``.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if set to None).
            tensor_send_prev: tensor to send to prev rank (no tensor sent if set to None).
            recv_prev: boolean for whether tensor should be received from previous rank.
            recv_next: boolean for whether tensor should be received from next rank.
            tensor_shape: optional, use when the input sequence contains less tokens than the default sequence length
            override_scatter_gather_tensors_in_pipeline:
                optional, this is used when tensor_shape is provided to override scatter gather tensors
            dtype_: This is used when tensor_shape is provided and what is the type of tensor_shape

        Keyword args:
            scatter_gather_tensors_in_pipeline: Optional. If :obj:`True`, use scatter/gather to optimize communication of tensors.
            params_dtype: Optional and legacy. Defaults to torch.float. If you manually call `.half()` or `.bfloat16()` on
                your model deliberately, pass this argument.
            fp32_residual_connection: Optional. If :obj:`True`, move residual connections to fp32.

        Returns:
            tuple containing

            - tensor_recv_prev: `torch.Tensor` if `recv_prev` is :obj:`True`, `None` otherwise.
            - tensor_recv_next: `torch.Tensor` if `recv_next` is :obj:`True`, `None` otherwise.
        """
        # Create placeholder tensors for receive in forward and backward directions if needed.
        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_shape is None:
            # In megatron, `tensor_shape` is set to `(args.seq_length, args.micro_batch_size, args.hidden_size)`
            raise RuntimeError(
                "`tensor_shape` must be specified. Common `tensor_shape` is `(seq_length, micro_batch_size, hidden_size)`")
        if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
            tensor_chunk_shape = (reduce(operator.mul, tensor_shape, 1) // parallel_state.get_tensor_model_parallel_world_size(),)
        else:
            tensor_chunk_shape = tensor_shape

        # The dtype logic below is copied from NVIDIA/Megatron-LM repo:
        # https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/p2p_communication.py#L74-L81
        # NOTE (mkozuki): Currently NeMo is implementing APEX AMP O2 style using PyTorch. In O2 style, forcing p2p comm to
        # use FP32 will be a perf killer so that I decided to reanimate `dtype_` argument with the default value of `None`.
        # NOTE (mkozuki): In PyTorch AMP, i.e. `torch.cuda.amp.autocast` context, activation tensors can be either FP32,
        # FP16, or BF16 and there's no way to tell the dtypes of tensors on different devices in general.
        # It might be possible if we restrict model architecture.
        dtype = params_dtype or torch.float
        if fp32_residual_connection:
            dtype = torch.float
        requires_grad = True
        if dtype_ is not None:
            dtype = dtype_
            requires_grad = False

        if recv_prev:
            tensor_recv_prev = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        if recv_next:
            tensor_recv_next = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )

        # Split tensor into smaller chunks if using scatter-gather optimization.
        if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
            if tensor_send_next is not None:
                tensor_send_next = split_tensor_into_1d_equal_chunks(tensor_send_next)

            if tensor_send_prev is not None:
                tensor_send_prev = split_tensor_into_1d_equal_chunks(tensor_send_prev)

        def p2p_type(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next):
            commtype = ''
            if tensor_send_prev is not None:
                commtype += 'send_prev '
            if tensor_send_next is not None:
                commtype += 'send_next '
            if tensor_recv_prev is not None:
                commtype += 'recv_prev '
            if tensor_recv_next is not None:
                commtype += 'recv_next '
            return commtype
            
        commtype = p2p_type(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)
        if self.info:
            print('rank %d'%self.global_rank, 'start p2p', commtype)
        # Send tensors in both the forward and backward directions as appropriate.
        self._run_p2pops(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)
        # To protect against race condition when using batch_isend_irecv().
        # torch.cuda.synchronize()

        if self.info:
            print('rank %d'%self.global_rank, 'done p2p', commtype)

        # If using scatter-gather optimization, gather smaller chunks.
        if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
            if recv_prev:
                tensor_recv_prev = (
                    gather_split_1d_tensor(tensor_recv_prev)
                    .view(tensor_shape)
                    .requires_grad_()
                )

            if recv_next:
                tensor_recv_next = (
                    gather_split_1d_tensor(tensor_recv_next)
                    .view(tensor_shape)
                    .requires_grad_()
                )

        return tensor_recv_prev, tensor_recv_next


    def recv_forward(
            self,
            tensor_shape: Shape,
            override_scatter_gather_tensors_in_pipeline: bool = False,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        if self.is_pipeline_first_stage():
            return None
        input_tensor, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
            dtype_=dtype,
        )
        return input_tensor


    def recv_backward(
            self,
            tensor_shape: Shape = None,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Receive tensor from next rank in pipeline (backward receive)."""
        if self.is_pipeline_last_stage():
            return None
        _, output_tensor_grad = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_forward(
            self,
            output_tensor: torch.Tensor,
            override_scatter_gather_tensors_in_pipeline: bool = False,
            tensor_shape: Shape = None,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor to next rank in pipeline (forward send)."""
        if self.is_pipeline_last_stage():
            return
        self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )


    def send_backward(
            self,
            input_tensor_grad: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor to previous rank in pipeline (backward send)."""
        if self.is_pipeline_first_stage():
            return
        self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )


    def send_forward_recv_backward(
            self,
            output_tensor: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """Batched send and recv with next rank in pipeline."""
        if self.is_pipeline_last_stage():
            return None
        _, output_tensor_grad = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_backward_recv_forward(
            self,
            input_tensor_grad: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """Batched send and recv with previous rank in pipeline."""
        if self.is_pipeline_first_stage():
            return None
        input_tensor, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor


    def send_forward_recv_forward(
            self,
            output_tensor: torch.Tensor,
            recv_prev: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline."""
        input_tensor, _ = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor


    def send_backward_recv_backward(
            self,
            input_tensor_grad: torch.Tensor,
            recv_next: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Batched recv from next rank and send to previous rank in pipeline."""
        _, output_tensor_grad = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_forward_backward_recv_forward_backward(
            self,
            output_tensor: torch.Tensor,
            input_tensor_grad: torch.Tensor,
            recv_prev: bool,
            recv_next: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched send and recv with previous and next ranks in pipeline."""
        input_tensor, output_tensor_grad = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor, output_tensor_grad

    def recv_forward_multi(
        self,
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if dtypes is not None:
            if len(dtypes) != len(tensor_shapes):
                dtypes.append(torch.float16)
            assert(len(dtypes) == len(tensor_shapes))
        input_tensors = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                input_tensors.append(None)
            else:
                input_tensors.append(self.recv_forward(tensor_shape=tensor_shape, dtype=dtype))
        return input_tensors


    def recv_backward_multi(
        self,
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        output_tensor_grads = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                output_tensor_grads.append(None)
            else:
                output_tensor_grads.append(self.recv_backward(tensor_shape=tensor_shape, dtype=dtype))
        return output_tensor_grads


    def send_forward_multi(
        self,
        output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> None:
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        if dtypes is not None:
            if len(dtypes) != len(tensor_shapes):
                dtypes.append(torch.float16)
            assert(len(dtypes) == len(tensor_shapes))
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            output_tensor = output_tensors[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                continue
            if output_tensor is None and tensor_shape is not None:
                output_tensor = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            self.send_forward(output_tensor, tensor_shape=tensor_shape, dtype=dtype)


    def send_backward_multi(
        self,
        input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> None:
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            input_tensor_grad = input_tensor_grads[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                continue
            if input_tensor_grad is None and tensor_shape is not None:
                input_tensor_grad = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            self.send_backward(input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype)


    def send_forward_recv_backward_multi(
        self,
        output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        tensor_shapes_send = None,
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        output_tensor_grads = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            output_tensor = output_tensors[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                output_tensor_grads.append(None)
                continue
            if output_tensor is None and tensor_shape is not None:
                if tensor_shapes_send is not None:
                    output_tensor = torch.zeros(tensor_shapes_send[i],dtype=dtype).cuda(self.local_rank)
                else:
                    output_tensor = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            output_tensor_grad = self.send_forward_recv_backward(output_tensor, tensor_shape=tensor_shape, dtype=dtype)
            output_tensor_grads.append(output_tensor_grad)
        return output_tensor_grads


    def send_backward_recv_forward_multi(
        self,
        input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        tensor_shapes_send = None,
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        input_tensors = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            input_tensor_grad = input_tensor_grads[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                input_tensors.append(None)
                continue
            if input_tensor_grad is None and tensor_shape is not None:
                if tensor_shapes_send is not None:
                    input_tensor_grad = torch.zeros(tensor_shapes_send[i],dtype=dtype).cuda(self.local_rank)
                else:
                    input_tensor_grad = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            input_tensor = self.send_backward_recv_forward(input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype)
            input_tensors.append(input_tensor)
        return input_tensors




class PipeSequential(nn.Sequential):
    """
    Pipe variant of ``nn.Sequential`` which supports multiple inputs.
    """

    def forward(self, *inputs, inference_params=None, position_ids=None):
        for module in self:
            if isinstance(inputs, Tuple):  # type: ignore[arg-type]
                inputs = module(*inputs, position_ids=position_ids, inference_params=inference_params)
            else:
                # Don't expand single variables (ex: lists/Tensor)
                inputs = module(inputs, position_ids=position_ids, inference_params=inference_params)
        return inputs
