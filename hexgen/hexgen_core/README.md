## Features of HexGen
HexGen stands out for its exceptional handling of large-scale transformer models, offering flexibility and efficiency through its innovative features.

### Tensor Model Parallelism with Integrated Heterogeneous Communication
HexGen implements asymmetric tensor model parallelism in a heterogeneous environment, efficiently grouping GPUs for optimized computation. It incorporates the generation of heterogeneous communication groups, allowing for coordinated peer-to-peer communication and data management. A leader GPU node is selected within each tensor parallelism group to manage the broadcast operation of activations, ensuring efficient data distribution and reducing computational overhead.

### Pipeline Parallelism with Enhanced Communication Dynamics
The system aligns pipeline stages with the corresponding tensor parallelism groups, facilitating concurrent processing and improved throughput. In each pipeline stage, a leader GPU node is chosen to handle peer-to-peer communication between stages, streamlining the data transfer process. This integration of pipeline parallelism with advanced communication dynamics ensures smooth and efficient processing across different stages of the model.

### Fast Decoding Using Flash Attention
Incorporating Flash Attention, HexGen significantly enhances its decoding capabilities. This integration brings state-of-the-art efficiency to attention mechanism computations within transformer models, leading to faster and more effective processing.

### Core File Overview

- HexGen integrates the `models`, `modules`, and `generation.py` scripts from [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn) to utilize FlashAttention-2, enhancing the efficiency of attention mechanism computations.

- `gen_hetero_groups.py`, `gen_comm_groups.py`, and `gen_parallel_groups.py` files are crucial for automatically configuring various tensor model parallel and pipeline parallel groups. This setup paves the way for advanced asymmetric hybrid parallelism strategies.

- `gen_p2p_lists.py` and `heterogeneous_pipeline.py` scripts are designed to establish and manage the peer-to-peer communication (pipeline parallel communication) essential for asymmetric pipeline configurations.

