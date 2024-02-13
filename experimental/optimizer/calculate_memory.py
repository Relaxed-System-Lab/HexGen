def calculate_memory(allocations, memory_cost_per_stage_list):
    gpu_memory_consumption = []

    stage_index = 0  # to index into memory_cost_per_stage_list
    gpu_index = 0

    for alloc in allocations:
        # Loop through each stage in device's allocation
        for parallel_degree in alloc:
            if stage_index < len(memory_cost_per_stage_list):
                # Get the memory cost for this specific stage
                memory_cost = memory_cost_per_stage_list[stage_index]
                
                # Calculate memory consumption per GPU in this stage
                memory_consumption = memory_cost / parallel_degree

                for _ in range(parallel_degree):
                    if gpu_index >= len(gpu_memory_consumption):
                        gpu_memory_consumption.append(0)
                    gpu_memory_consumption[gpu_index] += memory_consumption
                    gpu_index += 1

                stage_index += 1  # move to the next stage
                
            else:
                print(f"Warning: Not enough memory cost values provided for all stages. Using default value for stage {stage_index + 1}.")
                break

    return gpu_memory_consumption

'''
# Test case
# Usage
allocations = [(1, 1, 1, 1), (1, 1), (2,)]
memory_cost_per_stage_list = [0.5] * 7  # Example value

gpu_memory_consumption = calculate_memory(allocations, memory_cost_per_stage_list)
print(f"Memory Consumption per GPU: {gpu_memory_consumption}")
'''