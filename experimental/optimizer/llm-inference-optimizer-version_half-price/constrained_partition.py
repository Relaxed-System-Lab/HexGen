import numpy as np

# # The init GPU Config: assume we have 3 types of GPU, 64 GPUs of type 0, 32 GPUs of type 1 and 32 GPUs of type2
# initial_array = [64, 100, 32, 8]
# # The certain number of GPU of each device, Device of type 0 has 8 GPUs, Device of type 1 and 2 have 4 GPUs
# restriction_array = [16, 4, 4, 2]
# # GPU memory limit of each type (in GiB)
# gpu_mem_limit_list = [40, 24, 24, 8]
# # Different computation and communication ability of different GPU types
# comp_abilities = [1.5, 1, 0.5, 0.1]
# comm_abilities = [1.5, 1, 0.5, 0.1]
# # comm_cost = 0.1
# # inter_device_comm_cost = 0.1

# assert len(initial_array) == len(restriction_array) == len(gpu_mem_limit_list) == len(comp_abilities) == len(comm_abilities)
# assert sum(np.array(initial_array) % np.array(restriction_array)) == 0

# num_devices = [all_gpus // per_gpus for all_gpus, per_gpus in zip(initial_array, restriction_array)]    # eg: 8 devices, each with 8 gpus
# properties = np.array(comp_abilities) / np.array(comm_abilities)   # for roughly splitting
# print(num_devices)
# # for future use
# def set_ids():
#     device_ids = []
#     start_id = 0
#     for n in num_devices:
#         device_ids.append([i for i in range(start_id, start_id + n)])
#         start_id += n
#     return device_ids

# device_names = ["3090", "3080", "A100"]
# device_ids = set_ids()

"""
    This differentiation of names is for following cases:
        initial_array = [64, 32, 32, 32]
        restriction_array = [8, 4, 8, 4]
        device_names = ["3090", "3080", "3080", "A100"]
    There are two types of 3080, the machine has different number of gpus
"""

def constrained_partition(array, restriction_array):
    # num_groups = random.randint(1, len(array)) 
    # Num_groups is determine with respect to restrictions and inital_arrays
    # num_groups are as small as possible
    # num_groups = int(min(num_devices))
    
    num_devices = [all_gpus // per_gpus for all_gpus, per_gpus in zip(array, restriction_array)]  
    
    # possible imporvement
    num_groups = np.median(num_devices).astype(int)

    partitions = [[0] * len(array) for _ in range(num_groups)]  # Initialize partitions
    
    for i in range(len(array)):
        values = [0] * num_groups

        # Make sure values is divisible by restriction. In other words, a single machine won't be splitted
        for idx in range(num_groups):
            if sum(values) + restriction_array[i] <= array[i]:
                values[idx] = restriction_array[i]
        
        # Add remaining machines to first few group
        if num_devices[i] >= num_groups:
            for idx in range(num_devices[i] - num_groups):
                values[idx % num_groups] += restriction_array[i]

        # Assign the values to the i-th dimension of each partition
        for idx in range(num_groups):
            partitions[idx][i] = values[idx]
    
    return partitions

# # compare
# def uniform_partition(array):
#     # num_groups = random.randint(1, len(array)) 
#     # Num_groups is determine with respect to restrictions and inital_arrays
#     num_groups = int(max(np.array(initial_array) / np.array(restriction_array)))
#     partitions = [[0] * len(array) for _ in range(num_groups)]  # Initialize partitions
    
#     for i in range(len(array)):
#         quotient, remainder = divmod(array[i], num_groups)  # Divide each dimension value uniformly among the groups
#         values = [quotient] * num_groups  # Initialize values for each partition for the i-th dimension

#         # Allocate the remainder to the first few groups
#         for idx in range(remainder):
#             values[idx] += 1
    
#         # Assign the values to the i-th dimension of each partition
#         for idx in range(num_groups):
#             partitions[idx][i] = values[idx]

#     return partitions

# print("Modi", constrained_partition(initial_array))
# print("=" * 50)
# print("Orig", uniform_partition(initial_array))
