import numpy as np

alpa=1e1
beta=1e1
gamma=1e1

def generate_communication_matrix(region_gpus, device_gpus):
    """
    Generates the communication matrix based on the given setup of GPUs.
    
    Parameters:
        region_gpus (list): Number of GPUs in each region.
        device_gpus (list): Number of GPUs per device in each region.
    
    Returns:
        numpy.ndarray: The communication matrix.
    """
    
    # Validate input
    if len(region_gpus) != len(device_gpus):
        raise ValueError("Mismatch between region_gpus and device_gpus lengths.")
    
    # Calculate the total number of GPUs
    total_gpus = sum(region_gpus)
    
    # Initialize the matrix with the highest communication cost
    comm_matrix = np.ones((total_gpus, total_gpus)) * 1e1

    # Set communication cost for GPUs in the same device and the same region
    start_idx = 0
    for region_idx, region_gpu_count in enumerate(region_gpus):
        device_gpu_count = device_gpus[region_idx]
        end_idx = start_idx + region_gpu_count
        
        # Loop through the GPUs in the region
        for i in range(start_idx, end_idx, device_gpu_count):
            # Set cost for GPUs in the same device
            comm_matrix[i:i+device_gpu_count, i:i+device_gpu_count] = 1

            # Set cost for GPUs in the same region but different devices
            for j in range(start_idx, end_idx):
                if i // device_gpu_count != j // device_gpu_count:
                    comm_matrix[i][j] = 8e2
                    comm_matrix[j][i] = 8e2
        
        # Move the start_idx to the next region's start
        start_idx = end_idx
    return comm_matrix

def compute_grouping_comm_cost(groupings, device_gpus, comm_matrix):
    cost_inter_region = 0
    cost_inter_device = 0
    for group in groupings:
        # Count the number of regions represented in each group
        regions_present = sum(1 for gpu_count in group if gpu_count > 0)
        # Increment the cost by the number of inter-region communications
        # If 3 regions are present, then 3 pairs of regions communicate: (1,2), (2,3), (1,3)
        cost_inter_region += regions_present * (regions_present - 1) // 2 - 1
        for i in range(len(group)):
            if group[i] > device_gpus[i]:
                cost_inter_device += comm_matrix[i][device_gpus[i]+1]
    cost_inter_region *= comm_matrix[0][-1]
    cost = cost_inter_region + cost_inter_device
    return cost

# def compute_grouping_comm_cost(groupings, region_gpus, comm_matrix):
#     """
#     Computes the communication cost for the given groupings.
    
#     Parameters:
#         groupings (list): List of groups where each group is a list representing the number of GPUs from each region.
#         comm_matrix (numpy.ndarray): The communication matrix.
    
#     Returns:
#         float: The total communication cost for the given groupings.
#     """

#     total_comm_cost = 0
    
#     # Get the indices where each region starts in the comm_matrix
#     region_starts = [0] + [sum(region_gpus[:i+1]) for i in range(len(region_gpus)-1)]

#     for group in groupings:
#         for r1_idx, r1_gpus in enumerate(group):
#             for r2_idx, r2_gpus in enumerate(group):
#                 if r1_idx != r2_idx:  # Only consider GPUs from different regions
#                     start_r1 = region_starts[r1_idx]
#                     start_r2 = region_starts[r2_idx]
#                     # For every GPU in r1, account for its communication with every GPU in r2
#                     for i in range(r1_gpus):
#                         for j in range(r2_gpus):
#                             total_comm_cost += comm_matrix[start_r1 + i][start_r2 + j]

#     return total_comm_cost

def computation_balance(groupings, comp_abilities):
    group_abilities = []
    for group in groupings:
        # Compute the total computation ability for each group
        total_ability = sum(gpu_count * comp_ability for gpu_count, comp_ability in zip(group, comp_abilities))
        group_abilities.append(total_ability)
    # The imbalance is the difference between the max and min computational ability among the groups
    imbalance = max(group_abilities) - min(group_abilities)
    return imbalance

def memory_balance(groupings, mem_abilities):
    group_abilities = []
    for group in groupings:
        # Compute the total computation ability for each group
        total_ability = sum(gpu_count * mem_ability for gpu_count, mem_ability in zip(group, mem_abilities))
        group_abilities.append(total_ability)
    # The imbalance is the difference between the max and min computational ability among the groups
    imbalance = max(group_abilities) - min(group_abilities)
    return imbalance

def cost_function(region_gpus, device_gpus, groupings, comp_abilities, mem_abilities):
    comm_matrix = generate_communication_matrix(region_gpus, device_gpus) 
    comm_overhead = compute_grouping_comm_cost(groupings, device_gpus, comm_matrix)
    comp_balance = computation_balance(groupings, comp_abilities)
    mem_balance = memory_balance(groupings, mem_abilities)
    cost_function = comm_overhead * alpa + comp_balance * beta + mem_balance * gamma
    # print(comm_overhead * alpa)
    # print(comp_balance * beta)
    # print(mem_balance * gamma)
    return cost_function

# region_gpus = [32,16,16]
# device_gpus = [8,4,4]
# groupings = [[8,4,4],[8,4,4],[8,4,4],[8,4,4]]
# # groupings = [[32,16,16]]
# # groupings = [[32,0,0],[0,16,0],[0,0,16]]
# # groupings = [[4,2,2],[4,2,2],[4,2,2],[4,2,2],[4,2,2],[4,2,2],[4,2,2],[4,2,2]]
# comp_abilities = [1, 1, 1]
# mem_abilities = [40, 24, 24]
# print(cost_function(region_gpus=region_gpus, device_gpus=device_gpus, groupings=groupings, comp_abilities=comp_abilities, mem_abilities=mem_abilities))
