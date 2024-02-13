import copy
import numpy as np
from deap import base, creator, tools, algorithms
from operator import attrgetter
import random
from itertools import product
from gen_plan import generate_unique_combinations
from predict_cost import predict_cost
from calculate_memory import calculate_memory
from validate_and_adjust import validate_and_adjust

initial_array = [8,8,8]
gpu_mem_limit_list = [0.4,0.2,0.5]
comp_abilities = [0.1, 0.2, 0.3]
comm_abilities = [0.1, 0.2, 0.3]
comm_cost = 0.1
inter_device_comm_cost = 0.1

def compare_arrays_and_calculate_cost(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    # This will create a boolean array where True represents where array1 > array2
    comparison = array1 > array2
    
    # Summing the True values and multiplying by 1e9
    cost = np.sum(comparison) * 1e9
    
    return cost

def generate_gpu_memory_limits(gpu_array, gpu_limit):
    result = []
    for i, subgroup in enumerate(gpu_array):
        limit = gpu_limit[i % len(gpu_limit)]  # Get the corresponding limit, using modulo to avoid index errors
        total_gpus_in_subgroup = sum(subgroup)  # Sum of all elements in the subgroup tuple
        result.extend([limit] * total_gpus_in_subgroup)  # Extend the result list with the limit value, repeated total_gpus_in_subgroup times
    return result

def evaluate(individual):
    sub_group_plan_list = []
    sub_group_cost_list = []
    sub_group_mem_list = []
    sub_group_gpu_limits_list = []
    for sub_group in individual:
        all_unique_combinations = [generate_unique_combinations(num_gpus) for num_gpus in sub_group]
        final_combinations = []
        for combination in product(*all_unique_combinations):
            final_combinations.append(list(combination))
        cost_list = []
        for alloc in final_combinations:
            # Each alloc is a sub_group plan: [(1, 2, 1), (1, 2, 1), (1, 2, 1)]
            stage_num = sum(len(group) for group in alloc)
            stage_costs = [0.5] * stage_num # TODO: INPUT COST LIST
            stage_mems = [0.5] * stage_num # TODO: INPUT MEM LIST
            # Predicted cost for this sub_group plan
            cost = predict_cost(alloc, comp_abilities, comm_abilities, stage_costs, comm_cost, inter_device_comm_cost)
            # Memory list for all stages
            mem = calculate_memory(alloc, stage_mems)
            gpu_limits = generate_gpu_memory_limits(alloc, gpu_mem_limit_list) # TODO: GIVE REAL GPU_MEM_LIMIT_LIST
            cost += compare_arrays_and_calculate_cost(mem, gpu_limits)
            cost_list.append([cost, alloc, mem, gpu_limits])
        min_item = min(cost_list, key=lambda x: x[0])
        sub_group_cost = min_item[0]
        # Select the best sub_group plan
        sub_group_plan = min_item[1]
        sub_group_mem = min_item[2]
        sub_group_gpu_limits = min_item[3]
        sub_group_cost_list.append(sub_group_cost)
        sub_group_plan_list.append(sub_group_plan)
        sub_group_mem_list.append(sub_group_mem)
        sub_group_gpu_limits_list.append(sub_group_gpu_limits)
    # Obtain fitness
    fitness = sum(sub_group_cost_list) / len(sub_group_cost_list) # TODO: REPLACE WITH SIMULATOR
    return sub_group_cost_list, sub_group_plan_list, sub_group_mem_list, sub_group_gpu_limits_list # Return the average cost as fitness

individual = [[6, 8, 8], [2, 0, 0]]
sub_group_cost_list, sub_group_plan_list, sub_group_mem_list, sub_group_gpu_limits_list = evaluate(individual)

for i in range(len(sub_group_plan_list)):
    print(sub_group_cost_list[i])
    print(individual[i])
    print(sub_group_plan_list[i])
    print(sub_group_mem_list[i])
    print(sub_group_gpu_limits_list[i])
    print()