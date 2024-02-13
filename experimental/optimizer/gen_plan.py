from itertools import product, permutations

# # Generate plan for each sub_group
# def generate_unique_combinations(num_gpus):
#     allocations = set()
#     def recursive_allocation(remaining_gpus, current_allocation):
#         if remaining_gpus == 0:
#             allocations.add(tuple(sorted(current_allocation)))
#             return
#         for parallel_degree in [1, 2, 4, 8]:
#             if remaining_gpus >= parallel_degree:
#                 recursive_allocation(remaining_gpus - parallel_degree, current_allocation + [parallel_degree])
    
#     recursive_allocation(num_gpus, [])
#     unique_allocations = set()
    
#     for alloc in allocations:
#         for perm in permutations(alloc):
#             unique_allocations.add(tuple(perm))
            
#     return list(unique_allocations)

# Cut useless search space
def generate_unique_combinations(num_gpus):
    allocations = set()  # Changed to set to handle uniqueness
    
    def recursive_allocation(remaining_gpus, current_allocation):
        if remaining_gpus == 0:
            allocations.add(tuple(sorted(current_allocation)))  # Sort and add as tuple
            return
        for parallel_degree in [1, 2, 3, 4, 6, 8]:
            # if remaining_gpus >= parallel_degree:
            # Check if adding more to the current allocation would exceed the dimension limit, current limit: 8
            if remaining_gpus >= parallel_degree and len(current_allocation) < 8:
                recursive_allocation(remaining_gpus - parallel_degree, current_allocation + [parallel_degree])
    
    recursive_allocation(num_gpus, [])
    return list(allocations)  # Convert set to list before returning

# # Test case
# # Define the array of num_gpus for each device
# # num_gpus_array = [4, 4]
# # num_gpus_array = [16, 20, 8, 2]
# # num_gpus_array = [16, 32, 8, 2]
# num_gpus_array = [32, 0, 0]
# all_unique_combinations = [generate_unique_combinations(num_gpus) for num_gpus in num_gpus_array]

# final_combinations = []
# for combination in product(*all_unique_combinations):
#     final_combinations.append(list(combination))

# # Print the final combined unique allocations for all devices

# for alloc in final_combinations:
#     print(alloc)
