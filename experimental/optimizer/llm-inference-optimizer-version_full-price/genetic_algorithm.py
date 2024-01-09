import copy
import numpy as np
from deap import base, creator, tools, algorithms
from operator import attrgetter
import random
from itertools import product
from gen_plan import generate_unique_combinations
from predict_cost import predict_cost
# from calculate_memory import calculate_memory
from validate_and_adjust import validate_and_adjust
from cost_model_impl import compute_costs, tp_communication_costs, inter_device_communication_cost
from simulator_interface import PlacementSimulator
from constrained_partition import constrained_partition
from simulator_v2 import Simulator
from cost_function_for_mutation import cost_function
from intra_region_init_group import intra_region_init_group

# Add random seed
random.seed(123)
np.random.seed(123)

# Define the problem as an optimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", object, fitness=creator.FitnessMin)

# The init GPU Config: assume we have 3 types of GPU, 64 GPUs of type 0, 32 GPUs of type 1 and 32 GPUs of type2
initial_array = [[8, 8, 8, 4], [8, 8], [3, 3], [8]]
# initial_array = [[8, 8, 8, 4]] 

# The certain number of GPU of each device, Device of type 0 has 8 GPUs, Device of type 1 and 2 have 4 GPUs
restriction_array = [[8, 8, 8, 4], [8, 8], [3, 3], [8]] 
# restriction_array = [[8, 8, 8, 4]]

# GPU memory limit of each type (in GiB)
gpu_mem_limit_list = [[48, 48, 24, 48], [24, 24], [24, 24], [24]]
# gpu_mem_limit_list = [[48, 48, 24, 48]]

# Different computation and communication ability of different GPU types
comp_abilities = [[1, 1, 0.75, 1], [0.75, 0.75], [0.75, 0.75], [0.75]]
# comp_abilities = [[1, 1, 0.75, 1]]

comm_abilities = [[1, 1, 1, 1], [1, 1], [1, 1], [1]]
# comm_abilities = [[1, 1, 1, 1]]

def compare_arrays_and_calculate_cost(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    # This will create a boolean array where True represents where array1 > array2
    comparison = array1 > array2
    
    # Summing the True values and multiplying by 1e9
    cost = np.sum(comparison) * 1e12
    
    return cost

def generate_gpu_memory_limits(gpu_array, gpu_limit):
    result = []
    for i, subgroup in enumerate(gpu_array):
        limit = gpu_limit[i % len(gpu_limit)]  # Get the corresponding limit, using modulo to avoid index errors
        total_gpus_in_subgroup = sum(subgroup)  # Sum of all elements in the subgroup tuple
        result.extend([limit] * total_gpus_in_subgroup)  # Extend the result list with the limit value, repeated total_gpus_in_subgroup times
    return result

def initialize_individual():
    ind = creator.Individual()
    ind.genes = [[[8,8,8,4]],[[8,8]],[[3,3]],[[8]]]
    # ind.genes = [[[8,8,8,4]]]
    ind.bias = [0] * len(ind.genes) # Example: [0] * 8
    ind.batch = [1] * len(ind.genes) # Example: [1] * 8
    ind.iter = 0
    ind.goodput_store = 0
    return ind

def evaluate(individual): 
    group_plan_list = []
    group_cost_list = []
    group_mem_list = []
    group_pp_partition_list = []
    group_batch_list = []
    set_interval_for_global_search = 10
    for sub_group_index in range(len(individual.genes)):
        individual_genes = individual.genes[sub_group_index] 
        sub_group_plan_list = []
        sub_group_cost_list = []
        sub_group_mem_list = []
        sub_group_pp_partition_list = []
        sub_batch_list = []
        if len(individual.bias) < len(individual_genes):
            # init new sub group's partition coe as 0
            individual.bias = individual.bias + [0] * (len(individual_genes) - len(individual.bias))
            # init new sub group's batch as 1
            individual.batch = individual.batch + [0] * (len(individual_genes) - len(individual.batch))
        for sub_group, bias_value, bsz in zip(individual_genes, individual.bias, individual.batch):
            all_unique_combinations = [generate_unique_combinations(num_gpus) for num_gpus in sub_group]
            final_combinations = []
            for combination in product(*all_unique_combinations):
                final_combinations.append(list(combination))
            cost_list = []
            # print(final_combinations)
            for alloc in final_combinations:
                # Each alloc is a sub_group plan: [(1, 2, 1), (1, 2, 1), (1, 2, 1)]
                # Convert alloc to parallel config list: [1, 2, 1, 1, 2, 1, 1, 2, 1]
                parallel_config = [item for sublist in alloc for item in sublist]
                # If the parallel_config is NULL, we return a huge value and continue searching
                if len(parallel_config) == 0:
                    cost_list.append([1e9, alloc])
                    continue
                # Obtain computation cost and memory cost for each stage
                stage_costs, mem_costs, pp_layer_list =  compute_costs(parallel_config, bias_value, bsz)
                # Obtain communication cost for each stage
                comm_costs = tp_communication_costs(pp_layer_list, parallel_config, bsz)
                # Obtain intermediate communication cost between devices: poor bandwidth condition
                inter_device_comm_cost = inter_device_communication_cost(bsz)
                # Predicted cost for this sub_group plan
                cost = predict_cost(alloc, comp_abilities[sub_group_index], comm_abilities[sub_group_index], stage_costs, comm_costs, inter_device_comm_cost)
                # Form memory limit for each stage: different devices have different memory limit
                gpu_limits = list(np.array(generate_gpu_memory_limits(alloc, gpu_mem_limit_list[sub_group_index])))
                # If memory is over limit, make cost overflow
                cost += compare_arrays_and_calculate_cost(mem_costs, gpu_limits)
                cost_list.append([cost, alloc, mem_costs, pp_layer_list, bsz])
            min_item = min(cost_list, key=lambda x: x[0])
            sub_group_cost = min_item[0]
            # Select the best sub_group plan
            sub_group_plan = min_item[1]
            sub_group_mem = min_item[2]
            sub_group_pp_partition = min_item[3]
            sub_batch = min_item[4]
            sub_group_cost_list.append(sub_group_cost)
            sub_group_plan_list.append([item for sublist in sub_group_plan for item in sublist])
            sub_group_mem_list.append(sub_group_mem)
            sub_group_pp_partition_list.append(sub_group_pp_partition)
            sub_batch_list.append(sub_batch)
        group_cost_list.append(sub_group_cost_list)
        group_plan_list.append(sub_group_plan_list)
        group_mem_list.append(sub_group_mem_list)
        group_pp_partition_list.append(sub_group_pp_partition_list)
        group_batch_list.append(sub_batch_list)
    # Obtain fitness
    # We init Simulator to evaluate the fitness of the given sub_group_plan_list
    summation_group_cost = sum(sum(inner_list) for inner_list in group_cost_list)
    
    if summation_group_cost > 1e9:
        # We dont want the overflow data to be evaluated by the simuator
        return 1e9, group_plan_list
    
    if individual.iter % set_interval_for_global_search == 0 and individual.iter >= 100:
        flattened_plan_list = [sublist for inner_list in group_plan_list for sublist in inner_list]
        simulator = Simulator(flattened_plan_list, individual.bias, individual.batch, slo=0.05) # In current impl, slo can be of any value.
        goodput = simulator.exec()
        print(goodput)
        individual.goodput_store = goodput
        fitness = 1 / goodput
    else:
        # Local optimal strategy search
        fitness =  1 / sum(len(inner_list) for inner_list in individual.genes)
        # flattened_cost_list = [sublist for inner_list in group_cost_list for sublist in inner_list] 
        # fitness = 1 / (sum(flattened_cost_list) / len(flattened_cost_list))
    return fitness, individual.genes # Return the average cost as fitness

def is_group_valid(group, gpu_mem_list):
    MULTIPLIER_ARRAY = gpu_mem_list
    VALID_THRESHOLD = 144
    """Check if the given group is valid based on the multiplication criterion."""
    total = sum(a*b for a, b in zip(group, MULTIPLIER_ARRAY))
    return total >= VALID_THRESHOLD

def mutate(individual):
    for sub_gruop_index in range(len(individual.genes)):
        genes = copy.deepcopy(individual.genes[sub_gruop_index])
        
        initial_array_ = initial_array[sub_gruop_index]
        restriction_array_ = restriction_array[sub_gruop_index]
        gpu_mem_limit_list_ = gpu_mem_limit_list[sub_gruop_index]
        comp_abilities_ = comp_abilities[sub_gruop_index]
        
        # Implementing Hill Climbing to guide mutation process
        max_attempts = 1
        attempts = 0
        pre_cost = cost_function(initial_array_, restriction_array_, genes, comp_abilities_, gpu_mem_limit_list_)
        while attempts < max_attempts:

            # Start mutation for genes
            valid_mutation = False
            validate_and_adjust(genes, initial_array_)

            while not valid_mutation:
                mutation_type = random.choice(["adjust_composition", "adjust_groups"]) 
                if mutation_type == "adjust_composition" and len(genes) > 1:
                    group_idx1, group_idx2 = random.sample(range(len(genes)), 2)
                    group1 = genes[group_idx1]
                    group2 = genes[group_idx2]
                    
                    # Get the dominant type (dimension) in group2
                    dominant_dim_group2 = group2.index(max(group2))
                    
                    # Check if group1 has any of that dominant type to give
                    if group1[dominant_dim_group2] > 0:
                        dim = dominant_dim_group2
                    else:
                        # If not, just pick any type from group1 that it can give
                        valid_dims = [i for i, v in enumerate(group1) if v > 0]
                        if not valid_dims:
                            continue
                        dim = random.choice(valid_dims)
                    
                    # Transfer a unit of the chosen type from group1 to group2
                    group1[dim] -= 1
                    group2[dim] += 1

                    if is_group_valid(group1, gpu_mem_limit_list_) and is_group_valid(group2, gpu_mem_limit_list_): 
                        valid_mutation = True
                    else:
                        group1[dim] += 1 
                        group2[dim] -= 1
                
                elif mutation_type == "adjust_groups":
                    if len(genes) > 1 and random.random() < 0.001:
                        idx1, idx2 = random.sample(range(len(genes)), 2)
                        genes[idx1] = [x + y for x, y in zip(genes[idx1], genes[idx2])]
                        del genes[idx2]
                        valid_mutation = True  # Merging two groups preserves the original sum in each dimension.
                    else:
                        if len(genes) > 1:
                            idx = random.randrange(len(genes))
                        else:
                            idx = 0
                            valid_mutation = True 
                        group = genes[idx]
                        new_group = [x // 2 for x in group]
                        if is_group_valid(new_group, gpu_mem_limit_list_):
                            genes[idx] = [x - x // 2 for x in group]
                            genes.insert(idx + 1, new_group)
                            valid_mutation = True
                
                validate_and_adjust(genes, initial_array_)
                
                # Ensure no sub-array is all zeros
                valid_mutation &= all(any(value > 0 for value in group) for group in genes)
            
            post_cost = cost_function(initial_array_, restriction_array_, genes, comp_abilities_, gpu_mem_limit_list_)

            if post_cost < pre_cost:
                break
            attempts += 1
        
        # Finish mutation for genes
        individual.genes[sub_gruop_index] = genes
    
    bias = copy.deepcopy(individual.bias)
    batch = copy.deepcopy(individual.batch) 
    
    # Start mutation for bias
    if random.random() < 0.5:
        for i in range(len(individual.bias)):
            # Determine the extent of bias change
            bias_change = random.choice([-1, 1])
            bias[i] += bias_change
    
    # Finish mutation for bias
    individual.bias = bias
    
    # Finish mutation for batch
    individual.batch = batch
     
    # Append or pop element of bias with respect to genes's number of groups
    while len(individual.bias) < len(individual.genes):
        individual.bias.append(0) # init new group's pp partition as unifrom partition
    while len(individual.bias) > len(individual.genes):
        individual.bias.pop()
    
    # Append or pop element of batch with respect to genes's number of groups
    while len(individual.batch) < len(individual.genes):
        individual.batch.append(1) # init new group's batch as 1
    while len(individual.batch) > len(individual.genes):
        individual.batch.pop()
        
    return individual,



# Define your Validation Function:
def validate_individual(individual):
    genes = individual.genes
    for sub_group_index in range(len(genes)):
        initial_array_ = initial_array[sub_group_index]
        genes_ = genes[sub_group_index]
        sums = [0] * len(initial_array_)  # Initialize sum array
        for sub_array in genes_:
            for i in range(len(sub_array)):
                sums[i] += sub_array[i]

        # Check if each subarray has at least one non-zero element
        for sub_array in genes_:
            if all(element == 0 for element in sub_array):
                return False  # return False if any subarray is all zeros
    return sums == initial_array

# Define Custom Selection Function:
def selValidTournament(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = tools.selRandom(individuals, tournsize)
        valid_aspirants = [ind for ind in aspirants if validate_individual(ind)]
        if valid_aspirants:
            chosen.append(max(valid_aspirants, key=attrgetter('fitness')))
        else:
            chosen.append(tools.selRandom(individuals, 1)[0])  # choose randomly if no valid aspirants
    return chosen

toolbox = base.Toolbox()
# toolbox.register("individual", tools.initIterate, creator.Individual, initialize_individual)
toolbox.register("individual", initialize_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", selValidTournament, k=10, tournsize=3)

# Remaining parts of your genetic algorithm implementation would be same
# Including initializing the population and running the genetic algorithm

# For Example:
population = toolbox.population(n=100)
ngen = 300  # Number of generations
cxpb = 0.0  # Probability of mating two individuals
mutpb = 1.0  # Probability of mutating an individual

# Setup statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) + ['plan']

for gen in range(ngen):
    offspring = algorithms.varOr(population, toolbox, lambda_=10, cxpb=cxpb, mutpb=mutpb)
    eval_data = list(map(toolbox.evaluate, offspring))
    for ind, data in zip(offspring, eval_data):
        fit, plan = data
        ind.fitness.values = fit,
        ind.plan = [fit, plan]  # Assigning plan to the individual.
        ind.iter = gen + 1
    population = toolbox.select(offspring, k=3)
    # Record the plan of the selected individuals in the logbook:
    plans = [ind.plan for ind in population]
    min_pair = min(plans, key=lambda pair: pair[0])
    plan = min_pair[1]
    record = stats.compile(population) if stats else {}
    record['plan'] = plan  # Adding plans to the record.
    logbook.record(gen=gen, nevals=len(offspring), **record)
    print(logbook.stream)
    
# Extracting all the data after all generations are complete
gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
