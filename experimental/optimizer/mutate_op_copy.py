def mutate(individual):
    genes = copy.deepcopy(individual.genes)
    bias = copy.deepcopy(individual.bias)
    batch = copy.deepcopy(individual.batch)
    
    # Implementing Hill Climbing to guide mutation process
    max_attempts = 1
    attempts = 0
    pre_cost = cost_function(initial_array, restriction_array, genes, comp_abilities, gpu_mem_limit_list)
    while attempts < max_attempts:
        
        # Start mutation for genes
        valid_mutation = False
        validate_and_adjust(genes, initial_array)

        while not valid_mutation:
            mutation_type = random.choice(["adjust_composition", "adjust_groups"])
                
            if mutation_type == "adjust_composition" and len(genes) > 1:
                group_idx1, group_idx2 = random.sample(range(len(genes)), 2)
                group1 = genes[group_idx1]
                group2 = genes[group_idx2]
                if len(group1) > 1:
                    dim = random.randrange(len(group1))
                else:
                    dim = 0
                if group1[dim] > 0:
                    group1[dim] -= 1
                    group2[dim] += 1
                    valid_mutation = True  # This mutation type always preserves the original sum in each dimension.
                
            elif mutation_type == "adjust_groups":
                if len(genes) > 1 and random.random() < 0.5:
                    idx1, idx2 = random.sample(range(len(genes)), 2)
                    genes[idx1] = [x + y for x, y in zip(genes[idx1], genes[idx2])]
                    del genes[idx2]
                    valid_mutation = True  # Merging two groups preserves the original sum in each dimension.
                else:
                    if len(genes) > 1:
                        idx = random.randrange(len(genes))
                    else:
                        idx = 0
                    group = genes[idx]
                    new_group = [x // 2 for x in group]
                    genes[idx] = [x - x // 2 for x in group]
                    genes.insert(idx + 1, new_group)
                    valid_mutation = True 
                    # This mutation type also preserves the original sum in each dimension.
            
            validate_and_adjust(genes, initial_array)
            
            # Ensure no sub-array is all zeros
            valid_mutation &= all(any(value > 0 for value in group) for group in genes)
        
        post_cost = cost_function(initial_array, restriction_array, genes, comp_abilities, gpu_mem_limit_list)

        if post_cost < pre_cost:
            break
        attempts += 1
    
    # Finish mutation for genes
    individual.genes = genes
    
    # Start mutation for bias
    if random.random() < 0.5:
        for i in range(len(individual.bias)):
            # Determine the extent of bias change
            bias_change = random.choice([-1, 1])
            bias[i] += bias_change
    
    # Finish mutation for bias
    individual.bias = bias
    
    # # Start mutation for batch
    # if random.random() < 0.5:
    #     for i in range(len(individual.batch)):
    #         if batch[i] > 1:
    #             # Determine the extent of batch change
    #             batch_change = random.choice([-1, 1])
    #             batch[i] += batch_change
    #         else:
    #             batch[i] += 1
    
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