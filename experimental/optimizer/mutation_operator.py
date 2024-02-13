import random
import copy
from validate_and_adjust import validate_and_adjust

MULTIPLIER_ARRAY = [48, 48, 24, 48]
VALID_THRESHOLD = 144

def is_group_valid(group):
    """Check if the given group is valid based on the multiplication criterion."""
    total = sum(a*b for a, b in zip(group, MULTIPLIER_ARRAY))
    return total > VALID_THRESHOLD

def mutate(individual):
    new_individual = copy.deepcopy(individual)
    valid_mutation = False

    validate_and_adjust(new_individual, initial_array)


    while not valid_mutation:
        mutation_type = random.choice(["adjust_composition", "adjust_groups"])
        # mutation_type = random.choice(["adjust_groups"])
            
        # if mutation_type == "adjust_composition" and len(new_individual) > 1:
        #     group_idx1, group_idx2 = random.sample(range(len(new_individual)), 2)
        #     group1 = new_individual[group_idx1]
        #     group2 = new_individual[group_idx2]
        #     if len(group1) > 1:
        #         dim = random.randrange(len(group1))
        #     else:
        #         dim = 0
        #     if group1[dim] > 0:
        #         group1[dim] -= 1
        #         group2[dim] += 1
        #         if is_group_valid(group1) and is_group_valid(group2): 
        #             valid_mutation = True  # This mutation type always preserves the original sum in each dimension.
        
        if mutation_type == "adjust_composition" and len(new_individual) > 1:
            group_idx1, group_idx2 = random.sample(range(len(new_individual)), 2)
            group1 = new_individual[group_idx1]
            group2 = new_individual[group_idx2]

            # Get the dominant type (dimension) in each group
            dominant_dim_group1 = group1.index(max(group1))
            dominant_dim_group2 = group2.index(max(group2))

            # Check if dominant GPU types are same for both groups
            if dominant_dim_group1 == dominant_dim_group2:
                dim = dominant_dim_group1
            else:
                # Pick a random dimension/type from available options in group1
                valid_dims = [i for i, v in enumerate(group1) if v > 0]
                if not valid_dims:
                    continue
                dim = random.choice(valid_dims)

            # Transfer a unit of the chosen type from group1 to group2
            if group1[dim] > 0:
                group1[dim] -= 1
                group2[dim] += 1
                if is_group_valid(group1) and is_group_valid(group2):
                    valid_mutation = True
                else:
                    group1[dim] += 1
                    group2[dim] -= 1
                    
        elif mutation_type == "adjust_groups":
            if len(new_individual) > 1 and random.random() < 0.0001:
                idx1, idx2 = random.sample(range(len(new_individual)), 2)
                new_individual[idx1] = [x + y for x, y in zip(new_individual[idx1], new_individual[idx2])]
                del new_individual[idx2]
                valid_mutation = True  # Merging two groups preserves the original sum in each dimension.
            else:
                if len(new_individual) > 1:
                    idx = random.randrange(len(new_individual))
                else:
                    idx = 0
                # group = new_individual[idx]
                # new_group = [x // 2 for x in group]
                # if is_group_valid(new_group):
                #     new_individual[idx] = [x - x // 2 for x in group]
                #     new_individual.insert(idx + 1, new_group)
                #     valid_mutation = True
                group = new_individual[idx]
                
                original_group = group.copy()

                 # Randomly choose a split factor between 0.1 and 0.9
                factor = random.uniform(0.1, 0.9)

                # Split the group based on the factor
                new_group = [int(x * factor) for x in group]
                
                # Adjust the original group by subtracting the new_group's values
                for i in range(len(group)):
                    group[i] -= new_group[i]

                if is_group_valid(new_group) and is_group_valid(group):
                    new_individual.insert(idx + 1, new_group)
                    valid_mutation = True
                else:
                    # Revert the group to its original state if mutation is not valid
                    new_individual[idx] = original_group
                    

        validate_and_adjust(new_individual, initial_array)
        
        # Ensure no sub-array is all zeros
        valid_mutation &= all(any(value > 0 for value in group) for group in new_individual)

        # Ensure sub-arrays are within restriction
        valid_mutation &= all(group[i] <= restriction_array[i] for group in new_individual for i in range(len(group)))

    return new_individual,

# Example
initial_array = [8,8,8,4]
restriction_array = [8,8,8,4]
# individual = [[8,0,0],[0,8,0],[0,0,4]] 
# new_individual = mutate(individual)
# print(new_individual)

def perform_mutate_locally(initial_individual):
    current_individual = initial_individual
    for _ in range(10000):
        current_individual = mutate(current_individual)[0]  # Extracting the first item as mutate returns a tuple
        print(current_individual)
    return current_individual,

# Example usage
initial_individual = [[8,0,0,0],[0,8,0,0],[0,0,8,0],[0,0,0,4]] 
final_individual = perform_mutate_locally(initial_individual)
print(final_individual)
