new_individual = [[16, 20, 8, 2], [16, 16, 8, 2], [16, 16, 4, 2], [16, 16, 4, 2], [0, 16, 4, 0], [0, 16, 4, 0]]
initial_array = [64, 100, 32, 8]
restriction_array = initial_array 

for idx, group in enumerate(new_individual):
            # Check and split the group if it exceeds the restriction_array
            for i, (val, restriction) in enumerate(zip(group, restriction_array)):
                if val > restriction:
                    excess = val - restriction
                    group[i] = restriction
                    
                    # Create a new group with the excess value and zeros for other dimensions
                    new_group = [0] * len(group)
                    new_group[i] = excess
                    new_individual.insert(idx + 1, new_group)
                    
print(new_individual)