def intra_region_init_group(input_list):
    n = len(input_list)
    group = []

    for i in range(n):
        current_group = [0] * n
        current_group[i] = input_list[i]
        group.append(current_group)
        
    return group