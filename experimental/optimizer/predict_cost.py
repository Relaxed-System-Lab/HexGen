from cost_model_impl import tp_communication_costs

def predict_cost(allocations, comp_abilities, comm_abilities, stage_costs, comm_costs, inter_device_comm_cost):
    total_cost = 0
    stage_index = 0  # to index into stage_costs
    
    # Loop through each device's allocation
    for i, alloc in enumerate(allocations):
        device_comp_ability = comp_abilities[i]
        device_comm_ability = comm_abilities[i]
        # Loop through each stage in device's allocation
        for j, parallel_degree in zip(range(len(alloc)), alloc):
            if stage_index < len(stage_costs):
                # Get the computation cost for this specific stage
                stage_cost = stage_costs[stage_index]
                
                # Calculate computation cost
                comp_cost = (stage_cost / parallel_degree) * device_comp_ability

                # Add intra-device communication cost if tensor parallel degree > 1
                additional_comm_cost = comm_costs[j] * device_comm_ability if parallel_degree > 1 else 0

                total_cost += comp_cost + additional_comm_cost
                stage_index += 1  # move to the next stage
            else:
                print(f"Warning: Not enough stage cost values provided for all stages. Using the last available value for stage {stage_index + 1}.")
                # Here we are using the last available stage_cost if the list is exhausted
                comp_cost = (stage_cost / parallel_degree) * device_comp_ability
                additional_comm_cost = comm_costs[j] * device_comm_ability * (parallel_degree - 1) if parallel_degree > 1 else 0
                total_cost += comp_cost + additional_comm_cost
                
            
        # Add inter-device communication cost
        if i < len(allocations) - 1:
            total_cost += inter_device_comm_cost
            
    return total_cost


# # Test case
# # Define abilities and costs
# comp_abilities = [0.1, 0.1, 0.1, 0.1]
# comm_abilities = [0.1, 0.1, 0.1, 0.1]
# stage_costs = [0.5] * 4  # Example list of stage costs
# comm_cost = [0.1, 0.1, 0.1, 0.1]
# inter_device_comm_cost = 0.1

# alloc = [(2,1), (2,1)]
# cost = predict_cost(alloc, comp_abilities, comm_abilities, stage_costs, comm_cost, inter_device_comm_cost)
# print(f"Allocation: {alloc}, Predicted Cost: {cost}")

