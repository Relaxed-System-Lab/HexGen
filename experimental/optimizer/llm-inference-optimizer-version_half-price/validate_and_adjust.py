def validate_and_adjust(individual, target_sums):
    for i, target_sum in enumerate(target_sums):
        current_sum = sum(sub_array[i] for sub_array in individual)
        difference = current_sum - target_sum
        
        # Check each sub-array in the individual
        while difference != 0:
            for sub_array in individual:
                if difference < 0 and current_sum < target_sum:  # Need to increase the current_sum
                    sub_array[i] += 1
                    difference += 1
                    current_sum += 1
                elif difference > 0 and current_sum > target_sum and sub_array[i] > 0:  # Need to decrease the current_sum
                    amount_to_decrease = min(sub_array[i], difference)  # Decrease the sub_array[i] but not below 0
                    sub_array[i] -= amount_to_decrease
                    difference -= amount_to_decrease
                    current_sum -= amount_to_decrease
                    
                if difference == 0:
                    break  # Exit the loop if difference has reached 0


# # Test case
# individual = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 1, 1]]
# target_sums = [8, 8, 8]
# validate_and_adjust(individual, target_sums)
# print(individual)  # This should print adjusted individual so that each dimension's sum equals to 8.
