import random
from validate_and_adjust import validate_and_adjust
from copy import deepcopy

def custom_two_point_cx(ind1, ind2, initial_array):
    # Choose two random crossover points
    size = min(len(ind1), len(ind2))
    if size > 1:
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
    else:
        cxpoint1 = cxpoint2 = 1

    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
    # Perform crossover between the two chosen points
    offspring1 = ind1[:cxpoint1] + ind2[cxpoint1:cxpoint2] + ind1[cxpoint2:]
    offspring2 = ind2[:cxpoint1] + ind1[cxpoint1:cxpoint2] + ind2[cxpoint2:]
    
    # Validate and adjust the offsprings to meet the problem constraints
    validate_and_adjust(offspring1, initial_array)
    validate_and_adjust(offspring2, initial_array)
    
    return offspring1, offspring2


def test_custom_two_point_cx():
    # Here, using an example to demonstrate, modify accordingly
    ind1 = [[8, 8, 8]]
    ind2 = [[3, 5, 5], [5, 3, 3]]
    initial_array = [8, 8, 8]

    offspring1, offspring2 = custom_two_point_cx(deepcopy(ind1), deepcopy(ind2), initial_array)

    print("Parent 1:")
    print(ind1)
    print("Parent 2:")
    print(ind2)
    print("Offspring 1:")
    print(offspring1)
    print("Offspring 2:")
    print(offspring2)

    # Checking if the sum of corresponding dimensions are preserved
    assert all(
        sum(offspring1[i][j] for i in range(len(offspring1))) == initial_array[j]
        for j in range(len(initial_array))
    ), "Offspring 1 is invalid"
    
    assert all(
        sum(offspring2[i][j] for i in range(len(offspring2))) == initial_array[j]
        for j in range(len(initial_array))
    ), "Offspring 2 is invalid"

    print("Test passed successfully!")

# Run the test function
test_custom_two_point_cx()


