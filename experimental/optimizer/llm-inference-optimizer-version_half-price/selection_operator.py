from operator import attrgetter
from deap import base, creator, tools

# Original Array
original_array = [8, 8, 8]

# Define your Validation Function:
def validate_individual(individual):
    sums = [0] * len(original_array)  # Initialize sum array
    for sub_array in individual:
        for i in range(len(sub_array)):
            sums[i] += sub_array[i]
    return sums == original_array

# Define Custom Selection Function:
def selValidTournament(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = tools.selRandom(individuals, tournsize)
        valid_aspirants = [ind for ind in aspirants if validate_individual(ind)]
        if valid_aspirants:
            chosen.append(max(valid_aspirants, key=attrgetter('fitness')))
    return chosen

# Define Fitness and Individual:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define Individuals
ind1 = creator.Individual([[4, 4, 4], [4, 4, 4]])  # valid
ind2 = creator.Individual([[5, 3, 3], [3, 5, 5]])  # valid
ind3 = creator.Individual([[6, 5, 5], [3, 3, 3]])  # invalid
ind4 = creator.Individual([[4, 3, 3], [4, 5, 5]])  # valid

# Assign a fitness attribute to each individual (for the sake of the example)
for ind in [ind1, ind2, ind3, ind4]:
    ind.fitness = creator.FitnessMin((0,))

# Create a Population
population = [ind1, ind2, ind3, ind4]

# Use Custom Selection Function to Select Valid Individuals
selected = selValidTournament(population, 2, 3)
print("Selected:")
for ind in selected:
    print(ind)
