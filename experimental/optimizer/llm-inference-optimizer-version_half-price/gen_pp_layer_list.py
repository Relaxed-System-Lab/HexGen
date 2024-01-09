def partition_layers(layers, stages, bias):
    
    if stages == 1:
        return [layers] 
    
    # Start with a uniform distribution
    base = layers // stages
    remainder = layers % stages
    distribution = [base] * stages
    for i in range(remainder):
        distribution[-(i+1)] += 1

    # Adjust for positive bias
    while bias > 0:
        for i in range(stages - 1):
            if distribution[i] > 1: # Ensure there's at least one layer
                distribution[i] -= 1
                distribution[i+1] += 1
                bias -= 1
                if bias <= 0:
                    break

    # Adjust for negative bias
    while bias < 0:
        for i in range(stages - 1, 0, -1):
            if distribution[i] > 1: # Ensure there's at least one layer
                distribution[i] -= 1
                distribution[i-1] += 1
                bias += 1
                if bias >= 0:
                    break

    return distribution

# layers = 80
# stages = 6
# for bias in range(-20, 20): # Testing for biases from -20 to 20
#     print(f"Bias {bias}: {partition_layers(layers, stages, bias)}")