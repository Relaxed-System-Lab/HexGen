def gen_model_id_to_service_name(num_model, num_replicas):
    # print(num_model, num_replicas)
    model_id_to_service_name = {}
    model_counter = 0
    for i in range(num_model):
        for j in range(num_replicas[i]):
                model_id_to_service_name[model_counter] = f'Model_{i}_{j}'
                model_counter += 1
    # print(model_id_to_service_name)
    return model_id_to_service_name

# print(gen_model_id_to_service_name(4, [5,1,1,1]))