import json
import copy

class ModelConfig:
    def __init__(self, model_name, executable_name, device_counts, start_device_id=0):
        self.model_name = model_name
        self.executable_name = executable_name
        self.device_counts = device_counts
        self.start_device_id = start_device_id

def create_json(executables, num_replicas):
    data_list = []
    # data_list_ = []
    i = 0
    for executable in executables:
            service_name_ = f"Model_{i}"
            data = {}
            data["service_name"] = service_name_
            data["model_name"] = executable.model_name
            data["executable_name"] = executable.executable_name

            mesh_group = []
            last_device_id = executable.start_device_id - 1

            for devices_count in executable.device_counts: 
                node_group = {}
                node_group["node_ids"] = [0] 
                node_group["devices"] = [[i for i in range(last_device_id+1, last_device_id+1+devices_count)]]
                last_device_id += devices_count
                mesh_group.append(node_group)

            data["mesh_group"] = mesh_group
            for r in range(num_replicas[i]):
                data_ = copy.deepcopy(data)
                data_["service_name"] = f"Model_{i}_{r}"
                data_list.append(data_)
            i += 1
    with open('placements/interface_test.json', 'w') as f:
        json.dump(data_list, f, indent=4)

# # Define the cases
# cases = [
#     ModelConfig("Bert_2.6B", "Bert_2.6B_16_device_hybrid_0", [4,4,2,6], 0),
#     ModelConfig("Bert_2.6B", "Bert_2.6B_16_device_hybrid_1", [2,2,6,6], 16),
#     ModelConfig("Bert_2.6B", "Bert_2.6B_16_device_hybrid_2", [8,1,1,6], 32),
#     ModelConfig("Bert_2.6B", "Bert_2.6B_16_device_hybrid_3", [3,5,2,6], 48),
# ]

# total_devices = 1
# model_name = "Bert_2.6B"
# device_counts_array = [[2, 2, 2], [3, 2, 1], [4]]

def generate_cases(total_devices, model_name, device_counts_array):
    start_device_id = 0
    num_configs = len(device_counts_array)
    cases = []
    for i in range(num_configs):
        index_string = ''.join(map(str, device_counts_array[i]))
        index = int(index_string)
        executable_name = f"{model_name}_{sum(device_counts_array[i])}_device_hybrid_{index}"
        cases.append(ModelConfig(model_name, executable_name, device_counts_array[i], start_device_id))
        start_device_id += sum(device_counts_array[i])
    return cases

# # Generate a JSON file for each case
# cases = generate_cases(total_devices, model_name, device_counts_array)
# create_json(cases, num_replicas=[3, 4, 2])
