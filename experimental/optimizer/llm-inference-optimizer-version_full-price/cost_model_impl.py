from cost_model import CostModel
from gen_pp_layer_list import partition_layers

# Provided Configurations
alpha = 1e-6 # comm latency
beta = 1e11 # comm bandwidth
H = 8192 # hidden size
n_layers = 80 # layer number
B_type = 2 # mem relate coe
m_d = 1e12 # comp relate coe
s_in_i = 32 # input seq
s_out_i = 128 # output seq
b_i = 1 # batch

cost_model = CostModel(alpha=alpha, beta=beta, H=H, B_type=B_type, s_in_i=s_in_i, s_out_i=s_out_i, b_i=b_i, m_d=m_d)

def compute_costs(parallel_config, bias, bsz):

    # Cache the functions in local var to avoid dot operation in each iter
    compute_no_tp_func_prompt = cost_model.compute_prompting_phase_no_tensor_parallelism
    compute_no_tp_func_token_gen = cost_model.compute_token_generation_phase_no_tensor_parallelism
    compute_memory_func = cost_model.compute_memory_usage

    # # Currently we uniformly distribute our layer among stages like 80 --> 3 stages --> [27, 27, 26]
    # pp_degree = len(parallel_config)
    # base = n_layers // pp_degree
    # remainder = n_layers % pp_degree
    # pp_layer_list = [base + (1 if i < remainder else 0) for i in range(pp_degree)]
    
    # We change unifrom distribution of layers among stages --> distribute layers according to input bias
    pp_degree = len(parallel_config)
    pp_layer_list = partition_layers(n_layers, pp_degree, bias)
    
    comp_cost_list = []
    mem_cost_list = []
    
    # Compute comp_cost_list
    for layer in pp_layer_list:
        comp_cost = compute_no_tp_func_prompt(layer, bsz) + compute_no_tp_func_token_gen(layer, bsz) * (s_out_i - s_in_i)
        comp_cost_list.append(comp_cost)
        
    # Compute mem_cost_list
    for i in range(len(parallel_config)):
        for _ in range(parallel_config[i]):
            mem_cost = compute_memory_func(pp_layer_list[i], parallel_config[i], bsz)
            mem_cost_list.append(mem_cost)
    
    return comp_cost_list, mem_cost_list, pp_layer_list

def tp_communication_costs(pp_layer_list, parallel_config, bsz):
    comm_costs = []
    for i in range(len(pp_layer_list)):
        comm_costs.append(cost_model.compute_prompting_phase_communication_with_tensor_parallelism(pp_layer_list[i], parallel_config[i], bsz) + \
            cost_model.compute_token_generation_phase_communication_with_tensor_parallelism(pp_layer_list[i], parallel_config[i], bsz) * (s_out_i - s_in_i))
    return comm_costs

def inter_device_communication_cost(bsz):
    inter_device_comm_cost = cost_model.compute_pipeline_parallelism_prompting_phase(1, bsz) + cost_model.compute_pipeline_parallelism_token_generation_phase(1, bsz) * (s_out_i - s_in_i)
    return inter_device_comm_cost
