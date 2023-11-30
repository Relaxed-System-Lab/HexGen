# hetero_config = [2,2,1]
# process_groups = [CommGroup([0,1]), CommGroup([2,3]), CommGroup([4])]
# process_groups_whole_model = [process_groups[rank//2]] * 6
# pp_ranks_whole_model = [0] + [0,1,2] + [2,2]
# pp_groups = [[0,2,4],[1,3]]
# tp_groups = [dist.new_group([0,1]), dist.new_group([2,3]), dist.new_group([4])]

from hexgen_core import CommGroup
import torch.distributed as dist

def generate_index_mapping(original_lists):
    index_mapping = {}
    for index, group in enumerate(original_lists):
        for rank in group:
            index_mapping[rank] = index
    return index_mapping

def get_group_for_rank(rank, index_mapping):
    return index_mapping.get(rank)

def get_pp_groups(tp_groups):
    n = len(tp_groups)
    max_len = max(len(tp_group) for tp_group in tp_groups)
    pp_groups = []
    for i in range(max_len):
        pp_group = []
        for tp_group in tp_groups:
            if i < len(tp_group):
                pp_group.append(tp_group[i])
            else:
                pp_group.append(tp_group[-1])
        pp_groups.append(pp_group)
    return pp_groups

def gen_tp_rank_groups(hetero_config):
    tp_rank_groups = []
    current_tp_rank = 0
    for tp_rank in hetero_config:
        new_tp_group = []
        for _ in range(tp_rank):
            new_tp_group.append(current_tp_rank)
            current_tp_rank += 1
        tp_rank_groups.append(new_tp_group)
    return tp_rank_groups

def gen_tp_pp_rank_whole_model(stage_num, pp_partition, tp_rank_groups):
    pp_ranks_whole_model = [0]
    tp_ranks_whole_model = [0]

    for stage in range(stage_num):
        # If there are extra layers, add one more layer to the current stage
        # layers_in_current_stage = num_layer_per_stage + (1 if stage < extra_layers else 0)
        
        layers_in_current_stage = pp_partition[stage]

        pp_ranks_whole_model.extend([stage] * layers_in_current_stage)
        tp_ranks_whole_model.extend([len(tp_rank_groups[stage])] * layers_in_current_stage)

    pp_ranks_whole_model.extend([stage_num-1] * 2)
    tp_ranks_whole_model.extend([0] * 2)

    return pp_ranks_whole_model, tp_ranks_whole_model

def gen_hetero_groups(hetero_config, pp_partition, layer_num):
    """
    
    Arguments:
        Assume there are 8 GPUs in total and want to run llama-70b-chat-hf
        Assume we are applying 3 stages pipeline and they have tensor parallel degree equals to 4, 2, 2 respectively
        Then:
            hetero_config: [4, 2, 2]
            pp_partition: [40, 20, 20]
            layer_num: 80
    
    Return:
        Apply the same assupmtion as above
        Then each element of the return dict are explained as follows:
            tp_groups: torch.dist.new_group wrapped version of tp_rank_groups 
            tp_rank_groups: [[0, 1, 2, 3], [4, 5], [6, 7]]
            pp_rank_groups: [[0, 4, 6], [1, 5, 7], [2, 5, 7], [3, 5, 7]]
            pp_ranks_whole_model: List, length equals to 1 + 80 + 1 + 1 = 83, which consists of embedding layer + transformer layers + pre_norm + cls
                                  assume 0 * 41 means 41 zero in total, then this variable will be [0 * 41, 1 * 20, 2 * 22]
            process_groups_whole_model: [CommGroup(current tp_group)] * (1 + 80 + 2)
            tp_ranks_whole_model: assume 4 * 40 means 40 four in total, then this variable will be [0, 4 * 40, 2 * 20, 2 * 20, 0, 0]
            tp_rank_mapping: Index all the tp groups, from 0 to num_stages, return which index of this rank belongs to. 
                             In this case, rank 0:4 returns 0, rank 4:6 returns 1, rank 6:8 returns 2
            pp_rank_mapping: Index all the pp groups, from 0 to num_stages, return which index of this rank belongs to. 
                             In this case, rank 0, 4, 6 returns 0, rank 1 returns 1, rank 5, 7 returns 3, rank 2 returns 2, rank 3 returns 3
    """
    rank = dist.get_rank()
    stage_num = len(hetero_config)
    
    # Form tp_rank_groups
    tp_rank_groups = gen_tp_rank_groups(hetero_config)

    # Form pp_rank_groups
    pp_rank_groups = get_pp_groups(tp_rank_groups)

    # Form process_groups and tp_groups
    process_groups = []
    tp_groups = []
    for tp_group in tp_rank_groups:
        process_groups.append(CommGroup(tp_group))
        tp_groups.append(dist.new_group(tp_group))
    
    # Form process_groups_whole_model and pp_ranks_whole_model
    tp_index_mapping = generate_index_mapping(tp_rank_groups)
    pp_index_mapping = generate_index_mapping(pp_rank_groups)
    tp_rank_mapping = get_group_for_rank(rank, tp_index_mapping)
    pp_rank_mapping = get_group_for_rank(rank, pp_index_mapping)
    # 1 + layer_num + 2: embedding + layer + prenorm + cls
    process_groups_whole_model = [process_groups[tp_rank_mapping]] * (1 + layer_num + 2)

    # Form pp_ranks_whole_model and tp_ranks_whole_model
    pp_ranks_whole_model, tp_ranks_whole_model = gen_tp_pp_rank_whole_model(stage_num, pp_partition, tp_rank_groups)

    hetero_groups = {
        'tp_groups': tp_groups,
        'tp_rank_groups': tp_rank_groups,
        'pp_rank_groups': pp_rank_groups,
        'pp_ranks_whole_model': pp_ranks_whole_model,
        'process_groups_whole_model': process_groups_whole_model,
        'tp_ranks_whole_model': tp_ranks_whole_model,
        'tp_rank_mapping': tp_rank_mapping, 
        'pp_rank_mapping': pp_rank_mapping,
        'current_tp_group':  tp_groups[tp_rank_mapping],
        'current_pp_group':  pp_rank_groups[pp_rank_mapping],
        'current_tp_rank_group': tp_rank_groups[tp_rank_mapping],
    }

    return hetero_groups
