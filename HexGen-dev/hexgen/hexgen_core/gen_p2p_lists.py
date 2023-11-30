def generate_send_recv_lists(pipeline_groups, mainline):
    # initialize empty send and receive lists for each rank
    ranks = set(rank for group in pipeline_groups for rank in group)
    SendList = {rank: [] for rank in ranks}
    RecvList = {rank: [] for rank in ranks}
    SendBoolean = {rank: [] for rank in ranks}
    RecvBoolean = {rank: [] for rank in ranks}
    
    # fill up send and receive lists based on pipeline groups
    for group in pipeline_groups:
        is_mainline = set(group) == set(mainline)
        for i in range(len(group) - 1):
            # Avoid appending duplicates
            if group[i+1] not in SendList[group[i]]:
                SendList[group[i]].append(group[i+1])
                SendBoolean[group[i]].append(not is_mainline)
            if group[i] not in RecvList[group[i+1]]:
                RecvList[group[i+1]].append(group[i])
                RecvBoolean[group[i+1]].append(not is_mainline)
            
    return SendList, RecvList, SendBoolean, RecvBoolean

# pipeline_groups = [[0,2,4], [1,3,5], [1,3,6], [1,3,7]]
# send, recv, send_bool, recv_bool = generate_send_recv_lists(pipeline_groups)
# print("Send List:", send)
# print("Recv List:", recv)
# print("Send Boolean:", send_bool)
# print("Recv Boolean:", recv_bool)

