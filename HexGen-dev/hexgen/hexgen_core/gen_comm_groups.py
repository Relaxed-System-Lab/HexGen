import torch

class CommGroup(object):
    def __init__(self, ranks):
        assert isinstance(ranks, list) or isinstance(ranks, range), 'Rank list or range should be provided to create a CommGroup!'
        self.ranks = sorted(list(set(list(ranks))))
        self.size = len(self.ranks)
        self.group = torch.distributed.new_group(self.ranks)
    def has_rank(self, rank):
        if rank in self.ranks:
            self.intra_group_id = self.ranks.index(rank)
            return True
        return False
    def allgather(self, input):
        return gather_from_tensor_model_parallel_region_group(input, self.group)
    def print(self):
        print(self.ranks, end = ' ')
