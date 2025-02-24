import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def reindex_tensor(origin_tensor, index_seq):
    origin_tensor = origin_tensor.to(index_seq.device)
    expanded_index_tensor = index_seq.unsqueeze(-1).expand(-1, -1, origin_tensor.size(-1))
    reindexed_tensor = torch.gather(origin_tensor, dim=1, index=expanded_index_tensor)
    return reindexed_tensor

def cal_restore_index(index_seq_tensor):
    index_range = torch.arange(index_seq_tensor.size(1)).repeat(index_seq_tensor.size(0), 1).to(index_seq_tensor.device)
    sorted_indices = index_seq_tensor.argsort(dim=1)
    restore_index = index_range.gather(1, sorted_indices)
    return restore_index
