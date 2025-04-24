import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured

if __name__ == '__main__':
    A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()