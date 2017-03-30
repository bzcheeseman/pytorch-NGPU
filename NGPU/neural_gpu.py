#
# Created by Aman LaChapelle on 3/27/17.
#
# pytorch-NGPU
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NGPU/LICENSE.txt
#

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.autograd import Variable

from Utils import CGRUCell, CGRUdCell


class NeuralGPU(nn.Module):
    def __init__(self, I, l_cgru, l_cgrud, m):
        super(NeuralGPU, self).__init__()

        self.E = nn.Embedding(I, m)
        self.cgru = CGRUCell(l_cgru, m)
        self.cgrud = CGRUdCell(l_cgrud, m)
        self.O = nn.Linear(m, I, bias=False)
        self.Ep = nn.Embedding(I, m)

    def step_cgru(self, s_t):
        s_tp1 = self.cgru(s_t)
        return s_tp1

    def step_cgrud(self, d_t, p_t, k):
        d_tp1, p_t = self.cgrud(d_t, p_t)

        o_k = self.O.mm(d_tp1[0, k])
        p_tp1 = p_t
        p_tp1[0, k] = self.Ep.mm(o_k.long())

if __name__ == "__main__":
    cgru = CGRUCell()

