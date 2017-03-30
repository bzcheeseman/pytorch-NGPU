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
    def __init__(self, I, w, h, l_cgru, l_cgrud, m):
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
        p_tp1[0, k] = self.E.mm(o_k)



class Encode(nn.Module):  # need to embed the input into the initial state
    def __init__(self, i, w, n, m=24):
        super(Encode, self).__init__()

        self.I = i
        self.w = w
        self.n = n
        self.m = m
        self.embed = nn.Parameter(torch.rand(self.I, m))

    def forward(self, x):
        # input sequence i is a sequence of n discrete symbols from {0...I} - see code example if possible
        batch = x.size()[0]
        s0 = Variable(torch.zeros(batch, self.m, self.n, self.w))
        # embed x into s0
        return s0


if __name__ == "__main__":
    cgru = CGRUCell()
    encoder = Encode(1, 3, 6)

