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


class CGRUCell(nn.Module):  # As described in https://arxiv.org/abs/1511.08228
    def __init__(self, l=1, m=24):

        super(CGRUCell, self).__init__()

        self.l = l

        kernel = 3

        self.u_gate = nn.Sequential(
            nn.Conv2d(m, m, kernel, padding=1)

        )

        self.r_gate = nn.Sequential(
            nn.Conv2d(m, m, kernel, padding=1)
        )

        self.conv_r = nn.Sequential(
            nn.Conv2d(m, m, kernel, padding=1),
            nn.Tanh()
        )

    def step(self, s):
        u = torch.clamp(1.2 * Funct.sigmoid(self.u_gate(s)) - 0.1, min=0.0, max=1.0)
        r = torch.clamp(1.2 * Funct.sigmoid(self.r_gate(s)) - 0.1, min=0.0, max=1.0)

        output = u * s + (1.0 - u) * self.conv_r(r * s)

        return output

    def forward(self, s_t):
        s_tp1 = s_t
        for i in range(self.l):  # apply the CGRU cell recursively
            s_tp1 = self.step(s_tp1)

        return s_tp1


class CGRUdCell(nn.Module):  # As described in https://arxiv.org/abs/1610.08613
    def __init__(self, l=1, w=4, n=10, m=256):
        super(CGRUdCell, self).__init__()

        self.l = l

        kernel = 3

        self.u_s = nn.Conv2d(m, m, kernel, padding=1)
        self.u_p = nn.Conv2d(m, m, kernel, padding=1, bias=False)
        self.r_s = nn.Conv2d(m, m, kernel, padding=1)
        self.r_p = nn.Conv2d(m, m, kernel, padding=1, bias=False)
        self.conv_r = nn.Conv2d(m, m, kernel, padding=1)
        self.conv_p = nn.Conv2d(m, m, kernel, padding=1, bias=False)

    def step(self, s_t, p_t):
        u = torch.clamp(1.2 * Funct.sigmoid(self.u_s(s_t) + self.u_p(p_t)) - 0.1, min=0.0, max=1.0)
        r = torch.clamp(1.2 * Funct.sigmoid(self.r_s(s_t) + self.r_p(p_t)) - 0.1, min=0.0, max=1.0)

        output = u * s_t + (1.0 - u) * Funct.tanh(self.conv_r(r * s_t) + self.conv_p(p_t))

        return output

    def forward(self, d_t, p_t):
        d_tp1 = d_t
        for i in range(self.l):  # apply GRUcell recursively
            d_tp1 = self.step(d_tp1, p_t)

        return d_tp1


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

    # embed input into CGRU, s0 ~ (w, n, m)

    # s0 = Variable(torch.rand(5, 24, 25, 4))  # (batch, m, h, w)
    # print(s0)
    print(cgru(s0))

