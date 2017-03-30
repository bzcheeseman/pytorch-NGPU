#
# Created by Aman LaChapelle on 3/28/17.
#
# pytorch-NGPU
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NGPU/LICENSE.txt
#

import torch
import torch.nn as nn
import torch.nn.functional as Funct


class CGRUdCell(nn.Module):  # As described in https://arxiv.org/abs/1610.08613
    def __init__(self, l=1, m=256, depth_first=False):
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
        if not self.depth_first:
            d_t = d_t.permute(2, 1, 0)  # go from (w, h, m) to (m, h, w) for pytorch conv
            p_t = p_t.permute(2, 1, 0)

        d_tp1 = d_t
        for i in range(self.l):  # apply CGRU cell recursively
            d_tp1 = self.step(d_tp1, p_t)

        if not self.depth_first:
            d_tp1 = d_tp1.permute(2, 1, 0)  # return it to (w, h, m)
            p_t = p_t.perute(2, 1, 0)

        return d_tp1, p_t  # p_t needs to be processed at each time step!
