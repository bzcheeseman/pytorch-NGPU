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


class CGRUCell(nn.Module):  # As described in https://arxiv.org/abs/1511.08228
    def __init__(self, l=1, m=256, depth_first=False):

        super(CGRUCell, self).__init__()

        self.l = l
        self.depth_first = depth_first

        kernel = 3

        self.u_gate = nn.Conv2d(m, m, kernel, padding=1)
        self.r_gate = nn.Conv2d(m, m, kernel, padding=1)
        self.conv_r = nn.Conv2d(m, m, kernel, padding=1)

    def step(self, s):
        u = torch.clamp(1.2 * Funct.sigmoid(self.u_gate(s)) - 0.1, min=0.0, max=1.0)
        r = torch.clamp(1.2 * Funct.sigmoid(self.r_gate(s)) - 0.1, min=0.0, max=1.0)

        output = u * s + (1.0 - u) * Funct.tanh(self.conv_r(r * s))

        return output

    def forward(self, s_t):
        if not self.depth_first:
            s_t = s_t.permute(2, 1, 0)  # go from (w, h, m) to (m, h, w) for pytorch conv

        s_tp1 = s_t
        for i in range(self.l):  # apply the CGRU cell recursively
            s_tp1 = self.step(s_tp1)

        if not self.depth_first:
            s_tp1 = s_tp1.permute(2, 1, 0)  # return it to (w, h, m)

        return s_tp1
