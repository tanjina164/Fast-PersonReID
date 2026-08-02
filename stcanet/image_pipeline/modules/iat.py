"""
IAT (Interconnection-Accumulation-Transformation) module -- the combined
CAM (Channel Attention Module) + SAM (Spatial Attention Module) block from
the original STCANet paper's 2D (image-based) architecture.

This is a direct port from the Deepreid reference notebook
("Image Based Person ReID STCANet.ipynb"), with one change: .cuda() calls
are replaced with device-agnostic .to(x.device) so the module works
correctly regardless of which GPU/CPU the input tensor is on (needed for
FastReID's DataParallel / multi-GPU training setup).

Architecture is otherwise UNCHANGED from the reference notebook, per the
project's core constraint: only the mask-generation source (SHP-LIP -> SCHP)
and backbone/framework (deep-person-reid -> FastReID) are being replaced;
the attention module design itself is preserved exactly.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Gconv(nn.Module):
    def __init__(self, in_channels):
        super(Gconv, self).__init__()
        fsm_blocks = []
        fsm_blocks.append(nn.Conv2d(in_channels * 2, in_channels, 1))
        fsm_blocks.append(nn.BatchNorm2d(in_channels))
        fsm_blocks.append(nn.ReLU(inplace=True))
        self.fsm = nn.Sequential(*fsm_blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, W, x):
        bs, n, c = x.size()
        x_neighbor = torch.bmm(W, x)
        x = torch.cat([x, x_neighbor], 2)
        x = x.view(-1, x.size(2), 1, 1)
        x = self.fsm(x)
        x = x.view(bs, n, c)
        return x


class Wcompute(nn.Module):
    def __init__(self, in_channels):
        super(Wcompute, self).__init__()
        self.in_channels = in_channels
        edge_block = []
        edge_block.append(nn.Conv2d(in_channels * 2, 1, 1))
        edge_block.append(nn.BatchNorm2d(1))
        self.relation = nn.Sequential(*edge_block)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, W_id, y):
        bs, N, C = x.size()
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2)
        W_new = torch.abs(W1 - W2)
        W_new = torch.transpose(W_new, 1, 3)
        y = y.view(bs, C, 1, 1).expand_as(W_new)
        W_new = torch.cat((W_new, y), 1)
        W_new = self.relation(W_new)
        W_new = torch.transpose(W_new, 1, 3)
        W_new = W_new.squeeze(3)
        W_new = W_new - W_id.expand_as(W_new) * 1e8
        W_new = F.softmax(W_new, dim=2)
        return W_new


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.in_channels = in_channels
        self.module_w = Wcompute(in_channels)
        self.module_l = Gconv(in_channels)

    def forward(self, x, y):
        bs, N, C = x.size()
        W_init = torch.eye(N, device=x.device).unsqueeze(0)
        W_init = W_init.repeat(bs, 1, 1)
        W = self.module_w(x, W_init, y)
        s = self.module_l(W, x)
        return s


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class SpatialAttn(nn.Module):
    """Spatial Attention -- produces a 4-channel attention map, one channel
    per body-part group (head, upper_clothes, lower_clothes, shoes)."""
    def __init__(self, in_channels, number):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBlock(in_channels, number, 1)

    def forward(self, x):
        x = self.conv(x)
        a = torch.sigmoid(x)
        return a


class IAT(nn.Module):
    """CAM (channel self-attention) + SAM (part-based graph attention).
    Input: feature map (b, in_channels, h, w).
    Output: (refined_feature_map, part_attention_maps [b, 4, h, w])
    """
    def __init__(self, in_channels):
        super(IAT, self).__init__()
        inter_stride = 2
        self.in_channels = in_channels
        self.inter_channels = in_channels // inter_stride

        self.sa = SpatialAttn(in_channels, number=4)
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.SAM = SAM(self.inter_channels)

        self.W1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )
        self.W2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W1[1].weight.data, 0.0)
        nn.init.constant_(self.W1[1].bias.data, 0.0)
        nn.init.constant_(self.W2[1].weight.data, 0.0)
        nn.init.constant_(self.W2[1].bias.data, 0.0)

    def reduce_dimension(self, x, global_node):
        bs, c = global_node.size()
        x = x.transpose(1, 2).unsqueeze(3)
        x = torch.cat((x, global_node.view(bs, c, 1, 1)), 2)
        x = self.g(x).squeeze(3)
        global_node = x[:, :, -1]
        x = x[:, :, :-1].transpose(1, 2)
        return x, global_node

    def forward(self, x):
        # CAM
        batch_size = x.size(0)
        g_x = x.view(batch_size, self.in_channels, -1)
        theta_x = g_x
        phi_x = g_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=-1)
        y = torch.matmul(f, g_x)
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W1(y)
        z = y + x

        # SAM
        x = z
        inputs = x
        b, c, h, w = x.size()
        u = x.view(b, c, -1).mean(2)

        a = self.sa(x)
        x = torch.bmm(a.view(b, -1, h * w), x.view(b, c, -1).transpose(1, 2))
        x, u = self.reduce_dimension(x, u)
        y = self.SAM(x, u)

        y = torch.mean(y, 1)
        u = torch.cat((y, u), 1)

        y = self.W2(u.view(u.size(0), u.size(1), 1, 1))
        z = y + inputs
        return z, a
