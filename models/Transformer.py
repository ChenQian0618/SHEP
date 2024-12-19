# 增加了位置编码

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 三个线性层做矩阵乘法生成q, k, v.
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        # ScaledDotProductAttention见下方
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # b: batch_size, lq: translation task的seq长度, n: head数, dv: embedding vector length
        # Separate different heads: b x lq x n x dv.
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # project & reshape
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
            # (batchSize, 1, seqLen) -> (batchSize, 1, 1, seqLen)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # view只能用在contiguous的variable上
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # add & norm
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q x k^T
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 把mask中为0的数置为-1e9, 用于decoder中的masked self-attention
            attn = attn.masked_fill(mask == 0, -1e9)

        # dim=-1表示对最后一维softmax
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        # add & norm
        x += residual

        x = self.layer_norm(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        # 将tensor注册成buffer, optim.step()的时候不会更新
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            # 2i, 所以此处要//2.
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # shape:(1, maxLen(n_position), d_hid)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()  # 数据、梯度均无关


class Transformer(nn.Module):
    def __init__(self, in_channel=1, out_channel=10, L=2000, **kwargs):
        super(Transformer, self).__init__()
        self.conv_k, self.conv_s, self.conv_c = 50, 25, 256
        self.nhead = 4
        self.conv1 = nn.Conv1d(in_channel, self.conv_c, kernel_size=self.conv_k, stride=self.conv_s, padding=0)
        new_L = (L - self.conv_k) // self.conv_s + 1
        self.position_enc = PositionalEncoding(self.conv_c, n_position=new_L)
        self.Attention = MultiHeadAttention(self.nhead, self.conv_c, self.conv_c // self.nhead,
                                            self.conv_c // self.nhead)
        self.FeedForward = PositionwiseFeedForward(self.conv_c, self.conv_c * 4)
        self.max, self.avg = nn.AdaptiveMaxPool1d(4), nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(nn.Linear(8 * new_L, 64),
                                nn.ReLU(),
                                nn.Linear(64, out_channel))

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2)
        x = self.position_enc(x)
        x, _ = self.Attention(x, x, x)
        x = self.FeedForward(x)
        x = torch.concat((self.max(x), self.avg(x)), dim=-1)
        z = x.view(x.size(0), -1)
        x = self.fc(z)
        if verbose:
            return x, z
        else:
            return x


if __name__ == '__main__':
    import torch
    from utils.mysummary import summary

    for L in [1024, 2000, 5000]:
        model = Transformer(L=L).to(torch.device('cuda'))
        input = torch.randn(1, 1, L).to(torch.device('cuda'))
        output = model(input)
        summary(model, (1, L))
