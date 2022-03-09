import math
from deepquantum import Circuit
import torch
import torch.nn as nn
import numpy as np
from deepquantum.nn.modules.qulinear import QuLinear


class QuGRUCell(nn.Module):
    """
    自定义QGRU单元, 其中经典线性变换被量子线性变换替代。

    参数:
    input_size: 输入x的特征数
    hidden_size: 隐藏状态的特征数

    输入:
    `x`: 张量形状是(N, input_size)， 输入矢量， 代表一个词语或字母的数字化表示
    `h_prev`: 张量形状是 (N, hidden_size)， 隐藏状态矢量， 代表之前输入到模型信息的数字化表示。
    其中N是Batch Size，用于把多个序列中同时间步的输入进行批量计算。

    输出:
    `h_new`: 张量形状是 (N, hidden_size)，隐藏状态矢量，代表考虑到当前输入后，
    隐藏状态的更新。
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size          # 隐藏层与输入层特征长度相等
        self.linear_x_r = QuLinear(input_size, self.hidden_size)
        self.linear_x_u = QuLinear(input_size, self.hidden_size)
        self.linear_x_n = QuLinear(input_size, self.hidden_size)
        self.linear_h_r = QuLinear(self.hidden_size, self.hidden_size)
        self.linear_h_u = QuLinear(self.hidden_size, self.hidden_size)
        self.linear_h_n = QuLinear(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h_prev):
        x_r = self.linear_x_r(x)
        x_u = self.linear_x_u(x)
        x_n = self.linear_x_n(x)
        h_r = self.linear_h_r(h_prev)
        h_u = self.linear_h_u(h_prev)
        h_n = self.linear_h_n(h_prev)

        resetgate = torch.sigmoid(x_r + h_r)
        updategate = torch.sigmoid(x_u + h_u)
        newgate = torch.tanh(x_n + (resetgate * h_n))
        h_new = newgate - updategate * newgate + updategate * h_prev

        return h_new


# if __name__ == '__main__':
#     n = 6
#     cell = QuGRUCell(n)
#     x = torch.rand(32, 1, 2**n) + 0j
#     print(cell(x, x).shape)


class QuGRU(nn.Module):
    """
    args:
        input_dim,  the size of embedding vector, i.e. embedding_dim
        hidden_dim, the size of GRU's hidden vector
        output_dim, the size of predicted vector

    input:
        tensor of shape (batch_size, seq_length, input_dim)

        e.g. input_dim is

    output:
        tensor of shape (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super().__init__()
        # input_dim: 比特数
        # assert input_dim == hidden_dim == output_dim

        self.hidden_dim = hidden_dim
        self.qgru_cell = QuGRUCell(input_dim, self.hidden_dim)
        self.fc = nn.Linear((1 << self.hidden_dim), (1 << output_dim))
        self.is_batch = True
        self.idim = (1 << input_dim)

    def forward(self, x):
        if x.ndim == 2:
            assert x.shape[1] == self.idim
            self.is_batch = False
        elif x.ndim == 3:
            assert x.shape[2] == self.idim
        else:
            raise ValueError("input x dimension error!")
        outputs = []
        # RNN 循环  初始化隐藏状态为零矢量

        if self.is_batch:
            h = torch.rand(x.shape[0], 1, (1 << self.hidden_dim)) + 0j
            for seq in range(x.shape[1]):
                tmp = x[:, seq, :].unsqueeze(1)
                h = self.qgru_cell(tmp, h)
                outputs.append(h)
            output = outputs[-1].real
            assert (output.shape[1] == 1) and (output.shape[-1] == (1 << self.hidden_dim))
        else:
            h = torch.rand(1, (1 << self.hidden_dim)) + 0j
            for seq in range(x.shape[0]):
                tmp = x[seq, :].unsqueeze(0)
                h = self.qgru_cell(tmp, h)
                outputs.append(h)
            output = outputs[-1].real
            assert (output.shape[0] == 1) and (output.shape[-1] == (1 << self.hidden_dim))

        output = self.fc(output)
        return output


if __name__ == '__main__':
    n = 8
    gru = QuGRU(n, n, 6)
    x = torch.rand(2, 4, 2**n) + 0j
    # x = torch.rand(6, 2**n) + 0j      # 非批处理
    print(gru(x).shape)
