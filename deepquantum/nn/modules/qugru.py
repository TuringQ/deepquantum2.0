import math
from deepquantum.gates.qcircuit import Circuit
import torch
import torch.nn as nn
import numpy as np
from deepquantum.gates.qmath import batched_kron2
import time


class QuLinear(nn.Module):
    def __init__(self, in_features, out_features, n_layers=1, axis='Z', gain=2 ** 0.5, use_wscale=True, lrmul=1, dev='cpu'):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_qubits = in_features
        self.n_part = out_features
        self.n_layers = n_layers
        self.N3 = 3 * self.n_qubits

        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")

        self.w = nn.Parameter(nn.init.uniform_(torch.empty(self.N3 * (self.n_layers + 1)).to(self.device),
                                               a=0.0, b=2 * np.pi) * init_std * self.w_mul)
        self.wires_lst = list(range(self.n_qubits))
        self.embedding_axis = axis
        self.cir = Circuit(self.n_qubits, dev)

    def partial_measurements(self, final_state, n_qubits, n_part):
        # type: (Tensor, int, int) -> Tensor
        diff = (1 << (n_qubits - n_part))
        tmp = final_state.reshape(final_state.shape[0], 1, -1, diff)
        res = tmp.abs().square().sum(dim=3)
        return res

    def encoding_layer(self, x):
        if self.dev == "gpu" or self.dev == "cuda":
            assert x.is_cuda, "------input of encoding-layer must be on-cuda-----"

        I = torch.eye(2, 2).to(self.device)
        if self.embedding_axis == 'X':
            PauliMat = torch.tensor([[0., 1], [1, 0]]).to(self.device) + 0j
        elif self.embedding_axis == 'Y':
            PauliMat = torch.tensor([[0., -1j], [1j, 0]]).to(self.device) + 0j
        elif self.embedding_axis == 'Z':
            PauliMat = torch.tensor([[1., 0], [0, -1]]).to(self.device) + 0j
        else:
            raise ValueError("embedding axis must be one of 'X', 'Y' or 'Z'")
        c, s = torch.cos(0.5 * x), torch.sin(0.5 * x)
        lst = [torch.ones(x.shape[0], 1, 1).to(self.device).kron(torch.eye(2).to(self.device))] * self.n_qubits
        for i, qbit in enumerate(self.wires_lst):
            m = c[:, :, i].unsqueeze(1).kron(I) - 1j * s[:, :, i].unsqueeze(1).kron(PauliMat)
            lst[qbit] = m
        rst = lst[0]
        for i in range(1, len(lst)):
            rst = batched_kron2(rst, lst[i])
        return rst

    def variational_layer(self):
        for i in range(self.n_layers):
            index = i * self.N3
            self.cir.YZYLayer(self.wires_lst, self.w[index: index + self.N3])
            self.cir.ring_of_cnot(self.wires_lst)
        index += self.N3
        self.cir.YZYLayer(self.wires_lst, self.w[index:])

    def forward(self, x):
        if x.ndim == 2:
            assert (x.shape[0] == 1) and (x.shape[1] == (1 << self.n_qubits)), \
                "shape of input x must be (1, 2**n_qubits), or be (batches, 1, 2**n_qubits)"
            x = x.unsqueeze(0)
        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] == (1 << self.n_qubits)), \
                "shape of input x must be (1, 2**n_qubits), or be (batches, 1, 2**n_qubits)"

        self.cir.clear()
        init_state = self.cir.state_init().reshape(1, -1)
        U_encoding = self.encoding_layer(x)
        self.variational_layer()

        state_vec = (U_encoding @ (init_state.permute(1, 0))).permute(0, 2, 1)
        U = self.cir.U()
        tmp = U @ (state_vec.permute(0, 2, 1))
        final_state = tmp.permute(0, 2, 1)
        hidden = self.partial_measurements(final_state, self.n_qubits, self.n_part)
        return hidden


class QuGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, dev='cpu'):
        super().__init__()
        self.hidden_size = hidden_size          # 隐藏层与输入层特征长度相等
        self.linear_x_r = QuLinear(input_size, hidden_size, dev=dev)
        self.linear_x_u = QuLinear(input_size, hidden_size, dev=dev)
        self.linear_x_n = QuLinear(input_size, hidden_size, dev=dev)
        self.linear_h_r = QuLinear(self.hidden_size, hidden_size, dev=dev)
        self.linear_h_u = QuLinear(self.hidden_size, hidden_size, dev=dev)
        self.linear_h_n = QuLinear(self.hidden_size, hidden_size, dev=dev)

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

    def __init__(self, input_dim, hidden_dim, output_dim, bias=True, dev='cpu'):
        super().__init__()
        # input_dim: 比特数
        # assert input_dim == hidden_dim == output_dim
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.hidden_dim = hidden_dim
        self.qgru_cell = QuGRUCell(input_dim, self.hidden_dim, dev=dev)
        self.fc = nn.Linear(2 ** self.hidden_dim, output_dim).to(self.device)

    def forward(self, x):
        outputs = []
        h = torch.zeros(x.shape[0], 1, 2**self.hidden_dim).to(self.device) + 0j
        for seq in range(x.shape[1]):
            tmp = x[:, seq, :].unsqueeze(1)
            h = self.qgru_cell(tmp, h)      # GRU网络，GRUCell算子传播
            outputs.append(h)
        output = outputs[-1].real
        output = self.fc(output)
        return output

# #
# if __name__ == '__main__':
#     n_qubits = 8        # 量子线路比特
#     output_qubits = 9   # 定义输出大小
#     gru = QuGRU(n_qubits, n_qubits, output_qubits)  # 定义GRU网络
#     x = torch.rand(3, 4, 2**n_qubits) + 0j      # 输入 --- 支持批处理
#     print(f'|---> input shape: {x.shape}')
#     print(f'|---> shape of GRU output: {gru(x).shape}')
