import math
from deepquantum import Circuit
import torch
import torch.nn as nn
import numpy as np


class QuLinear(nn.Module):
    """
    quantum linear layer
    """
    def __init__(self, n_qubits,  n_layers=2, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_qubits = n_qubits
        self.N3 = 3 * self.n_qubits
        self.dim = (1 << self.n_qubits)  # 2**n_qubits
        self.n_layers = n_layers

        self.n_param = self.N3 * (self.n_layers + 1)
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(self.n_param), a=0.0, b=2 * np.pi) * init_std)
        self.w = self.weight * self.w_mul

        self.cir = Circuit(self.n_qubits)
        self.wires_lst = list(range(self.n_qubits))
        self.is_batch = False

    def get_zeros_state(self, n_qubits):
        zero_state = torch.zeros(1, (1 << n_qubits)) + 0j
        zero_state[0][0] = 1. + 0j
        return zero_state

    def encoding_layer(self, data):
        for which_q in range(self.n_qubits):
            self.cir.rx(data[which_q], which_q)

    def forward(self, x):
        self.cir.clear()

        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] <= self.dim)
            if x.shape[2] < self.dim:
                pad = nn.ZeroPad2d(padding=(0, self.dim - x.shape[2], 0, 0))
                x = pad(x)
            self.is_batch = True
        elif x.ndim == 2:
            assert (x.shape[0] == 1) and (x.shape[1] <= self.n_qubits)
            if x.shape[1] < self.n_qubits:
                pad = nn.ZeroPad2d(padding=(0, self.n_qubits - x.shape[1], 0, 0))
                x = pad(x)
                x_min = x.min(1, keepdim=True)[0]
                x_max = x.max(1, keepdim=True)[0]
                x = (x - x_min) / (x_max - x_min) * 2 * torch.pi
            x = x.squeeze()
            self.encoding_layer(x)
        else:
            raise ValueError("input x dimension error!")

        for i in range(self.n_layers):
            index = i * self.N3
            self.cir.YZYLayer(self.wires_lst, self.w[index: index + self.N3])
            self.cir.ring_of_cnot(self.wires_lst)
        index += self.N3
        self.cir.YZYLayer(self.wires_lst, self.w[index:])

        if self.is_batch:
            x = x.view([x.shape[0]] + [2] * self.n_qubits)
            x = self.cir.TN_contract_evolution(x, batch_mod=True)
            x = x.reshape(x.shape[0], 1, -1)
            assert x.shape[2] == self.dim
        else:
            # x = nn.functional.normalize(x, dim=1)
            x = self.get_zeros_state(self.n_qubits)
            x = x.view([2] * self.n_qubits)
            x = self.cir.TN_contract_evolution(x, batch_mod=False)
            x = x.reshape(1, -1)
            assert x.shape[1] == self.dim

        return x


if __name__ == '__main__':
    n_qubits = 8
    # n_layers = 2
    # x = torch.rand((4, 1, n_qubits)) + 0j
    # x = nn.functional.normalize(x, dim=2)
    x = torch.rand((1, n_qubits))
    qulinear = QuLinear(n_qubits)
    res = qulinear(x)
    # r = res.sum()
    # r.backward()
    # print(qulinear.weight.grad)
    print(res.shape)

