import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from deepquantum import Circuit
from deepquantum.gates.qmath import dag

DEBUG = True


class Policy(nn.Module):
    """
    Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置n个量子门，也即有x个参数。
    """

    def __init__(self, n_qubits=3, layer=3, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight0 = nn.Parameter(nn.init.uniform_(torch.empty(14 * 3 * layer * n_qubits), a=0.0, b=2*np.pi) * init_std)
        self.theta = nn.Parameter(torch.ones(14 * 3 * layer * n_qubits))
        self.n_qubits = n_qubits
        self.N3 = self.n_qubits * 3
        self.layer = layer
        self.q1 = nn.Linear(2**self.n_qubits, 14)

        self.saved_log_probs = []
        self.rewards = []

    # decision circuit to replace neural network. We use 4 qubits and get a density matrix of 16x16.
    def layers(self, x):
        # w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        wires = list(range(self.n_qubits))
        wt = self.weight0 * (x[:14].repeat(3 * self.layer * self.n_qubits))
        for k in range(self.layer):
            i = k * self.N3 * 14
            cir.ZYZLayer(wires, wt[i: i + self.N3] + self.theta[i: i + self.N3])
            i += self.N3
            cir.ZYZLayer(wires, wt[i: i + self.N3] + self.theta[i: i + self.N3])
            i += self.N3
            cir.ZYZLayer(wires, wt[i: i + self.N3] + self.theta[i: i + self.N3])
            i += self.N3
            cir.ZYZLayer(wires, wt[i: i + self.N3] + self.theta[i: i + self.N3])
            i += self.N3
            w = wt[i: i + self.n_qubits * 2] + self.theta[i: i + self.n_qubits * 2]
            w = w.reshape(-1, 2)
            pad = nn.ZeroPad2d(padding=(0, 1, 0, 0))
            w = pad(w)
            w = w.view(-1)
            cir.ZYZLayer(wires, w)

            if k != self.layer - 1:
                if self.n_qubits > 1:
                    for j in range(self.n_qubits - 1):
                        cir.cz([j, j + 1])
                if self.n_qubits > 2:
                    cir.cz([self.n_qubits - 1, 0])
        # U = cir.U()
        # state = U @ cir.state_init()
        # return state.reshape(-1, 1)
        return cir

    def forward(self, x):
        assert (x.ndim == 1) and (14 <= x.shape[0] <= (1 << self.n_qubits))
        cir = self.layers(x)

        x = cir.state_init()
        x = x.unsqueeze(0)
        x = x.view([2] * self.n_qubits)
        x = cir.TN_contract_evolution(x, batch_mod=False)
        x = x.reshape(1, -1)
        assert x.shape[1] == (1 << self.n_qubits)

        output = x
        output1 = dag(output)
        q = (output1[0] * output.reshape(1, int(1 << self.n_qubits))[0]).real
        q = self.q1(q)
        return F.softmax(q, dim=0)


if __name__ == '__main__':
    n_qubits = 8
    n_layers = 4
    x = torch.rand((1 << n_qubits)) + 0j
    cir = Policy(n_qubits, n_layers)
    print(cir(x).shape)
