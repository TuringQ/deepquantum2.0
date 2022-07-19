import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from deepquantum.gates.qcircuit import Circuit
from deepquantum.nn.modules.measure import measure_mps
import time


class Policy(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4, pauli_string="", dimA=14, gain=2 ** 0.5, use_wscale=True, lrmul=1, dev='cpu'):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_qubits = n_qubits
        self.dim = (1 << self.n_qubits)
        self.N3 = self.n_qubits * 3
        self.n_layers = n_layers
        self.pauli_string = pauli_string
        pauli_str_n = n_qubits if pauli_string == "" else len(pauli_string.split(','))
        self.dimA = dimA
        self.dev = dev
        device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(2 * self.n_qubits * self.n_layers).to(device),
                                                    a=0.0, b=2*np.pi) * init_std)
        self.theta = nn.Parameter(torch.ones(self.N3 * self.n_layers).to(device))
        self.q1 = nn.Linear(pauli_str_n, self.dimA).to(device)
        self.wires = list(range(self.n_qubits))

        self.saved_log_probs = []
        self.rewards = []

    def layers(self, x):
        # w = self.weight * self.w_mul
        x = x.squeeze()
        cir = Circuit(self.n_qubits, self.dev)
        for k in range(self.n_layers):
            i1 = k * self.N3
            cir.ZYZLayer(self.wires, self.theta[i1: i1 + self.N3])
            cir.ring_of_cnot(self.wires)
            for i in range(self.n_qubits):
                i2 = i * 2
                cir.ry(self.weight[i2] * x[i], i)
                cir.rz(self.weight[i2+1] * x[i], i)
        return cir


    def forward(self, x):  # measure_mps
        assert (x.ndim == 2) and (x.shape[0] == 1) and (x.shape[1] >= self.n_qubits), \
            "shape of x must be (1, n_qubits)"
        cir = self.layers(x)
        init_state = cir.state_init().unsqueeze(0)
        MPS = init_state.view([2] * self.n_qubits)
        TN_res = cir.TN_contract_evolution(MPS, batch_mod=False)

        q = measure_mps(TN_res, self.n_qubits, self.pauli_string, dev=self.dev)
        res = self.q1(q)
        return F.softmax(res, dim=0)

