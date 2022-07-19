import math
from deepquantum.gates.qcircuit import Circuit
import torch
import torch.nn as nn
import numpy as np
from deepquantum.gates.qmath import batched_kron2
import time
from deepquantum.gates.qmath import partial_trace_batched


class QuLinearAngleEmbedding(nn.Module):
    """
    quantum linear layer
    """
    def __init__(self, n_qubits,  n_layers=5, axis='Z', gain=2 ** 0.5, use_wscale=True, lrmul=1, dev='cpu'):
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

        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")

        self.n_param = self.N3 * (self.n_layers + 1)
        self.w = nn.Parameter(nn.init.uniform_(torch.empty(self.n_param).to(self.device), a=0.0, b=2 * np.pi)
                              * init_std * self.w_mul)

        self.cir = Circuit(self.n_qubits, dev)
        self.wires_lst = list(range(self.n_qubits))
        self.batches = 1
        self.embedding_axis = axis

    def encoding_layer(self, x):
        if self.dev == "gpu" or self.dev == "cuda":
            assert x.is_cuda, "------input of encoding-layer must be on-cuda-----"
        # if len(self.wires_lst) > self.n_qubits:
        #     raise ValueError("encoding-layer: number of wires must less than or equal to n_qubits")
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

    def add_variational_layer(self):
        for i in range(self.n_layers):
            index = i * self.N3
            self.cir.YZYLayer(self.wires_lst, self.w[index: index + self.N3])
            self.cir.ring_of_cnot(self.wires_lst)
        index += self.N3
        self.cir.YZYLayer(self.wires_lst, self.w[index:])

    def forward(self, x, dimA):
        assert dimA <= self.n_qubits, "dim of dimA must less than or equal to dim of n_qubits"
        self.cir.clear()
        dimB = self.n_qubits - dimA
        trace_lst = list(range(dimB))
        if x.ndim == 2:
            assert (x.shape[0] == 1) and (x.shape[1] == self.dim), \
                "shape of input x must be (1, 2**n_qubits), or be (batches, 1, 2**n_qubits)"
            x = x.unsqueeze(0)
        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] == self.dim), \
                "shape of input x must be (1, 2**n_qubits), or be (batches, 1, 2**n_qubits)"
            self.batches = x.shape[0]
            U_encoding = self.encoding_layer(x)
            self.add_variational_layer()
            state_vec = self.cir.state_init().reshape(1, -1)
            state_vec = (U_encoding @ (state_vec.permute(1, 0))).permute(0, 2, 1)
            MPS = state_vec.reshape([self.batches] + [2] * self.n_qubits)
            evo_res = self.cir.TN_contract_evolution(MPS, batch_mod=True)
            final_state = evo_res.reshape(self.batches, 1, -1)
            assert final_state.shape[2] == self.dim, \
                "shape of output must be (batches, 1, 2**n_qubits)"
            DM = final_state.permute(0, 2, 1) @ final_state.conj()
            reduced_DM = partial_trace_batched(DM, self.n_qubits, trace_lst, self.dev)
            assert reduced_DM.shape[1] == (1 << dimA) and reduced_DM.shape[2] == (1 << dimA), \
                "output shape must be [batches, 2**dimA, 2**dimA]"
            return reduced_DM
        else:
            raise ValueError("dimensionality of input must be 2 or 3")



if __name__ == '__main__':
    n_layers = 3
    # n_qubits = 4
    for n_qubits in range(4, 13):
        s = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)).cuda() + 0j
        qulinear = QuLinearAngleEmbedding(n_qubits, n_layers, 'Z', dev='cuda')
        res = qulinear(x, 2)      # 输出维度2**n_out, n_out <= n_qubits
        e = time.perf_counter()
        print((e - s) * 1000)

