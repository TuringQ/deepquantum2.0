from deepquantum.gates.qcircuit import Circuit
import torch
import torch.nn as nn
import numpy as np
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator
from deepquantum.gates.qmath import partial_trace_batched
import time


class QuConVXYZ(nn.Module):
    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1, dev='cpu'):
        super().__init__()
        assert n_qubits % 2 == 0, "n_qubits必须为2的倍数"
        he_std = gain * 5 ** (-0.5)     # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
            self.n_qubits = n_qubits
        self.n_qubits = n_qubits
        self.N3 = 3 * self.n_qubits
        self.dim = (1 << self.n_qubits)
        self.dev = dev
        device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(9*self.n_qubits - 3).to(device), a=0.0,
                                                    b=2*np.pi) * init_std)

    def qconv0(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits, self.dev)
        wires_lst = list(range(self.n_qubits))
        cir.XYZLayer(wires_lst, w[0: self.N3])
        for which_q in range(0, self.n_qubits - 1, 1):
            i = self.N3 + 3 * which_q
            cir.rxx(w[i], [which_q, which_q + 1])
            i += 1
            cir.ryy(w[i], [which_q, which_q + 1])
            i += 1
            cir.rzz(w[i], [which_q, which_q + 1])
        cir.XYZLayer(wires_lst, w[6 * self.n_qubits - 3:])
        return cir

    def forward(self, x, dimA):
        cir = self.qconv0()
        dimB = self.n_qubits - dimA
        trace_lst = list(range(dimB))
        if x.ndim == 2:
            assert ((x.shape[0] == 1) or (x.shape[0] == self.dim)) and (x.shape[1] == self.dim), \
                "2D input shape must be [1, 2**n] or [2**n, 2**n]"
            x = x.unsqueeze(0)
        if x.ndim == 3:
            assert ((x.shape[1] == 1) or (x.shape[1] == self.dim)) and (x.shape[2] == self.dim), \
                "3D input shape must be [batches, 1, 2**n] or [batches, 2**n, 2**n]"
            batch_size = x.shape[0]
            if x.shape[1] == 1:           # 输入为态矢情况
                MPS = x.reshape([batch_size] + [2] * self.n_qubits)
                res = cir.TN_contract_evolution(MPS, batch_mod=True)
                final_state = res.reshape(batch_size, 1, -1)
                assert final_state.shape[2] == self.dim, "shape of final state must be [batches, 1, 2**n]"
                DM = final_state.permute(0, 2, 1) @ final_state.conj()
                reduced_DM = partial_trace_batched(DM, self.n_qubits, trace_lst, self.dev)
                assert reduced_DM.shape[1] == (1 << dimA) and reduced_DM.shape[2] == (1 << dimA), \
                    "output shape must be [batches, 2**dimA, 2**dimA]"
                return reduced_DM
            elif x.shape[1] == self.dim:  # 输入为密度矩阵情况
                MPDO = MatrixProductDensityOperator(x, self.n_qubits)
                res = cir.TN_contract_evolution(MPDO, batch_mod=True)
                res_DM = res.reshape(batch_size, self.dim, self.dim)
                reduced_DM = partial_trace_batched(res_DM, self.n_qubits, trace_lst, self.dev)
                assert reduced_DM.shape[1] == (1 << dimA) and reduced_DM.shape[2] == (1 << dimA), \
                    "output shape must be [batches, 2**dimA, 2**dimA]"
                return reduced_DM
            else:
                raise ValueError("input x dimension error!")
        else:
            raise ValueError("input x dimension error!")


if __name__ == '__main__':
    dimA = 2
    bs = 10
    for n_qubits in range(2, 14, 2):
        s = time.perf_counter()
        x = torch.rand((bs, 1, 2**n_qubits)).cuda() + 0j
        quconv = QuConVXYZ(n_qubits, dev='cuda')
        rst = quconv(x, dimA)
        e = time.perf_counter()
        print((e - s) * 1000)





