from deepquantum import Circuit
import torch
import torch.nn as nn
import numpy as np
from deepquantum.gates.qtensornetwork import StateVec2MPS, MPS2StateVec, MatrixProductDensityOperator
from deepquantum.gates.qmath import partial_trace, partial_trace_batched


class QuConvSXZ(nn.Module):
    """
    Simple Quantum Conv layer.
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        assert n_qubits % 2 == 0, "nquits应该为2的位数"
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits
        self.dim = (1 << self.n_qubits)
        self.N3 = 3 * self.n_qubits

    def qconv0(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            i = self.N3 + 3 * which_q
            cir.rxx(w[i], [which_q, which_q + 1])
            i += 1
            cir.ryy(w[i], [which_q, which_q + 1])
            i += 1
            cir.rzz(w[i], [which_q, which_q + 1])

        return cir

    def forward(self, x, dimA):
        cir = self.qconv0()
        dimB = self.n_qubits - dimA
        trace_lst = list(range(dimB))

        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] == self.dim)
            batch_size = x.shape[0]
            x = x.permute(0, 2, 1) @ x.conj()  # 得到一个batched密度矩阵
            x = MatrixProductDensityOperator(x, self.n_qubits)
            x = cir.TN_contract_evolution(x, batch_mod=True)
            x = x.reshape(batch_size, self.dim, self.dim)

            x = partial_trace_batched(x, self.n_qubits, trace_lst)
            x = x.view([batch_size, 1, -1])
            assert x.shape[2] == self.dim
        elif x.ndim == 2:
            assert (x.shape[0] == 1) and (x.shape[1] == self.dim)
            x = x.permute(1, 0) @ x.conj()
            x = MatrixProductDensityOperator(x, self.n_qubits)
            x = cir.TN_contract_evolution(x, batch_mod=False)
            x = x.reshape(self.dim, self.dim)

            x = partial_trace(x, self.n_qubits, trace_lst)
            x = x.view([1, -1])
            assert x.shape[1] == self.dim
        else:
            raise ValueError("input x dimension error!")

        return x


class QuConVXYZ(nn.Module):
    """
    Quantum Conv layer.
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        assert n_qubits % 2 == 0, "nqubits应该为2的倍数"
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(9*self.n_qubits - 3), a=0.0, b=2*np.pi) * init_std)

    def qconv0(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
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

        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] == self.dim)
            batch_size = x.shape[0]
            x = x.permute(0, 2, 1) @ x.conj()  # 得到一个batched密度矩阵
            x = MatrixProductDensityOperator(x, self.n_qubits)
            x = cir.TN_contract_evolution(x, batch_mod=True)
            x = x.reshape(batch_size, self.dim, self.dim)

            x = partial_trace_batched(x, self.n_qubits, trace_lst)
            x = x.view([batch_size, 1, -1])
            assert x.shape[2] == self.dim
        elif x.ndim == 2:
            assert (x.shape[0] == 1) and (x.shape[1] == self.dim)
            x = x.permute(1, 0) @ x.conj()
            x = MatrixProductDensityOperator(x, self.n_qubits)
            x = cir.TN_contract_evolution(x, batch_mod=False)
            x = x.reshape(self.dim, self.dim)

            x = partial_trace(x, self.n_qubits, trace_lst)
            x = x.view([1, -1])
            assert x.shape[1] == self.dim
        else:
            raise ValueError("input x dimension error!")

        return x


if __name__ == '__main__':
    n_qubits = 6
    dimA = 3
    quconv = QuConVXYZ(n_qubits)
    # quconv = QuConvSXZ(6)
    data = nn.functional.normalize(torch.rand(2, 1, 2**n_qubits, dtype=torch.cfloat), dim=2)
    rst = quconv(data, dimA)
    rst.sum().backward()
    print(f'weight:\n {quconv.weight}\n')
    print(f'weight.grad: \n{quconv.weight.grad}\n')

    print(f'\nrst.shape: {rst.shape}')
