from deepquantum import Circuit
import torch
import torch.nn as nn
import numpy as np
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator
from deepquantum.gates.qmath import partial_trace, partial_trace_batched
import time


class QuPoolXYZ(nn.Module):
    """
    quantum pool layer
    """
    def __init__(self, n_qubits, gain=2**0.5, use_wscale=True, lrmul=1):
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6 * self.n_qubits - 3), a=0.0, b=2*np.pi)*init_std)

        # y = 3 * self.weight
        # y = y.sum()
        # # y.backward()
        # print(f'y.grad:\n{self.weight.grad}')

    def qpool(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        wires_lst = list(range(self.n_qubits))
        for which_q in range(self.n_qubits-1):
            cir.rxx(w[3 * which_q + 0], [which_q, which_q + 1])
            cir.ryy(w[3 * which_q + 1], [which_q, which_q + 1])
            cir.rzz(w[3 * which_q + 2], [which_q, which_q + 1])
            cir.cnot([which_q, which_q + 1])
        cir.ZYXLayer(wires_lst, w[3 * self.n_qubits - 3:])

        return cir

    def forward(self, x, dimA):
        cir = self.qpool()
        dimB = self.n_qubits - dimA
        trace_lst = list(range(dimB))

        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] == self.dim)
            batch_size = x.shape[0]
            x = x.permute(0, 2, 1) @ x.conj()  # 得到一个batched密度矩阵
            x = MatrixProductDensityOperator(x, self.n_qubits)
            x = cir.TN_contract_evolution(x, batch_mod=True)
            x = x.reshape(batch_size, self.dim, self.dim)
            print(f'x1.shape: {x.shape}')

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


class QuPoolSX(nn.Module):
    """
    Quantum Pool layer.
    """
    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(2 * self.n_qubits - 2), a=0.0, b=2 * np.pi) * init_std)

    def qpool(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(self.n_qubits - 1):
            cir.rxx(w[2 * which_q + 0], [which_q, which_q + 1])
            cir.cnot([which_q, which_q + 1])
            cir.rx(-w[2 * which_q + 1], which_q + 1)

        return cir

    def forward(self, x, dimA):
        cir = self.qpool()
        dimB = self.n_qubits - dimA
        trace_lst = list(range(dimB))

        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] == self.dim)
            x = x.permute(0, 2, 1) @ x.conj()  # 得到一个batched密度矩阵
            batch_size = x.shape[0]
            x = MatrixProductDensityOperator(x, self.n_qubits)
            x = cir.TN_contract_evolution(x, batch_mod=True)
            x = x.reshape(batch_size, self.dim, self.dim)
            # print(f'x1.shape: {x.shape}')

            x = partial_trace_batched(x, self.n_qubits, trace_lst)
            x = x.view([batch_size, 1, -1])
            assert x.shape[2] == self.dim
        elif x.ndim == 2:
            assert (x.shape[0] == 1) and (x.shape[1] == self.dim)
            x = x.permute(1, 0) @ x.conj()
            x = MatrixProductDensityOperator(x, self.n_qubits)
            x = cir.TN_contract_evolution(x, batch_mod=False)
            x = x.reshape(self.dim, self.dim)
            # print(f'x1.shape: {x.shape}')
            x = partial_trace(x, self.n_qubits, trace_lst)
            x = x.view([1, -1])
            assert x.shape[1] == self.dim
        else:
            raise ValueError("input x dimension error!")

        return x


if __name__ == '__main__':
    n_qubits = 10
    dimA = 5

    t1 = time.time()
    x = torch.rand(1, 1, 2**n_qubits) + 0j
    # print(f'x:\n{x}')

    x = nn.functional.normalize(x, dim=2)        # batch test
    # print(f'x:\n{x}')

    qpool = QuPoolXYZ(n_qubits)
    # qpool = QuPoolSX(n_qubits)

    rst = qpool(x, dimA).real
    print(rst.shape)
    t2 = time.time()
    print(f'time: {t2 - t1}')
    # print(f'\nrst:\n{rst[0, :, :10]}')
    #
    # rst = rst.sum()
    # print(rst.requires_grad)
    #
    # rst.backward()
    # print(f'\nweight:\n{qpool.weight}')
    # print(f'\nweight.grad:\n{qpool.weight.grad}')
