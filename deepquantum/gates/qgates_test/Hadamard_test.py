import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import Hadamard
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def Hadamard_test():
    N = 8
    wire = 1
    dev = 'cpu'
    hadamard = Hadamard(N, wire, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = hadamard.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    MPDO = MPDO.to(dev)
    rst2 = hadamard.TN_contract_Rho(MPDO)
    print(rst2.device)


def Hadamard_test_N_cpu_dev():
    wire = 1
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        hadamard = Hadamard(N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        #     s = time.time()
        hadamard.TN_contract(MPS)
        e = time.perf_counter()
        # st.append(e-s)
        # ts.append(np.mean(st) * 1000)
        ts.append((e-s) * 1000)
        # print(np.mean(st) * 1000)
        print((e-s) * 1000)

    # write_csv('D:/data.csv', ts)



def Hadamard_test_N_gpu_dev():
    wire = 1
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        hadamard = Hadamard(N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        hadamard.TN_contract(MPS)
        e = time.perf_counter()
        # st.append(e-s)
        # ts.append(np.mean(st) * 1000)
        ts.append((e-s) * 1000)
        # print(np.mean(st) * 1000)
        print((e-s) * 1000)

    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # Hadamard_test()

    # Hadamard_test_N_cpu_dev()
    Hadamard_test_N_gpu_dev()
