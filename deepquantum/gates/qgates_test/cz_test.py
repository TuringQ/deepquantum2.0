import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import cz
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def cz_test():
    N = 8
    wire = [0, 1]
    dev = 'cuda'
    Cz = cz( N, wire, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = Cz.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    rst2 = Cz.TN_contract_Rho(MPDO)
    print(rst2.device)


def cz_test_N_cpu_dev():
    wire = [0, 1]
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        Cz = cz(N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        Cz.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def cz_test_N_gpu_dev():
    wire = [0, 1]
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        Cz = cz(N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        Cz.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)

    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # cz_test()

    # cz_test_N_cpu_dev()
    cz_test_N_gpu_dev()
