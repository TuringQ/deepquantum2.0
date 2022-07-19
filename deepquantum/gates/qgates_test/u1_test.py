import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import u1
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def u1_test():
    theta = 10.0
    N = 8
    wire = 1
    dev = 'cuda'
    U1 = u1(theta, N, wire, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = U1.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    rst2 = U1.TN_contract_Rho(MPDO)
    print(rst2.device)


def u1_test_N_cpu_dev():
    theta = 10.0
    wire = 1
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        U1 = u1(theta, N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        U1.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def u1_test_N_gpu_dev():
    theta = 10.0
    wire = 1
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        U1 = u1(theta, N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        U1.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # u1_test()

    u1_test_N_cpu_dev()
    # u1_test_N_gpu_dev()
