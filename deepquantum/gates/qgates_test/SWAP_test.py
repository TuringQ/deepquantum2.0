import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import SWAP
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def SWAP_test():
    theta = 10.0
    N = 8
    wire = [0, 1]
    dev = 'cuda'
    swap = SWAP(N, wire, dev)
    state = torch.rand(1, 1 << N) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N).to(dev)
    rst = swap.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    MPDO = MPDO.to(dev)
    rst2 = swap.TN_contract_Rho(MPDO)
    print(rst2.device)



def SWAP_test_N_cpu_dev():
    theta = 10.0
    wire = [0, 1]
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        swap = SWAP(N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        swap.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def SWAP_test_N_gpu_dev():
    theta = 10.0
    wire = [0, 1]
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        swap = SWAP(N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        swap.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)

    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # SWAP_test()

    SWAP_test_N_cpu_dev()
    # SWAP_test_N_gpu_dev()
