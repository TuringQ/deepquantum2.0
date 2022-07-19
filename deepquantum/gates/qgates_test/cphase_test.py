import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import cphase
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def cphase_test():
    theta = 10.0
    N = 8
    wire = [0, 1]
    dev = 'cuda'
    Cphase = cphase(theta, N, wire, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = Cphase.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    rst2 = Cphase.TN_contract_Rho(MPDO)
    print(rst2.device)


def cphase_test_N_cpu_dev():
    theta = 10.0
    wire = [0, 1]
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        Cphase = cphase(theta, N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        Cphase.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def cphase_test_N_gpu_dev():
    theta = 10.0
    wire = [0, 1]
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        Cphase = cphase(theta, N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        Cphase.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)

    # write_csv('D:/data.csv', ts)


def cphase_rho_test_N_gpu_dev():
    theta = 10.0
    wire = [0, 1]
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        Cphase = cphase(theta, N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPDO = state.reshape(-1, 1) @ state.conj()
        print(f'|---> MPDO.shape: {MPDO.shape}')
        Cphase.TN_contract_Rho(MPDO)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)

    # write_csv('D:/data.csv', ts)



def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # cphase_test()

    # cphase_test_N_cpu_dev()
    cphase_test_N_gpu_dev()

    # cphase_rho_test_N_gpu_dev()
