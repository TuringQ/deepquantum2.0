import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import rz
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def rz_test():
    theta = 10.0
    N = 8
    wire = 1
    dev = 'cuda'
    Rz = rz(theta, N, wire, dev)
    state = torch.rand(1, 1 << N) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N).to(dev)
    rst = Rz.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    MPDO = MPDO.to(dev)
    rst2 = Rz.TN_contract_Rho(MPDO)
    print(rst2.device)


def rz_test_N_cpu_dev():
    theta = 10.0
    wire = 1
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        Rz = rz(theta, N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        #     s = time.time()
        Rz.TN_contract(MPS)
        e = time.perf_counter()
        # st.append(e-s)
        # ts.append(np.mean(st) * 1000)
        ts.append((e-s) * 1000)
        # print(np.mean(st) * 1000)
        print((e-s) * 1000)
    # ts = np.array(ts)
    # np.savetxt('D:/data_cpu.csv', ts, delimiter=',')
    # write_csv('D:/data.csv', ts)


def rz_test_N_gpu_dev():
    theta = 10.0
    wire = 1
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        Rz = rz(theta, N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        Rz.TN_contract(MPS)
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
    # rz_test()

    # rz_test_N_cpu_dev()
    rz_test_N_gpu_dev()
