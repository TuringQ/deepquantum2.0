import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import PauliY
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def PauliY_test(dev):
    N = 8
    wire = 1
    pauliy = PauliY(N, wire, dev)
    state = torch.rand(1, 1 << N) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N).to(dev)
    rst = pauliy.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    MPDO = MPDO.to(dev)
    rst2 = pauliy.TN_contract_Rho(MPDO)
    print(rst2.device)


def PauliY_test_N_cpu_dev():
    wire = 1
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        pauliy = PauliY(N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        #     s = time.time()
        pauliy.TN_contract(MPS)
        e = time.perf_counter()
        # st.append(e-s)
        # ts.append(np.mean(st) * 1000)
        ts.append((e-s) * 1000)
        # print(np.mean(st) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)



def PauliY_test_N_gpu_dev():
    wire = 1
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        pauliy = PauliY(N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        pauliy.TN_contract(MPS)
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
    # PauliY_test('cpu')
    # PauliY_test('cuda')


    # PauliY_test_N_cpu_dev()
    PauliY_test_N_gpu_dev()
