import time
import torch
import numpy as np
from deepquantum.gates.qoperator import cnot
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator
import csv


def cnot_test():
    N = 8
    wires = [0, 1]
    dev = 'cuda'
    c_not = cnot(N, wires, dev)
    state = torch.rand(1, 1 << N).cuda() + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = c_not.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MDO = MatrixProductDensityOperator(DM, N)
    rst2 = c_not.TN_contract_Rho(MDO)
    print(rst2.device)


def cnot_test_N_cpu():
    wires = [0, 1]
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        Rx = cnot(N, wires, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N).to(dev)
        st = []
        for _ in range(20):
            s = time.perf_counter()
            Rx.TN_contract(MPS)
            e = time.perf_counter()
            st.append(e-s)
        ts.append(np.mean(st) * 1000)
        print(np.mean(st) * 1000)

    # write_csv('D:/data.csv', ts)


def cnot_test_N_gpu():
    wires = [0, 1]
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        Rx = cnot(N, wires, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N).to(dev)
        st = []
        for _ in range(20):
            s = time.perf_counter()
            Rx.TN_contract(MPS)
            e = time.perf_counter()
            st.append(e-s)
        ts.append(np.mean(st) * 1000)
        print(np.mean(st) * 1000)
    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # cnot_test()
    # cnot_test_N_cpu()
    cnot_test_N_gpu()
