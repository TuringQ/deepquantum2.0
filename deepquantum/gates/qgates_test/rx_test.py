import csv
import time

import numpy as np
import torch

from deepquantum.gates.qoperator import rx
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def rx_test():
    theta = 10.0
    N = 8
    wire = 1
    dev = 'cuda'
    Rx = rx(theta, N, wire, dev)
    state = torch.rand(1, 1 << N) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N).to(dev)
    rst = Rx.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    MPDO = MPDO.to(dev)
    rst2 = Rx.TN_contract_Rho(MPDO)
    print(rst2.device)


def rx_test_N_cpu():
    theta = 10.0
    wire = 1
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        Rx = rx(theta, N, wire, dev)
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
    # ts = np.array(ts)
    # np.savetxt('D:/data_cpu.csv', ts, delimiter=',')
    write_csv('D:/data.csv', ts)
    return ts


def rx_test_N_gpu():
    theta = 10.0
    wire = 1
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        Rx = rx(theta, N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)

        MPS = MPS.to(dev)
        st = []
        for _ in range(20):
            s = time.perf_counter()
            Rx.TN_contract(MPS)
            e = time.perf_counter()
            st.append(e-s)
        ts.append(np.mean(st) * 1000)
        print(np.mean(st) * 1000)
    write_csv('D:/data.csv', ts)
    return ts


def rx_test_N_cpu_dev():
    theta = 10.0
    wire = 1
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        Rx = rx(theta, N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        #     s = time.time()
        Rx.TN_contract(MPS)
        e = time.perf_counter()
        # st.append(e-s)
        # ts.append(np.mean(st) * 1000)
        ts.append((e-s) * 1000)
        # print(np.mean(st) * 1000)
        print((e-s) * 1000)
    # ts = np.array(ts)
    # np.savetxt('D:/data_cpu.csv', ts, delimiter=',')
    write_csv('D:/data.csv', ts)
    return ts


def rx_test_N_gpu_dev():
    theta = 10.0
    wire = 1
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        Rx = rx(theta, N, wire, dev)
        state = torch.rand(1, 1 << N) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        s = time.perf_counter()
        MPS = MPS.to(dev)
        # st = []
        # for _ in range(20):
        Rx.TN_contract(MPS)
        e = time.perf_counter()
        # st.append(e-s)
        # ts.append(np.mean(st) * 1000)
        ts.append((e-s) * 1000)
        # print(np.mean(st) * 1000)
        print((e-s) * 1000)

    write_csv('D:/data.csv', ts)
    return ts


def rx_rho_test_N_gpu_dev():
    theta = 10.0
    wire = 1
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        s = time.perf_counter()
        Rx = rx(theta, N, wire, dev)
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPDO = state.reshape(-1, 1) @ state.conj()
        Rx.TN_contract_Rho(MPDO)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)

    write_csv('D:/data.csv', ts)
    return ts


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # rx_test()

    # rx_test_N_cpu()
    # rx_test_N_gpu()

    # rx_test_N_cpu_dev()
    rx_test_N_gpu_dev()

    # rx_rho_test_N_gpu_dev()
