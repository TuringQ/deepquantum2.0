import csv
import time

import numpy as np
import torch

from deepquantum.layers.qlayers import YZYLayer
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def YZYLayer_test():
    N = 8
    # wire = 1
    wires = list(range(N))
    param_lst = [10.0] * N * 3
    dev = 'cuda'
    yzy_layer = YZYLayer(N, wires, param_lst, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = yzy_layer.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    rst2 = yzy_layer.TN_contract_Rho(MPDO)
    print(rst2.device)


def YZYLayer_test_N_cpu_dev():
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        param_lst = [10.0] * N * 3
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        yzy_layer = YZYLayer(N, wires, param_lst, dev)
        yzy_layer.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def YZYLayer_test_N_gpu_dev():
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        param_lst = [10.0] * N * 3
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        yzy_layer = YZYLayer(N, wires, param_lst, dev)
        yzy_layer.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # YZYLayer_test()

    # YZYLayer_test_N_cpu_dev()
    YZYLayer_test_N_gpu_dev()
