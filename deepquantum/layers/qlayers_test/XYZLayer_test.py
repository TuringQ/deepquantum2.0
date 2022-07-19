import csv
import time

import numpy as np
import torch

from deepquantum.layers.qlayers import XYZLayer
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def XYZLayer_test():
    N = 8
    # wire = 1
    wires = list(range(N))
    param_lst = [10.0] * N * 3
    dev = 'cuda'
    xyz_layer = XYZLayer(N, wires, param_lst, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = xyz_layer.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    rst2 = xyz_layer.TN_contract_Rho(MPDO)
    print(rst2.device)


def XYZLayer_test_N_cpu_dev():
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        param_lst = [10.0] * N * 3
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        xyz_layer = XYZLayer(N, wires, param_lst, dev)
        xyz_layer.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def XYZLayer_test_N_gpu_dev():
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        param_lst = [10.0] * N * 3
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        xyz_layer = XYZLayer(N, wires, param_lst, dev)
        xyz_layer.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # XYZLayer_test()

    # XYZLayer_test_N_cpu_dev()
    XYZLayer_test_N_gpu_dev()
