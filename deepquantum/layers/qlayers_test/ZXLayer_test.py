import csv
import time

import torch

from deepquantum.layers.qlayers import ZXLayer
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def ZXLayer_test():
    N = 8
    wires = list(range(N))
    param_lst = [10.0] * N * 2
    dev = 'cuda'
    zx_layer = ZXLayer(N, wires, param_lst, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = zx_layer.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    rst2 = zx_layer.TN_contract_Rho(MPDO)
    print(rst2.device)


def ZXLayer_test_N_cpu_dev():
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        param_lst = [10.0] * N * 2
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        zx_layer = ZXLayer(N, wires, param_lst, dev)
        zx_layer.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def ZXLayer_test_N_gpu_dev():
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        param_lst = [10.0] * N * 2
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        zx_layer = ZXLayer(N, wires, param_lst, dev)
        zx_layer.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # ZXLayer_test()

    # ZXLayer_test_N_cpu_dev()
    ZXLayer_test_N_gpu_dev()
