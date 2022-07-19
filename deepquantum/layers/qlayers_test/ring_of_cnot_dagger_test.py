import csv
import time
import torch

from deepquantum.layers.qlayers import ring_of_cnot_dagger
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


def ring_of_cnot_dagger_test():
    N = 8
    wires = list(range(N))
    dev = 'cuda'
    Ring_Of_CNot_Dagger = ring_of_cnot_dagger(N, wires, dev)
    state = torch.rand(1, 1 << N).to(dev) + 0j
    state = torch.nn.functional.normalize(state, p=2, dim=1)
    MPS = state.reshape([2] * N)
    rst = Ring_Of_CNot_Dagger.TN_contract(MPS)
    print(rst.device)

    DM = state.permute(1, 0) @ state.conj()
    MPDO = MatrixProductDensityOperator(DM, N)
    rst2 = Ring_Of_CNot_Dagger.TN_contract_Rho(MPDO)
    print(rst2.device)


def ring_of_cnot_dagger_test_N_cpu_dev():
    dev = 'cpu'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        Ring_Of_CNot_Dagger = ring_of_cnot_dagger(N, wires, dev)
        Ring_Of_CNot_Dagger.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def ring_of_cnot_dagger_test_N_gpu_dev():
    dev = 'cuda'
    ts = []
    for N in range(12, 28):
        wires = list(range(N))
        s = time.perf_counter()
        state = torch.rand(1, 1 << N).to(dev) + 0j
        state = torch.nn.functional.normalize(state, p=2, dim=1)
        MPS = state.reshape([2] * N)
        Ring_Of_CNot_Dagger = ring_of_cnot_dagger(N, wires, dev)
        Ring_Of_CNot_Dagger.TN_contract(MPS)
        e = time.perf_counter()
        ts.append((e-s) * 1000)
        print((e-s) * 1000)
    # write_csv('D:/data.csv', ts)


def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # ring_of_cnot_dagger_test()

    # ring_of_cnot_dagger_test_N_cpu_dev()
    ring_of_cnot_dagger_test_N_gpu_dev()
