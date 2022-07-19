import torch
import time
from deepquantum.nn.modules.quml import Qu_mutual


def quml_cpu_test():
    dev = 'cpu'
    n_layers = 3
    for n_qubits in range(4, 12, 2):
        n_sub = n_qubits // 2
        dimA, dimB = n_sub, n_sub
        start_time = time.perf_counter()
        inputA = torch.rand(2, 2 ** dimA, 2 ** dimA) + 0j
        inputB = torch.rand(2, 2 ** dimB, 2 ** dimB) + 0j
        QU=Qu_mutual(n_qubits, n_layers, dev=dev)
        rst = QU(inputA, inputB, dimA, dimB)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def quml_cuda_test():
    dev = 'cuda'
    n_layers = 3
    for n_qubits in range(4, 12, 2):
        n_sub = n_qubits // 2
        dimA, dimB = n_sub, n_sub
        start_time = time.perf_counter()
        inputA = torch.rand(2, 2 ** dimA, 2 ** dimA).cuda() + 0j
        inputB = torch.rand(2, 2 ** dimB, 2 ** dimB).cuda() + 0j
        QU=Qu_mutual(n_qubits, n_layers, dev=dev)
        rst = QU(inputA, inputB, dimA, dimB)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')



if __name__ == '__main__':
    # quml_cpu_test()
    quml_cuda_test()
