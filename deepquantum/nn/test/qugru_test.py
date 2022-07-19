import torch
import time
from deepquantum.nn.modules.qugru import QuGRUCell


def qugru_cpu_test():
    dev = 'cpu'
    for n_qubits in range(5, 12):
        input_size, hidden_size = n_qubits, n_qubits
        start_time = time.perf_counter()
        x = torch.rand(3, 1, 2**n_qubits) + 0j
        cell = QuGRUCell(input_size, hidden_size, dev=dev)
        rst = cell(x, x)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qugru_cuda_test():
    dev = 'cuda'
    bs = 40
    for n_qubits in range(5, 12):
        input_size, hidden_size = n_qubits, n_qubits
        start_time = time.perf_counter()
        x = torch.rand(bs, 1, 2**n_qubits).cuda() + 0j
        cell = QuGRUCell(input_size, hidden_size, dev=dev)
        rst = cell(x, x)
        end_time = time.perf_counter()
        print((end_time - start_time) * 1000)


if __name__ == '__main__':
    # qugru_cpu_test()
    qugru_cuda_test()
