import torch
import time
from deepquantum.nn.modules.qureinforce import Policy


def qureinforce_cpu_test():
    n_layers = 4
    for n_qubits in range(5, 21):
        start_time = time.perf_counter()
        x = torch.rand(1, 2**n_qubits) + 0j
        cir = Policy(n_qubits, n_layers,  'z0,y1,x2', 6, dev='cpu')
        rst = cir(x)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qureinforce_cuda_test():
    n_layers = 4
    for n_qubits in range(5, 21):
        start_time = time.perf_counter()
        x = torch.rand(1, 2**n_qubits).cuda() + 0j
        cir = Policy(n_qubits, n_layers,  'z0,y1,x2', 6, dev='cuda')
        rst = cir(x)
        end_time = time.perf_counter()
        print((end_time - start_time) * 1000)


if __name__ == '__main__':
    # qureinforce_cpu_test()
    qureinforce_cuda_test()
