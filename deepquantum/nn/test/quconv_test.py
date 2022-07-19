import torch
import time
from deepquantum.nn.modules.quconv import QuConVXYZ


def quconv_sv_cpu_test():
    dev = 'cpu'
    dimA = 2
    bs = 32
    for n_qubits in range(2, 14, 2):
        start_time = time.perf_counter()
        x = torch.rand((bs, 1, 2**n_qubits)) + 0j
        quconv = QuConVXYZ(n_qubits, dev=dev)
        rst = quconv(x, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def quconv_sv_cuda_test():
    dev = 'cuda'
    dimA = 2
    bs = 32
    for n_qubits in range(2, 14, 2):
        start_time = time.perf_counter()
        x = torch.rand((bs, 1, 2**n_qubits)).cuda() + 0j
        quconv = QuConVXYZ(n_qubits, dev=dev)
        rst = quconv(x, dimA)
        end_time = time.perf_counter()
        print((end_time - start_time) * 1000)


def quconv_dm_cpu_test():
    dev = 'cpu'
    dimA = 4
    for n_qubits in range(4, 12, 2):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)) + 0j
        data = x.permute(0, 2, 1) @ x.conj()
        quconv = QuConVXYZ(n_qubits, dev=dev)
        rst = quconv(data, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def quconv_dm_cuda_test():
    dev = 'cuda'
    dimA = 4
    for n_qubits in range(4, 12, 2):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)).cuda() + 0j
        data = x.permute(0, 2, 1) @ x.conj()
        quconv = QuConVXYZ(n_qubits, dev=dev)
        rst = quconv(data, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')



if __name__ == '__main__':
    # quconv_sv_cpu_test()
    quconv_sv_cuda_test()

    # quconv_dm_cpu_test()
    # quconv_dm_cuda_test()
