import torch
import time
from deepquantum.nn.modules.qupooling import QuPoolZYX, QuPoolSX


def qupoolzyx_sv_cpu_test():
    dev = 'cpu'
    dimA = 2
    bs = 64
    for n_qubits in range(2, 12):
        start_time = time.perf_counter()
        x = torch.rand((bs, 1, 2**n_qubits)) + 0j
        qupoolzyx = QuPoolZYX(n_qubits, dev=dev)
        rst = qupoolzyx(x, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qupoolzyx_sv_cuda_test():
    dev = 'cuda'
    dimA = 2
    bs = 64
    for n_qubits in range(2, 12):
        start_time = time.perf_counter()
        x = torch.rand((bs, 1, 2 ** n_qubits)).cuda() + 0j
        qupoolzyx = QuPoolZYX(n_qubits, dev=dev)
        rst = qupoolzyx(x, dimA)
        end_time = time.perf_counter()
        print((end_time - start_time) * 1000)


def qupoolzyx_dm_cpu_test():
    dev = 'cpu'
    dimA = 4
    for n_qubits in range(5, 12):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)) + 0j
        data = x.permute(0, 2, 1) @ x.conj()
        qupoolzyx = QuPoolZYX(n_qubits, dev=dev)
        rst = qupoolzyx(data, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qupoolzyx_dm_cuda_test():
    dev = 'cuda'
    dimA = 4
    for n_qubits in range(5, 11):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)).cuda() + 0j
        data = x.permute(0, 2, 1) @ x.conj()
        qupoolzyx = QuPoolZYX(n_qubits, dev=dev)
        rst = qupoolzyx(data, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qupoolsx_sv_cpu_test():
    dev = 'cpu'
    dimA = 4
    for n_qubits in range(5, 14):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)) + 0j
        qupoolsx = QuPoolSX(n_qubits, dev=dev)
        rst = qupoolsx(x, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qupoolsx_sv_cuda_test():
    dev = 'cuda'
    dimA = 4
    for n_qubits in range(5, 14):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)).cuda() + 0j
        qupoolsx = QuPoolSX(n_qubits, dev=dev)
        rst = qupoolsx(x, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qupoolsx_dm_cpu_test():
    dev = 'cpu'
    dimA = 4
    for n_qubits in range(5, 12):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)) + 0j
        data = x.permute(0, 2, 1) @ x.conj()
        qupoolsx = QuPoolSX(n_qubits, dev=dev)
        rst = qupoolsx(data, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


def qupoolsx_dm_cuda_test():
    dev = 'cuda'
    dimA = 4
    for n_qubits in range(5, 11):
        start_time = time.perf_counter()
        x = torch.rand((3, 1, 2**n_qubits)).cuda() + 0j
        data = x.permute(0, 2, 1) @ x.conj()
        qupoolsx = QuPoolSX(n_qubits, dev=dev)
        rst = qupoolsx(data, dimA)
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')


if __name__ == '__main__':
    # qupoolzyx_sv_cpu_test()
    qupoolzyx_sv_cuda_test()
    # qupoolzyx_dm_cpu_test()
    # qupoolzyx_dm_cuda_test()
    # qupoolsx_sv_cpu_test()
    # qupoolsx_sv_cuda_test()
    # qupoolsx_dm_cpu_test()
    # qupoolsx_dm_cuda_test()
