import torch
import time
from deepquantum.nn.modules.qulinear_anlge_embedding import QuLinearAngleEmbedding


def qulinear_cpu_test():
    dev = 'cpu'
    n_layers = 3
    for n_qubits in range(2, 12):
        start_time = time.perf_counter()
        x = torch.rand((32, 1, 2**n_qubits)) + 0j
        qulinear = QuLinearAngleEmbedding(n_qubits, n_layers, 'Z', dev=dev)
        res = qulinear(x, 2)      # 输出维度2**n_out, n_out <= n_qubits
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')
    # print(f'线性层结果维度: {res.shape}')
    # print(f'|---> res.device: {res.device}')


def qulinear_cuda_test():
    dev = 'cuda'
    n_layers = 3
    for n_qubits in range(2, 12):
        start_time = time.perf_counter()
        x = torch.rand((32, 1, 2**n_qubits)).cuda() + 0j
        qulinear = QuLinearAngleEmbedding(n_qubits, n_layers, 'Z', dev=dev)
        res = qulinear(x, 2)      # 输出维度2**n_out, n_out <= n_qubits
        end_time = time.perf_counter()
        print(f'{(end_time - start_time) * 1000}')
    # print(f'线性层结果维度: {res.shape}')
    # print(f'|---> res.device: {res.device}')



if __name__ == '__main__':
    # qulinear_cpu_test()
    qulinear_cuda_test()
