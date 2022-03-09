import numpy as np
# import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from deepquantum import Circuit
# from deepquantum.gates.qmath import dag,measure_state,ptrace,multi_kron,encoding,expecval_ZI,measure
from deepquantum.gates.qmath import dag, ptrace, multi_kron, encoding, measure, batched_kron2
from deepquantum.gates.qtensornetwork import MatrixProductDensityOperator


# 声明量子互学习操作的类
class Qu_mutual(nn.Module):
    def __init__(self, n_qubits,
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.n_qubits = n_qubits
        self.dim = (1 << self.n_qubits)
        self.sub_dim = (1 << (self.n_qubits // 2))
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)
        self.wires_lst = list(range(self.n_qubits))

    # 定义互学习操作函数，返回对应的操作门
    def qumutual(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        deep_size = 6
        cir.XYZLayer(self.wires_lst, w[:3*self.n_qubits])
        for which_q in range(0, self.n_qubits - 1):
            cir.cnot([which_q, which_q + 1])
        cir.cnot([self.n_qubits - 1, 0])
        cir.XYZLayer(self.wires_lst, w[:3*self.n_qubits])

        # U = cir.U()
        # return U
        return cir

    # 定义量子互学习的数据流，输出为两种信息交互后对应的两种信息
    def forward(self, inputA, inputB, dimA, dimB):
        # type: (Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor]

        cir = self.qumutual()
        if (inputA.ndim == 3) and (inputB.ndim == 3):
            assert (inputA.shape[1] == self.sub_dim) and (inputA.shape[2] == self.sub_dim)
            assert (inputB.shape[1] == self.sub_dim) and (inputB.shape[2] == self.sub_dim)
            assert inputA.shape[0] == inputB.shape[0]
            batch_size = inputA.shape[0]
            inputAB = batched_kron2(inputA, inputB)
            inputAB = MatrixProductDensityOperator(inputAB, self.n_qubits)
            inputAB = cir.TN_contract_evolution(inputAB, batch_mod=True)
            U_AB = inputAB.reshape(batch_size, self.dim, self.dim)
            mutualBatA = ptrace(U_AB, dimA, dimB)

            inputBA = batched_kron2(inputB, inputA)
            inputBA = MatrixProductDensityOperator(inputBA, self.n_qubits)
            inputBA = cir.TN_contract_evolution(inputBA, batch_mod=True)
            U_BA = inputBA.reshape(batch_size, self.dim, self.dim)
            mutualAatB = ptrace(U_BA, dimA, dimB)

        elif (inputA.ndim == 2) and (inputB.ndim == 2):
            assert (inputA.shape[0] == self.sub_dim) and (inputA.shape[1] == self.sub_dim)
            assert (inputB.shape[0] == self.sub_dim) and (inputB.shape[1] == self.sub_dim)
            inputAB = torch.kron(inputA, inputB)
            inputAB = MatrixProductDensityOperator(inputAB, self.n_qubits)
            inputAB = cir.TN_contract_evolution(inputAB, batch_mod=False)
            U_AB = inputAB.reshape(self.dim, self.dim)
            mutualBatA = ptrace(U_AB, dimA, dimB)

            inputBA = torch.kron(inputB, inputA)
            inputBA = MatrixProductDensityOperator(inputBA, self.n_qubits)
            inputBA = cir.TN_contract_evolution(inputBA, batch_mod=False)
            U_BA = inputBA.reshape(self.dim, self.dim)
            mutualAatB = ptrace(U_BA, dimB, dimA)
        else:
            raise ValueError("input x dimension error!")

        return mutualBatA, mutualAatB

# QU=Qu_mutual(4)
# scripted_module=torch.jit.script(QU)
# print(scripted_module.code)


if __name__ == '__main__':
    n_qubits = 8
    n_sub = n_qubits // 2
    inputA = torch.rand(2, 2**n_sub, 2**n_sub) + 0j
    inputB = torch.rand(2, 2**n_sub, 2**n_sub) + 0j
    dimA, dimB = n_sub, n_sub

    QU=Qu_mutual(n_qubits)
    rst = QU(inputA, inputB, dimA, dimB)
    print(rst[0].shape)


hyber_para = 16
dim_embed = hyber_para
#qubits数，进行信息交互/拼接时qubits*2
qubits_cirAorB = int(np.log2(hyber_para))
qubits_cirAandB = 2 * qubits_cirAorB
dim_FC = qubits_cirAandB


# 声明经典卷积和量子互学习的类
class Qu_conv_mutual(nn.Module):
    def __init__(self, embedding_num_drug, embedding_num_target,
                 hyber_para=hyber_para,
                 embedding_dim_drug=dim_embed, embedding_dim_target=dim_embed, conv1_out_dim=qubits_cirAorB):
        super().__init__()
        self.hyber_para = hyber_para
        self.qubits_cirAorB = conv1_out_dim
        self.dim_embed = embedding_dim_target

        self.embed_drug = nn.Embedding(embedding_num_drug, embedding_dim_drug, padding_idx=0)
        self.embed_target = nn.Embedding(embedding_num_target, embedding_dim_target, padding_idx=0)
        # 设置药物部分第一个卷积的参数（一维卷积）
        self.drugconv1 = nn.Conv1d(embedding_dim_drug, conv1_out_dim, kernel_size=4, stride=1, padding='same')
        # 设置第二个卷积的参数（二维卷积）
        self.drugconv2 = nn.Conv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # 设置蛋白质部分第一个卷积的参数（一维卷积）
        self.targetconv1 = nn.Conv1d(embedding_dim_target, conv1_out_dim, kernel_size=4, stride=1, padding='same')
        # 设置蛋白质部分第二个卷积的参数（二维卷积）
        self.targetconv2 = nn.Conv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1)
        # 设置量子信息交互参数
        self.mutual = Qu_mutual(qubits_cirAandB)
        # 设置全连接参数
        self.FC1 = nn.Linear(1 * 2 * 4 * qubits_cirAorB * qubits_cirAorB, 32)
        self.FC2 = nn.Linear(32, 1)

    # 定义数据流
    def forward(self, drug, target):
        # 进行embedding

        hyber_para = self.hyber_para
        qubits_cirAorB = self.qubits_cirAorB
        dim_embed = self.dim_embed

        d = self.embed_drug(drug)
        t = self.embed_target(target)
        # 生成半正定矩阵
        Gram_d = d.T @ d
        Gram_d = Gram_d.view(-1, hyber_para, hyber_para)
        # 进行药物部分的第一次卷积，输出为d1conv
        d1conv = (self.drugconv1(Gram_d)).view(qubits_cirAorB, dim_embed)
        # 将d1conv进行encoding编码成量子态，以便接下来进行信息交互
        d_mutual_input = encoding(d1conv.T @ d1conv)
        # 将d1conv进行转置相乘，得到输出d2_conv_input，以便之后与信息交互后的信息进行拼接后一起进行第二次卷积
        d2_conv_input = d1conv @ d1conv.T  # conv1_out_dim * conv1_out_dim
        # 针对蛋白质的操作与上面对药物分子的操作一样
        Gram_t = t.T @ t
        Gram_t = Gram_t.view(-1, hyber_para, hyber_para)
        t1conv = (self.targetconv1(Gram_t)).view(qubits_cirAorB, dim_embed)
        t_mutual_input = encoding(t1conv.T @ t1conv)
        t2_conv_input = t1conv @ t1conv.T
        # 药物和蛋白质信息交互后输出t1atd1, d1att1，对应药物分子部分和蛋白质分子部分
        t1atd1, d1att1 = self.mutual(d_mutual_input, t_mutual_input, qubits_cirAorB, qubits_cirAorB)
        # 分别进行测量（每个qubit得到一个结果）
        ###这里删掉了一个cir=Circuit(self.n_qubits)

        d_measure = measure(t1atd1, 4)
        t_measure = measure(d1att1, 4)

        d2_conv_input_m = d_measure @ d_measure.T
        t2_conv_input_m = t_measure @ t_measure.T
        d2_conv_input = d2_conv_input.view(1, qubits_cirAorB, qubits_cirAorB)
        d2_conv_input_m = d2_conv_input_m.view(1, qubits_cirAorB, qubits_cirAorB)
        # 将交互后流出的信息与各自的第一次经典卷积流出的信息进行拼接
        d2convinput = (torch.cat((d2_conv_input, d2_conv_input_m))).view(1, 2, qubits_cirAorB, qubits_cirAorB)
        # 合并后的信息进行第二次卷积（二维卷积），输出d2conv
        d2conv = self.drugconv2(d2convinput)
        # 针对蛋白质的操作与上面对药物分子的操作一样，输出t2conv
        t2_conv_input = t2_conv_input.view(1, qubits_cirAorB, qubits_cirAorB)
        t2_conv_input_m = t2_conv_input_m.view(1, qubits_cirAorB, qubits_cirAorB)
        t2convinput = (torch.cat((t2_conv_input, t2_conv_input_m))).view(1, 2, qubits_cirAorB, qubits_cirAorB)
        t2conv = self.targetconv2(t2convinput)
        # 上面药物和蛋白质的信息合并后输入经典全连接网络，得到结合能输出
        input_linear = (torch.cat([d2conv, t2conv], dim=0)).view(1, 1 * 2 * 4 * qubits_cirAorB * qubits_cirAorB)
        out = F.leaky_relu(self.FC1(input_linear))
        out = F.leaky_relu(self.FC2(out))
        out = out.view(1)
        return out

# QuCM=Qu_conv_mutual(64,25)
# scripted_module=torch.jit.script(QuCM)
# print(scripted_module.code)