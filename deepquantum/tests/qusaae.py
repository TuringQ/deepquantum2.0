import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
#from deepquantum import Circuit
from deepquantum.gates.qcircuit import Circuit
import deepquantum.gates.qoperator as op
from deepquantum.gates.qmath import partial_trace, partial_trace_batched, partial_trace_batched2, batched_kron, batched_kron2
# from deepquantum.utils import dag,measure_state,ptrace,multi_kron,encoding,expecval_ZI,measure

'''
qusaae = quantum SAAE = quantum supervised adversarial auto-encoder = 有监督对抗量子自编码器
'''
# 量子线路模块（Encoder,Decoder,Discriminator):
class QuEn(nn.Module):
    """
    根据量子线路图摆放旋转门以及受控门
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_qubits = n_qubits

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(15 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)

    def forward(self, x):
        #输入合法性检测
        if x.ndim == 2:
            #输入的x是 1 X (2**N) 维度的态矢
            assert x.shape[0] == 1 and x.shape[1] == 2**(self.n_qubits)
            is_batch = False
            x = x.view([2]*self.n_qubits)
        elif x.ndim == 3:
            #输入的x是 batch_size X 1 X (2**N) 维度的批量态矢
            assert x.shape[1] == 1 and x.shape[2] == 2**(self.n_qubits)
            is_batch = True
            x = x.view([ x.shape[0] ]+[2]*self.n_qubits)
        else:
            #输入x格式有问题，发起报错
            raise ValueError("input x dimension error!")

        
        w = self.weight * self.w_mul
        wires_lst = list( range(self.n_qubits) )
        cir = Circuit(self.n_qubits)
        
        cir.XYZLayer(wires_lst, w[0:3*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[3*self.n_qubits:6*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[6*self.n_qubits:9*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[9*self.n_qubits:12*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[12*self.n_qubits:15*self.n_qubits])
        # U = cir.U()
        # x = (U @ x.permute(0,2,1)).permute(0,2,1)
        # return x
        x = cir.TN_contract_evolution(x, batch_mod=is_batch)
        
        if is_batch == True:
            x = x.view([ x.shape[0], 1, 2**self.n_qubits ])
        else:
            x = x.view([ 1, 2**self.n_qubits ])
        
        return x #返回编码后的态矢（或者是batched态矢）
        
         

class Q_Encoder(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()

        # 6比特的编码线路
        self.n_qubits = n_qubits

        self.encoder = QuEn(n_qubits)

    def forward(self, molecular, dimA):
        ts = time.time()
        #很不幸，求完偏迹后纯态变混态，以后不得不用混态密度矩阵来进行解码器的量子线路运算
        x = self.encoder(molecular)
        x_sv = torch.clone(x)
        dimB = self.n_qubits - dimA #dimB代表要丢弃的几个qubit
        trace_lst = list( range(dimB) )
        
        if x.ndim == 3:
            rho_batch = x.permute(0,2,1) @ x.conj() #得到一个batched密度矩阵
            t1 = time.time()
            encoder_rst = partial_trace_batched(rho_batch, self.n_qubits, trace_lst)
            print(time.time() - t1)
        elif x.ndim == 2:
            rho = x.permute(1,0) @ x.conj()
            encoder_rst = partial_trace(rho, self.n_qubits, trace_lst)
        else:
            raise ValueError("encoder x dimension error")
        print("enc:",time.time() - ts)  
        return encoder_rst, x_sv #编码器既返回偏迹后的约化密度矩阵，也返回态矢

#======================================================================================================================================
class QuDe(nn.Module):
    """
    根据量子线路图摆放旋转门以及受控门
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_qubits = n_qubits

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(15 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)

    

    def forward(self, x):
        w = self.weight * self.w_mul
        wires_lst = list( range(self.n_qubits) )
        cir = Circuit(self.n_qubits)
        cir.XYZLayer(wires_lst, w[0:3*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[3*self.n_qubits:6*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[6*self.n_qubits:9*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[9*self.n_qubits:12*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[12*self.n_qubits:15*self.n_qubits])
        U = cir.U()
        decoder_rst = U @ x @ U.permute(1,0).conj()
        return decoder_rst
        
        


class Q_Decoder(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()

        # 10比特量子解码器
        self.n_qubits = n_qubits
        self.decoder = QuDe(n_qubits)

    def forward(self, molecular, gene, dimA):
        ts = time.time()
        # type: (torch.Tensor, torch.Tensor, int) ->torch.Tensor
        m = molecular
        g = gene
        
        if m.ndim == 3 and g.ndim == 3:
            t1 = time.time()
            # rho1 = batched_kron(m, g)
            rho = batched_kron2(m, g)
            # print(torch.abs(rho[4] - rho1[4])<1e-6)
            print(time.time() - t1)
        elif m.ndim == 2 and g.ndim == 2:
            rho = torch.kron(m, g)   
        else:
            raise ValueError("m,g dimension error")
        t1 = time.time()
        rho = self.decoder(rho)
        print(time.time() - t1)
        
        dimB = self.n_qubits - dimA
        trace_lst = list(range(dimB))
        
        if rho.ndim == 3:
            decoder_rst = partial_trace_batched(rho, self.n_qubits, trace_lst)
        elif rho.ndim == 2:
            decoder_rst = partial_trace(rho, self.n_qubits, trace_lst)
        else:
            raise ValueError("decoder rho dimension error")
        print("dec:",time.time() - ts)
        return decoder_rst

#==========================================================================================================================
class QuDis(nn.Module):
    """
    根据量子线路图摆放旋转门以及受控门
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_qubits = n_qubits

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(12 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)

    

    def forward(self, x):
        #输入合法性检测
        if x.ndim == 2:
            #输入的x是 1 X (2**N) 维度的态矢
            assert x.shape[0] == 1 and x.shape[1] == 2**(self.n_qubits)
            is_batch = False
            x = x.view([2]*self.n_qubits)
        elif x.ndim == 3:
            #输入的x是 batch_size X 1 X (2**N) 维度的批量态矢
            assert x.shape[1] == 1 and x.shape[2] == 2**(self.n_qubits)
            is_batch = True
            x = x.view([ x.shape[0] ]+[2]*self.n_qubits)
        else:
            #输入x格式有问题，发起报错
            raise ValueError("input x dimension error!")
        
        w = self.weight * self.w_mul
        wires_lst = list( range(self.n_qubits) )
        cir = Circuit(self.n_qubits)
        cir.XYZLayer(wires_lst, w[0:3*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[3*self.n_qubits:6*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[6*self.n_qubits:9*self.n_qubits])
        cir.ring_of_cnot(wires_lst)
        cir.XYZLayer(wires_lst, w[9*self.n_qubits:12*self.n_qubits])
        
        x = cir.TN_contract_evolution(x, batch_mod=is_batch)
        
        
        x0 = torch.clone(x)
        x = op.PauliZ(self.n_qubits, 0).TN_contract(x, batch_mod=is_batch)
        s = x.shape
        if is_batch == True:
            x = x.reshape(s[0],-1,1)
            x0 = x0.reshape(s[0],1,-1)
        else:
            x = x.reshape(-1,1)
            x0 = x0.reshape(1,-1)
            
        rst = (x0 @ x).real
        rst = rst.squeeze(-1)
        return rst
        


class Q_Discriminator(nn.Module):
    def __init__(self, n_qubit):
        super().__init__()

        # 6比特量子判别器
        self.n_qubit = n_qubit
        self.discriminator = QuDis(self.n_qubit)

    def forward(self, molecular):
        t1 = time.time()
        rst = self.discriminator(molecular)
        print("dis:",time.time() - t1)
        return rst


# QE=Q_Encoder(6)
# scripted_qe=torch.jit.script(QE)
# print(scripted_qe.code)
#
# QD=Q_Decoder(4)
# scripted_de =torch.jit.script(QD)
# print(scripted_de.code)
#
# QDis=Q_Discriminator(10)
# scripted_dis=torch.jit.script(QDis)
# print(scripted_dis.code)

if __name__ == '__main__':
    
    
    N = 10; dimA = 5
    dimg = 6
    encod = Q_Encoder(N)
    decod = Q_Decoder(dimA+dimg)
    discr = Q_Discriminator(N)
    
    batchsize = 64
    m = nn.functional.normalize( torch.rand(batchsize,1,2**N)+torch.rand(batchsize,1,2**N)*1j,p=2,dim=2 )
    g = nn.functional.normalize( torch.rand(batchsize,1,2**dimg)+torch.rand(batchsize,1,2**dimg)*1j,p=2,dim=2 )
    # g_rho = g.permute(1,0) @ g.conj()
    g_rho = g.permute(0,2,1) @ g.conj()
    
    t1 = time.time()
    en_out_rho, en_out_sv = encod(m , dimA)
    # print(en_out_rho.shape)
    dimA = N
    decod(en_out_rho, g_rho, dimA)
    
    discr(m)
    discr(en_out_sv)
    
    t2 = time.time()
    print("forward耗时：", t2 - t1)
    # for i in range(5):
    #     t1 = time.time()
    #     dimA = 4
    #     en_out_rho, en_out_sv = encod(m , dimA)
    #     # print(en_out_rho.shape)
    #     dimA = 6
    #     decod(en_out_rho, g_rho, dimA)
        
    #     discr(m)
    #     discr(en_out_sv)
        
    #     t2 = time.time()
    #     print("forward耗时：", t2 - t1)
    input("END!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
