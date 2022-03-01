# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:30:23 2021

@author: shish
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time
import timeit


from deepquantum.gates.qmath import multi_kron, measure, IsUnitary, IsNormalized
import deepquantum.gates.qoperator as op
from deepquantum.gates.qcircuit import Circuit
from deepquantum.embeddings.qembedding import PauliEncoding
from deepquantum.layers.qlayers import YZYLayer, ZXLayer,ring_of_cnot, ring_of_cnot2, BasicEntangleLayer
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,MPS_expec,Rho2MPS,MPS2Rho, MatrixProductState, MatrixProductDensityOperator
from deepquantum.embeddings import PauliEncoding


if 0:
    N = 30    #量子线路的qubit总数
    wires_lst = list(range(N))
    
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    t1 = time.time()
    MPS = MatrixProductState(psi, N)
    c1 = Circuit(N)    
    c1.rx(0.215, N-1)
    
    MPS = c1.TN_contract_evolution(MPS)
    MPS = MPS.reshape(1,-1)
    print(time.time() - t1)
    print("===================12345================")
    # MPS = StateVec2MPS(psi,N)
    
    # for each in MPS:
    #     print(each.shape)
    
if 0:
    '''
    测试pauliencoding层加入TN功能后是否正确
    '''
    print('测试pauliencoding层加入TN功能后是否正确:')
    N = 5    #量子线路的qubit总数
    wires_lst = list(range(N))
    input_lst = [1,0.3,4.7]
    p1 = PauliEncoding(N,input_lst,wires_lst)
    
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    psif0 = (p1.U_expand() @ psi.view(-1,1)).view(1,-1)
    
    MPSf = p1.TN_operation( StateVec2MPS(psi,N) )
    psif1 = MPS2StateVec(MPSf)
    
    print(psif0)
    print(psif1)



#if __name__ == "__main__":
if 1:
    '''
    基于Tensor Network的量子线路态矢演化测试
    '''
    print('基于Tensor Network的量子线路态矢演化耗时测试:')    
    N = 10    #量子线路的qubit总数
    wires_lst = list(range(N))
    weight = torch.rand(21*N) * 2 * torch.pi
    
    c1 = Circuit(N)
    
    c1.YZYLayer(wires_lst, weight[0:3*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[3*N:6*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[6*N:9*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[9*N:12*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[12*N:15*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[15*N:18*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[18*N:21*N])
    
    c2 = Circuit(N)
    c2.BasicEntangleLayer(wires_lst, weight[0:18*N],repeat=6)
    c2.YZYLayer(wires_lst, weight[18*N:21*N])
    
    itern = 1
    T1 = 0.0;T2 = 0.0;T3 = 0.0;
    for ii in range(itern):
        
        psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
        # psi = torch.zeros(1,2**N)+0.0j
        # psi[0,0] = 1.0+0j;#psi[0,-1] = 1.0+0j
        # psi = nn.functional.normalize( psi,p=2,dim=1 )
        
        t1 = time.time()
        MPS0 = StateVec2MPS(psi, N)
        MPS_f = c1.TN_evolution(MPS0)
        psi_f0 = MPS2StateVec(MPS_f)
        t2 = time.time()
        T1 = T1 + (t2 - t1)
        
        t3 = time.time()
        psi_f1 = (c2.U() @ psi.view(-1,1) ).view(1,-1) 
        t4 = time.time()
        T2 = T2 + (t4 - t3)
        
        MPS0 = StateVec2MPS(psi, N)
        MPSc = MPS2StateVec( MPS0, return_sv=False )
        MPSc = c1.TN_contract_evolution(MPSc)
        psi_f2 = MPSc.reshape(1,-1)
        t5 = time.time()
        T3 = T3 + (t5 - t4)
    
    print(' U :',psi_f1)
    print('TN1:',psi_f0)
    print('TN2:',psi_f2)
    #TN的优势在比特数很小时并不明显，甚至更差
    print('比特数N：',N,'\n矩阵相乘耗时:',T2/itern,'\n张量网络耗时:',T1/itern,'\n ratio:',T2/T1)
    print('TN CONTRACT耗时:',T3/itern,'\n ratio:',T1/T3)
    #13qubit,优化矩阵后313s对8.5s，引入SWAP后TN：0.21s
    #12qubit时，65s对3s,优化矩阵后40s对2.2s，引入SWAP后40ss对0.16s
    #11qubit时，引入SWAP后5.3s对0.11s
    #10qubit是，1.5s对0.22s,优化矩阵后0.8s对0.2s,引入SWAP后0.8s对0.1s
    #14qubit:0.3s，15qubit:0.45s，16qubit:0.91s，18qubit:4.7s

if 0:
    print('开始密度矩阵MPDO相关测试：')
    N = 9    #超过9个电脑就卡死了。量子线路的qubit总数
    wires_lst = list(range(N))
    psi0 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;#psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    
    p = 0.45 + torch.tensor(0.0)
    rho0 = 1*p*( psi0.permute(1,0) @ psi0.conj() ) \
         + 1*(1-p)*( psi1.permute(1,0) @ psi1.conj() )
    #rho0 = (1.0/2**N)*torch.eye(2**N)+0j
    
    MPS = Rho2MPS(rho0,N)
    rho1 = MPS2Rho( MPS )
    print(torch.trace(rho0))
    print(torch.trace(rho1))
    print(rho0);print(rho1)
    
    print('密度矩阵MPDO与Rho相互转化耗时测试：')
    itern = 20
    T1 = 0.0;T2 = 0.0
    for i in range(itern):
        t1 = time.time()
        MPS = Rho2MPS(rho0,N)
        t2 = time.time()
        rho1 = MPS2Rho( MPS )
        t3 = time.time()
        T1 = T1 + (t2 - t1)
        T2 = T2 + (t3 - t2)
    print('to MPDO耗时：',T1/itern,';  to Rho耗时：',T2/itern)
    

    
if 1:
    print('测试所有门和layer对TN1和2的支持(正确性)：')
    N = 8
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    #psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    
    c1 = Circuit(N)
    for i in range(N):
        c1.Hadamard(i)
    c1.PauliX(3)
    c1.PauliY(5)
    c1.PauliZ(7)
    c1.rx(0.5, 0)
    c1.cz([2,1])
    c1.ry(0.1, 6)
    c1.toffoli([4,2,7])
    c1.rz(0.8, 4)
    c1.cnot([0,1])
    c1.toffoli([1,5,7])
    c1.SWAP([3,N-1])
    c1.cnot([N-1,0])
    c1.cnot([4,6])
    c1.cnot([7,2])
    c1.toffoli([0,1,2])
    # psif0 = (c1.U() @ psi.view(-1,1)).view(1,-1)
    # MPS = StateVec2MPS(psi, N)
    # MPS = c1.TN_evolution(MPS)
    # psif1 = MPS2StateVec(MPS).view(1,-1)
    # print(psif0)
    # print(psif1)
    
    N = 4
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    #psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    
    c1 = Circuit(N)
    for i in range(N):
        c1.Hadamard(i)
    c1.PauliX(0)
    c1.PauliY(1)
    c1.cphase(1.2, [2,0])
    c1.PauliZ(2);
    c1.rx(0.5, 0)
    c1.cz([2,1]);
    c1.ry(0.1, 2);
    c1.SWAP([1,2])
    c1.cu3([0.18,0.54,1.5], [0,1])
    c1.toffoli([1,2,0])
    c1.rz(0.8, 1)
    c1.cnot([0,1])
    c1.u1(0.1, 1)
    c1.cu3([0.18,0.54,1.5], [2,0])
    c1.cz([0,2])
    c1.toffoli([1,0,2])
    c1.cnot([2,0])
    c1.u1(0.381,2)
    c1.cnot([1,2])
    c1.cphase(0.49, [0,1])
    c1.u3([0.188,0,3.1], 0)
    c1.cnot([0,2])
    c1.toffoli([0,2,1])
    c1.ring_of_cnot(list(range(N)))
    c1.YZYLayer([0], [0.1,0.2,0.1])
    c1.XZXLayer([1], [0.1,0.2,0.1])
    c1.XYZLayer([2], [0.1,0.2,0.1])
    c1.ZYXLayer([0,N-1],[0.21,0.45,0.98,1,2,7])
    c1.cz([2,0])
    c1.u1(0.5, 2)
    c1.SWAP([0,1])
    c1.cphase(1.2, [1,0])
    c1.SWAP([0,2])
    c1.cz([0,1])
    c1.cnot([2,0])
    c1.SWAP([N-1,1])
    psif0 = (c1.U() @ psi.view(-1,1)).view(1,-1)
    
    MPS = StateVec2MPS(psi, N)
    MPS = c1.TN_evolution(MPS)
    psif1 = MPS2StateVec(MPS).view(1,-1)
    
    MPSc = MPS2StateVec( StateVec2MPS(psi, N), return_sv=False )
    MPSc = c1.TN_contract_evolution(MPSc)
    psif2 = MPSc.reshape(1,-1)
    print(psif0)
    print(psif1)
    print(psif2)
    
    print(torch.abs(psif2 - psif0) < 1e-6)


if 1:
    print('=======================================测试MPS收缩batch mod(正确性):')
    N = 4
    batch_size = 2
    params1 = torch.rand(3*N)*2*torch.pi
    params2 = torch.rand(2*N)*2*torch.pi
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    MPSc = MPS2StateVec( StateVec2MPS(psi, N), return_sv=False )
    MPSc_b = MPSc.unsqueeze(0)
    MPSc_b = torch.cat( tuple([MPSc_b]*batch_size),dim=0 )
    
    c1 = Circuit(N)
    wires = list( range(N) )
    for i in range(N):
        c1.Hadamard(i)
    c1.PauliX(0)
    c1.PauliY(1)
    c1.PauliZ(2)
    c1.u1(0.09, 1)
    c1.u1(0.12, 0)
    c1.rx(0.124, 2)
    c1.ry(5.048, 1)
    c1.rz(1.523, 3)
    c1.ring_of_cnot(wires)
    c1.u3([0.128,0.00,3.1], 0)
    c1.u3([0.972,0.54,2.1], 3)
    c1.rxx(0.65,[0,1])
    c1.ryy(0.15,[0,2])
    c1.rzz(3.02,[3,1])
    c1.cnot([0,1])
    c1.cnot([1,2])
    c1.cnot([2,3])
    c1.cnot([N-1,0])
    c1.cnot([3,1])
    c1.cz([0,3])
    c1.cz([3,1])
    c1.cz([0,1])
    c1.cphase(1.2, [2,0])
    c1.SWAP([0,2])
    c1.SWAP([1,3])
    c1.cu3([0.128,0.00,3.1],[3,1])
    c1.cu3([2.128,0.10,0.1],[2,3])
    c1.toffoli([1,0,2])
    c1.toffoli([3,2,0])
    c1.ZXLayer(wires,params2)
    c1.ring_of_cnot(wires)
    c1.XZLayer(wires,params2)
    c1.ring_of_cnot(wires)
    c1.YZYLayer(wires,params1)
    c1.ring_of_cnot(wires)
    c1.XYZLayer(wires,params1)
    c1.ring_of_cnot(wires)
    c1.XZXLayer(wires,params1)
    c1.ZYXLayer(wires,params1)
    
    MPSc = c1.TN_contract_evolution(MPSc)
    MPSc_b = c1.TN_contract_evolution(MPSc_b, batch_mod=True)
    psif1 = MPSc.reshape(1,-1)
    psif2 = MPSc_b.reshape(batch_size,1,-1)
    
    psif0 = (c1.U() @ psi.view(-1,1)).view(1,-1)
    
    print(psif0)
    print(psif1)
    print(torch.abs(psif0-psif1)<1e-6)
    print(psif2)
    print(torch.abs(psif0-psif2[0])<1e-6)

def create_circuit(N):
    assert N >= 4
    params1 = torch.rand(3*N)*2*torch.pi
    params2 = torch.rand(2*N)*2*torch.pi
    c1 = Circuit(N)
    wires = list( range(N) )
    for i in range(N):
        c1.Hadamard(i)
    c1.PauliX(0)
    c1.PauliY(1)
    c1.PauliZ(2)
    c1.u1(0.09, 1)
    c1.u1(0.12, 0)
    c1.rx(0.124, 2)
    c1.ry(5.048, 1)
    c1.cnot([N-1,0])
    c1.rz(1.523, 3)
    c1.ring_of_cnot(wires)
    c1.u3([0.128,0.00,3.1], 0)
    c1.u3([0.972,0.54,2.1], 3)
    c1.rxx(0.65,[0,1])
    c1.ryy(0.15,[0,2])
    c1.rzz(3.02,[3,1])
    c1.cnot([0,1])
    c1.cnot([1,2])
    c1.cnot([2,3])
    c1.cnot([3,1])
    c1.cz([0,3])
    c1.cz([3,1])
    c1.cz([0,1])
    c1.cphase(1.2, [2,0])
    c1.SWAP([0,2])
    c1.SWAP([1,3])
    c1.cu3([0.128,0.00,3.1],[3,1])
    c1.cu3([2.128,0.10,0.1],[2,3])
    c1.toffoli([1,0,2])
    c1.toffoli([3,2,0])
    c1.ZXLayer(wires,params2)
    c1.ring_of_cnot(wires)
    c1.XZLayer(wires,params2)
    c1.ring_of_cnot(wires)
    c1.YZYLayer(wires,params1)
    c1.ring_of_cnot(wires)
    c1.XYZLayer(wires,params1)
    c1.ring_of_cnot(wires)
    c1.XZXLayer(wires,params1)
    c1.toffoli([N-1,0,N-2])
    c1.ZYXLayer(wires,params1)
    return c1
    

def circuit_TN_contract_Rho_test():
    '''
    密度矩阵的张量网络演化测试函数 2022 02 21 ssy
    '''
    N = 6
    c1 = create_circuit(N)
    U = c1.U()
    
    psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    psi2 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    rho1 = psi1.permute(1,0) @ psi1.conj()
    rho2 = psi2.permute(1,0) @ psi2.conj()
    p = 0.59
    rho = p*rho1 + (1-p)*rho2
    assert torch.abs(torch.trace(rho) - 1) < 1e-6
    
    MPDO = MatrixProductDensityOperator(rho, N)
    rho_back = MPDO.reshape([2**N, 2**N])
    rst = torch.abs(rho_back - rho) < 1e-7
    rst = rst.reshape(-1)
    for each in rst:
        if each == False:
            raise ValueError("circuit_TN_contract_Rho_test失败，MatrixProductDensityOperator未通过测试")
    
    MPDO = c1.TN_contract_evolution(MPDO, batch_mod=False)
    rho_f = MPDO.reshape([2**N, 2**N])
    rho_f0 = U @ rho @ U.permute(1,0).conj()
    rst = torch.abs(rho_f - rho_f0) < 1e-6
    rst = rst.reshape(-1)
    
    for each in rst:
        if each == False:
            raise ValueError("circuit_TN_contract_Rho_test失败，无batch情况未通过测试")
            
    batch_size = 4
    psi1 = nn.functional.normalize( torch.rand(batch_size, 1,2**N)+torch.rand(batch_size, 1,2**N)*1j,p=2,dim=2 )
    psi2 = nn.functional.normalize( torch.rand(batch_size, 1,2**N)+torch.rand(batch_size, 1,2**N)*1j,p=2,dim=2 )
    rho1 = psi1.permute(0,2,1) @ psi1.conj()
    rho2 = psi2.permute(0,2,1) @ psi2.conj()
    p = 0.34
    rho = p*rho1 + (1-p)*rho2
    MPDO = MatrixProductDensityOperator(rho, N)
    MPDO = c1.TN_contract_evolution(MPDO, batch_mod=True)
    rho_f = MPDO.reshape([batch_size, 2**N, 2**N])
    rho_f0 = U @ rho @ U.permute(1,0).conj()
    rst = torch.abs(rho_f - rho_f0) < 1e-6
    # print(torch.max(torch.abs(rho_f - rho_f0)))
    rst = rst.reshape(-1)
    for each in rst:
        if each == False:
            raise ValueError("circuit_TN_contract_Rho_test失败，有batch情况未通过测试")
    print("-------------------------------------------------------------------")
    print("密度矩阵的张量网络演化circuit_TN_contract_Rho_test(有batch和无batch情况均)测试通过")
    print("-------------------------------------------------------------------")
    return True
    
circuit_TN_contract_Rho_test()
    
N = 6
c1 = create_circuit(N)
c1.rxx(0.124, [2,5])
t1 = timeit.default_timer()
c1.gate[0].U_expand()
t2 = timeit.default_timer()
print(t2-t1)
    
    
    


input('test_TN.py END')









        