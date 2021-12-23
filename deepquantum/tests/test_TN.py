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


from deepquantum.gates.qmath import multi_kron, measure, IsUnitary, IsNormalized
import deepquantum.gates.qoperator as op
from deepquantum.gates.qcircuit import Circuit
from deepquantum.embeddings.qembedding import PauliEncoding
from deepquantum.layers.qlayers import YZYLayer, ZXLayer,ring_of_cnot, ring_of_cnot2, BasicEntangleLayer
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,MPS_expec,Rho2MPS,MPS2Rho
from deepquantum.embeddings import PauliEncoding


if 0:
    N = 4    #量子线路的qubit总数
    wires_lst = list(range(N))
    
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    MPS = StateVec2MPS(psi,N)
    for each in MPS:
        print(each.shape)
    
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
    print('基于Tensor Network的量子线路态矢演化测试:')    
    N = 10    #量子线路的qubit总数
    wires_lst = list(range(N))
    weight = torch.rand(21*N) * 2 * torch.pi
    
    c1 = Circuit(N)
    #c1.rx(0.5,1)
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
    
    itern = 10
    T1 = 0.0;T2 = 0.0
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
    
    print(psi_f0)
    print(psi_f1)
    #TN的优势在比特数很小时并不明显，甚至更差
    print('比特数N：',N,' 矩阵相乘耗时:',T2/itern,' 张量网络耗时:',T1/itern,'  ratio:',T2/T1)
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
    print('测试单比特门+cnot门+toffoli等所有门和layer对TN的支持：')
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
    
    N = 3
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    #psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    
    c1 = Circuit(N)
    for i in range(N):
        c1.Hadamard(i)
    c1.PauliX(0);c1.PauliY(1)
    c1.cphase(0.9, [2,0]);c1.PauliZ(2);c1.rx(0.5, 0)
    c1.cz([2,1]);c1.ry(0.1, 2);c1.SWAP([1,2])
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
    c1.cz([2,0]);c1.u1(0.5, 2)
    c1.SWAP([0,1])
    c1.cphase(1.2, [1,0])
    c1.SWAP([0,2])
    c1.cz([0,1])
    psif0 = (c1.U() @ psi.view(-1,1)).view(1,-1)
    
    MPS = StateVec2MPS(psi, N)
    MPS = c1.TN_evolution(MPS)
    psif1 = MPS2StateVec(MPS).view(1,-1)
    print(psif0)
    print(psif1)

if 0:
    print('测试单比特门+cnot门+toffoli等门TN运行耗时：')
    N = 10
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
    c1.toffoli([1,5,9])
    c1.SWAP([3,N-1])
    c1.cnot([N-1,0])
    c1.cnot([4,6])
    c1.cnot([7,2])
    c1.toffoli([0,1,N-1])
    c1.cz([0,1])
    
    itern = 10
    T = 0.0
    for i in range(itern):
        t1 = time.time()
        MPS = StateVec2MPS(psi, N)
        MPS = c1.TN_evolution(MPS)
        psif1 = MPS2StateVec(MPS).view(1,-1)
        t2 = time.time()
        T += t2 - t1
    
    print(psif1)
    print('TN平均耗时：',T/itern)
    
input('test_TN.py END')









        