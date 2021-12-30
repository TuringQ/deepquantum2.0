# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:30:02 2021

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
from deepquantum.layers.qlayers import XYZLayer,ZYXLayer,YZYLayer, XZXLayer,ZXLayer,\
    ring_of_cnot, ring_of_cnot2, BasicEntangleLayer
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,MPS_expec,Rho2MPS,MPS2Rho
from deepquantum.embeddings import PauliEncoding

torch.set_printoptions(precision=20)
if 0:
    '''
    测试各个门、layer的operation_dagger()是否真的产生了U_dagger
    U_dagger @ U = 单位阵 
    '''
    N = 3
    params_lst1 = torch.rand(3*N)*2*torch.pi
    params_lst2 = torch.rand(2*N)*2*torch.pi
    wires = list(range(N))
    
    c1 = Circuit(N)
    c1.ring_of_cnot(wires)
    c1.XYZLayer(wires, params_lst1)
    c1.YZYLayer(wires, params_lst1)
    c1.XZXLayer(wires, params_lst1)
    c1.ZXLayer(wires, params_lst2)
    c1.XZLayer(wires, params_lst2)
    c1.Hadamard(wires)
    for idx, each in enumerate( c1.gate ):
        temp = each.operation_dagger().U_expand() @ each.U_expand()
        print(idx,'  ',temp.diag())
    print(c1.gate[1].params)
    print(c1.gate[1].operation_dagger().params)
    print('==================================================================')    
    c2 = Circuit(N)
    c2.Hadamard(1)
    c2.PauliX(0)
    c2.PauliY(1)
    c2.PauliZ(N-1)
    c2.rx(0.236, 0)
    c2.ry(6.263, 1)
    c2.rz(1.259, N-1)
    c2.u1(0.98, 1)
    c2.u3([0.54,2.14,0.95], 0)
    c2.rxx(0.48, [0,1])
    c2.ryy(5.48, [0,1])
    c2.ryy(2.50, [0,1])
    c2.cnot([1,0])
    c2.cz([N-1,0])
    c2.cphase(0.52, [0,1])
    c2.cu3([0.21,0.84,2.1], [1,0])
    for idx,each in enumerate( c2.gate ):
        temp = each.operation_dagger().U_expand() @ each.U_expand()
        print(idx,'  ',temp.diag())


if 1:
    '''
    测试各个门、layer的operation_dagger()产生的新实例的TN_operation是否有效
    '''
    N = 3
    params_lst1 = torch.rand(3*N)*2*torch.pi
    params_lst2 = torch.rand(2*N)*2*torch.pi
    wires = list(range(N))
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    MPS = StateVec2MPS(psi,N)
    
    c1 = Circuit(N)
    c1.ring_of_cnot(wires)
    c1.XYZLayer(wires, params_lst1)
    c1.YZYLayer(wires, params_lst1)
    c1.XZXLayer(wires, params_lst1)
    c1.ZXLayer(wires, params_lst2)
    c1.XZLayer(wires, params_lst2)
    c1.Hadamard(wires)
    for idx, each in enumerate( c1.gate ):
        MPS = each.TN_operation(MPS)
        MPS = each.operation_dagger().TN_operation(MPS)
        psif = MPS2StateVec(MPS)
        temp = (psif.conj() @ psi.view(-1,1) ).squeeze()
        print(idx,'  ',temp)
    # print(c1.gate[1].params)
    # print(c1.gate[1].operation_dagger().params)
    print('==================================================================')    
    c2 = Circuit(N)
    c2.Hadamard(1)
    c2.PauliX(0)
    c2.PauliY(1)
    c2.PauliZ(N-1)
    c2.rx(0.236, 0)
    c2.ry(6.263, 1)
    c2.rz(1.259, N-1)
    c2.u1(0.98, 1)
    c2.u3([0.54,2.14,0.95], 0)
    c2.rxx(0.48, [0,1])
    c2.ryy(5.48, [0,1])
    c2.ryy(2.50, [0,1])
    c2.cnot([1,0])
    c2.cz([N-1,0])
    c2.cphase(0.52, [0,1])
    c2.cu3([0.21,0.84,2.1], [1,0])
    for idx, each in enumerate( c2.gate ):
        if each.supportTN==True:
            MPS = each.TN_operation(MPS)
            MPS = each.operation_dagger().TN_operation(MPS)
            psif = MPS2StateVec(MPS)
            temp = (psif.conj() @ psi.view(-1,1) ).squeeze()
            print(idx,'  ',temp)

input("END")