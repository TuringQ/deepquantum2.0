# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 08:49:16 2021

@author: shish
"""
# import sys
# print(sys.path)

import torch
import torch.nn as nn
import random
import math
import time
from typing import List

from deepquantum.gates.qTN_contract import _StateVec2MPS, _MatrixProductState,\
    _MPS2StateVec, _TensorDecompAfterTwoQbitGate, _TensorDecompAfterThreeQbitGate,\
    _MPS_inner_product, _MPS_expec, _Rho2MPS, _MPS2Rho, _MatrixProductDensityOperator
# 若出现pyd文件无法import的情况，很可能是因为python版本不匹配
# 比如python3.9版本下用cython生成pyd，该pyd文件就依赖python39.dll
# 那么在低版本的python上就无法运行，因为找不到python39.dll
# 用visual studio的dumpbin工具分析可知该pyd文件依赖python39.dll，kernel32.dll
# VCRUNTIME140.dll，api-ms-win-crt-runtime-l1-1-0.dll这四个dll文件
# import deepquantum.gates.qTN_contract as q
# print(dir(q)) #查看导入的库中都有哪些api

def StateVec2MPS(psi:torch.Tensor, N:int, d:int=2)->List[torch.Tensor]:
    return _StateVec2MPS(psi, N, d=d)



def MatrixProductState(psi:torch.Tensor, N:int)->torch.Tensor:
    return _MatrixProductState(psi, N)



def MPS2StateVec(tensor_lst:List[torch.Tensor],return_sv=True)->torch.Tensor:
    return _MPS2StateVec(tensor_lst, return_sv=return_sv)
    


def TensorDecompAfterTwoQbitGate(tensor:torch.Tensor):
    return _TensorDecompAfterTwoQbitGate(tensor)
    

def TensorDecompAfterThreeQbitGate(tensor:torch.Tensor):
    return _TensorDecompAfterThreeQbitGate(tensor)
    

def MPS_inner_product(ketMPS,braMPS):
    '''
    MPS做内积完全没必要，不如直接恢复成state vector再做内积
    '''
    return _MPS_inner_product(ketMPS,braMPS)
   

def MPS_expec(MPS,wires:List[int],local_obserable:List[torch.Tensor]):
    return _MPS_expec(MPS, wires, local_obserable)
    
    
    
#============================================================================

def Rho2MPS(Rho:torch.Tensor,N:int)->List[torch.Tensor]:
    return _Rho2MPS(Rho, N)
    
    



def MPS2Rho(MPS:List[torch.Tensor])->torch.Tensor:
    return _MPS2Rho(MPS)
    




def MatrixProductDensityOperator(rho:torch.Tensor, N:int)->torch.Tensor:
    return _MatrixProductDensityOperator(rho, N)
    


  

# def TensorContraction(TensorA:torch.Tensor,TensorB:torch.Tensor,dimA:int,dimB:int):
#     '''
#     将TensorA的dimA维度与TensorB的dimB维度进行相乘收缩
#     '''
#     rankA = len(TensorA.shape)
#     rankB = len(TensorB.shape)
#     if dimA > rankA - 1 or dimB > rankB - 1:
#         raise ValueError('TensorContraction: dimA/dimB must less than rankA/rankB')
#     if TensorA.shape[dimA] != TensorB.shape[dimB]:
#         raise ValueError('TensorContraction: dimA&dimB not match')
    
#     permuteA = list(range(rankA)) 
#     permuteA.pop(dimA)
#     permuteA.append(dimA)
#     permuteA = tuple(permuteA)
    
#     permuteB = list(range(rankB))
#     permuteB.pop(dimB)
#     permuteB.append(dimB)
#     permuteB = tuple(permuteB)
    
#     TensorA = TensorA.permute(permuteA)
#     TensorB = TensorB.permute(permuteB)
#     pass



if __name__ == "__main__":
    '''
    12qubit时，SV2MPS平均要5ms，MPS2SV平均要3ms
    '''
    N = 8 #19个就是上限了，20个比特我的电脑立刻死给你看
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    
    #psi = nn.functional.normalize( torch.ones(1,2**N)+torch.rand(1,2**N)*0.0j,p=2,dim=1 )
    
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;#psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    '''
    验证StateVec2MPS和MPS2StateVec的正确性
    '''
    if 0:
        psi0 = psi
        MPS = StateVec2MPS(psi,N)
        psi1 = MPS2StateVec(MPS)
        print('psi0:',psi0)
        print('psi1:',psi1)
    
    '''
    统计StateVec2MPS和MPS2StateVec的平均耗时
    '''
    if 0:
        T1 = 0.0;T2 = 0.0
        itern = 20
        for i in range(itern):
            psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
            
            # psi = torch.zeros(1,2**N)+0.0j
            # psi[0,0] = 1.0+0j;psi[0,-1] = 1.0+0j
            # psi = nn.functional.normalize( psi,p=2,dim=1 )
            
            t1 = time.time()
            lst = StateVec2MPS(psi,N)
            t2 = time.time()
            psi1 = MPS2StateVec(lst)
            t3 = time.time()
            T1 += t2 - t1
            T2 += t3 - t2
        print('SV2MPS:',T1/itern,'    MPS2SV:',T2/itern)
    
    '''
    验证MPS_inner_product正确性
    '''
    if 1:
        print('验证MPS_inner_product正确性：')
        psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
        psi2 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
        print('态矢做内积的结果: ',( psi2.conj() @ psi1.permute(1,0) ).squeeze())
        lst1 = StateVec2MPS(psi1,N)
        lst2 = StateVec2MPS(psi2,N)
        t1 = time.time()
        rst = MPS_inner_product(lst1,lst2)
        t2 = time.time()
        print(t2-t1)
        print('tMPS做内积的结果: ',rst)
    
    input("END")