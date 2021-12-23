# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 08:49:16 2021

@author: shish
"""
import torch
import torch.nn as nn
import random
import math
import time

from typing import List

def StateVec2MPS(psi:torch.Tensor, N:int, d:int=2)->List[torch.Tensor]:
    #t1 = time.time()
    #输入合法性检测：输入的态矢必须是1行2^N列的张量
    if len(psi.shape) != 2:
        raise ValueError('StateVec2MPS:input dimension error!')
    if psi.shape[0] != 1 or psi.shape[1] != 2**N:
        raise ValueError('StateVec2MPS:input shape must be 1 ROW 2^N COLUMN')
    
    c_tensor = psi + 0j
    rst_lst = []
    for i in range(N):
        #按照列(dim=1)把张量c_tensor对半分成2块(chunk=2)，再按照行叠加
        c_tensor_block = torch.chunk(c_tensor, chunks=2, dim=1)
        c_tensor = torch.cat((c_tensor_block[0], c_tensor_block[1]), dim=0)
        
        #最后一个qubit的张量无需SVD分解
        if i == N-1:
            rst_lst.append(c_tensor.view(2,-1,c_tensor.shape[1]))
            continue
        
        U,S,V = torch.svd( c_tensor )
        V_d = V.permute(1,0).conj()
        D = len( (S[torch.abs(S)>1e-8]).view(-1) ) #D：bond dimension
        #print(D)
        S = torch.diag(S) + 0j
        #根据bond dimension对张量进行缩减
        if D < S.shape[0]:
            U = torch.index_select(U, 1, torch.tensor(list(range(D))))
            S = torch.index_select(S, 0, torch.tensor(list(range(D))))
        
        rst_lst.append(U.view(2,-1,U.shape[1]))
        c_tensor = S @ V_d
        
    return rst_lst


def MPS2StateVec(tensor_lst:List[torch.Tensor])->torch.Tensor:
    #t1 = time.time()
    N = len(tensor_lst)
    for i in range(N):
        temp = tensor_lst[i].unsqueeze(0)
        if i == 0:
            c_tensor = tensor_lst[i]
        else:
            c_tensor = c_tensor.unsqueeze(1) @ temp
            shape = c_tensor.shape
            c_tensor = c_tensor.view(shape[0]*shape[1],shape[2],shape[3])
    c_tensor = c_tensor.view(-1).view(1,-1)
    #t2 = time.time()
    #print('MPS:',t2-t1)
    return c_tensor #返回1行2^N列的张量，表示态矢的系数


def TensorDecompAfterTwoQbitGate(tensor:torch.Tensor):
    block1 = torch.cat((tensor[0,0],tensor[0,1]),dim=1)
    block2 = torch.cat((tensor[1,0],tensor[1,1]),dim=1)
    tensor = torch.cat((block1,block2),dim=0)
    
    U,S,V = torch.svd( tensor )
    V_d = V.permute(1,0).conj()
    D = len( (S[torch.abs(S)>1e-8]).view(-1) ) #D：bond dimension
    S = torch.diag(S) + 0j
    #根据bond dimension对张量进行缩减
    if D < S.shape[0]:
        U = torch.index_select(U, 1, torch.tensor(list(range(D))))
        S = torch.index_select(S, 0, torch.tensor(list(range(D))))
    
    rst1 = (U.view(2,-1,U.shape[1]))
    
    rst2 = S @ V_d
    rst2_block = torch.chunk(rst2, chunks=2, dim=1)
    rst2 = torch.cat((rst2_block[0], rst2_block[1]), dim=0)
    rst2 = rst2.view(2,-1,rst2.shape[1])
    
    return rst1,rst2

def TensorDecompAfterThreeQbitGate(tensor:torch.Tensor):
    if len(tensor.shape) != 5:
        raise ValueError('TensorDecompAfterThreeQbitGate: tensor must be rank-5')
    blk0 = torch.cat((tensor[0,0,0],tensor[0,0,1],tensor[0,1,0],tensor[0,1,1]),dim=1)
    blk1 = torch.cat((tensor[1,0,0],tensor[1,0,1],tensor[1,1,0],tensor[1,1,1]),dim=1)
    tensor = torch.cat((blk0,blk1),dim=0)
    
    rst_lst = []
    for i in range(3):
        if i != 0:
            blks = torch.chunk(tensor, chunks=2, dim=1)
            tensor = torch.cat((blks[0], blks[1]), dim=0)
        if i != 2:
            U,S,V = torch.svd( tensor )
            V_d = V.permute(1,0).conj()
            D = len( (S[torch.abs(S)>1e-8]).view(-1) ) #D：bond dimension
            S = torch.diag(S) + 0j
            if D < S.shape[0]:
                U = torch.index_select(U, 1, torch.tensor( list(range(D)) ))
                S = torch.index_select(S, 0, torch.tensor( list(range(D)) ))
            rst_lst.append(U.view(2,-1,U.shape[1]))
            
            tensor = S @ V_d
        else:
            rst_lst.append(tensor.view(2,-1,tensor.shape[1]))
    return tuple(rst_lst)

def MPS_inner_product(ketMPS,braMPS):
    '''
    MPS做内积完全没必要，不如直接恢复成state vector再做内积
    '''
    #上限12个qubit
    N = len(ketMPS)
    lst = []
    for i in range(N):
        t1 = ketMPS[i]
        t2 = braMPS[i]
        t1 = t1.permute(1,2,0)
        t2 = t2.permute(1,2,0)
        
        for j in range(3):
            t1 = t1.unsqueeze(2*j+1)
            t2 = t2.unsqueeze(2*j)
        qbit_tensor = (t2.conj() @ t1).squeeze()
        lst.append(qbit_tensor)
    
    for i in range(len(lst)-1):
        if i == 0:
            t = lst[i]
            
        t_nxt = lst[i+1]
        if i != len(lst) - 2:
            t_nxt = t_nxt.permute(2,3,0,1)
        
        t = t.unsqueeze(-2).unsqueeze(-4) #1212
        t_nxt = t_nxt.unsqueeze(-1).unsqueeze(-3) #442121
        
        temp = t @ t_nxt #442211
        temp = temp.squeeze() #4422
        if i != len(lst) - 2:
            '''
            einsum高阶张量求迹，TZH我的超人！！！
            '''
            t = torch.einsum('abii->ab',temp)
            # for k in range(temp.shape[2]):
            #     if k == 0:
            #         t = temp[:,:,0,0]
            #     else:
            #         t += temp[:,:,k,k]
        else:
            t = torch.trace(temp)
    #返回的是内积，不是内积的模平方
    return t

def MPS_expec(MPS,wires:List[int],local_obserable:List[torch.Tensor]):
    '''
    local_obserable是一个包含局部力学量(针对单个qubit的力学量，一个2X2矩阵)的list
    wires也是一个list，表明每个局部力学量作用在哪个qubit上
    '''
    SV0 = MPS2StateVec(MPS).view(1,-1)
    
    for i,qbit in enumerate(wires):
        temp = MPS[qbit]
        temp = temp.permute(1,2,0).unsqueeze(-1) #2421
        temp = torch.squeeze(local_obserable[i] @ temp, dim=3) #242 在指定维度squeeze
        MPS[qbit] = temp.permute(2,0,1)
    expec = SV0.conj() @ MPS2StateVec(MPS).view(-1,1)
    return expec.squeeze().real
    
#============================================================================

def Rho2MPS(Rho:torch.Tensor,N:int)->List[torch.Tensor]:
    '''
    SVD参考：https://blog.csdn.net/u012968002/article/details/91354566
    参考：Distributed Matrix Product State Simulations 
    of Large-Scale Quantum Circuits 2.6节
    author:Aidan Dang
    注意，密度矩阵density matrix是厄米算符，可以看做一个力学量
    所以，对密度矩阵的分解，实际上是分解成一个MPO，矩阵乘积算符
    不过，为了以示区别，我们称其为MPDO，矩阵乘积密度算符
    '''
    if len(Rho.shape) != 2:
        raise ValueError('Rho2MPS : rho must be matrix(rank-2 tensor)')
    if Rho.shape[0] != 2**N or Rho.shape[1] != 2**N:
        raise ValueError('Rho2MPS : dimension of rho must be [2^N, 2^N]')
    MPS_lst = []
    r_tensor = Rho.view(1,-1)
    for i in range(N):
        #密度矩阵的reshape过程比态矢复杂得多
        r_tensor = torch.cat(torch.chunk(r_tensor, chunks=2, dim=1),dim=0)#先竖着切一刀，按照行堆叠
        r_tensor_lst = []
        for j in range(2):
            bias = int( j*int(r_tensor.shape[0]/2) )
            block_0 = r_tensor[bias:int(r_tensor.shape[0]/2)+bias]
            #blk_tuple = torch.chunk(block_0, chunks=int(2**(N-i)), dim=1)
            lst_even = []
            lst_odd = []
            for k,blk in enumerate(torch.chunk(block_0, chunks=int(2**(N-i)), dim=1)):
                if k%2 == 0:
                    lst_even.append(blk)
                else:
                    lst_odd.append(blk)
            # blk00 = torch.cat(tuple(lst_even),dim=1)
            # blk01 = torch.cat(tuple(lst_odd),dim=1)
            # r_tensor_lst.append( torch.cat((blk00,blk01),dim=0) )
            r_tensor_lst.append( torch.cat((torch.cat(tuple(lst_even),dim=1),
                                            torch.cat(tuple(lst_odd),dim=1)),
                                           dim=0) )
        r_tensor = torch.cat(tuple(r_tensor_lst),dim=0)
        #print('r_tensor:',r_tensor)
        #print('reshape: ',r_tensor.shape)
        '''
        NOTE:已知矩阵A做SVD分解得到U、S、V，若对10*A做SVD分解，将得到U,10*S,V
        所以，Rho的MPS和10*Rho的MPS是完全相同的
        '''
        U,S,V = torch.svd(r_tensor)
        #print(torch.max(S))
        V = V.permute(1,0).conj() #dag(V)
        D = len( (S[torch.abs(S)>1e-8]).view(-1) ) #D：bond dimension
        S = torch.diag(S) + 0j
        if D < S.shape[0]:
            U = torch.index_select(U, 1, torch.tensor(list(range(D))))
            S = torch.index_select(S, 0, torch.tensor(list(range(D))))
        r_tensor = S @ V #V实际是dag(V)
        '''
        NOTE：如果是纯态密度矩阵，最后一个r_tensor的值为1；
        如果是混态密度矩阵，最后一个r_tensor的值小于1；
        这也是为什么MPS2Rho，想恢复一个混态时，最后一步须对迹归一化；
        '''
        # if i == N-1:
        #     print('r_tensor:',r_tensor)
        MPS_lst.append( U.view(2,2,-1,U.shape[1]) )
    return MPS_lst



def MPS2Rho(MPS:List[torch.Tensor])->torch.Tensor:
    N = len(MPS)
    for i in range(N):
        t = MPS[i]
        if i == 0:
            rho = t
        else:
            rho = rho.unsqueeze(1).unsqueeze(3) @ t.unsqueeze(0).unsqueeze(2)
            s = rho.shape
            rho = rho.view(s[0]*s[1],s[2]*s[3],s[4],s[5])
    rho = rho.squeeze()
    '''
    混态密度矩阵，最后一个r_tensor的值小于1，该数值不能忽略，但MPS又不包含此信息；
    所以MPS2Rho，想恢复一个混态时，最后一步须对迹归一化；
    '''
    trace = torch.trace(rho).real
    return rho*(1.0/trace)

    

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
    N = 12 #19个就是上限了，20个比特我的电脑立刻死给你看
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