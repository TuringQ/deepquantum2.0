# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:06:10 2021

@author: shish
"""
import torch
import torch.nn as nn
from deepquantum.gates import multi_kron
from deepquantum.gates.qoperator import Hadamard,rx,ry,rz,rxx,ryy,rzz,cnot,cz,SWAP,Operation
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,TensorDecompAfterTwoQbitGate
import time
from typing import List
import copy
#import multiprocessing as mp

class SingleGateLayer(Operation):
    '''
    单比特层的父类
    '''
    def __init__(self):
        self.nqubits = 4
        self.wires = []
        self.supportTN = True
    
    def _cal_single_gates(self)->List[torch.Tensor]:
        lst1 = [torch.eye(2,2)]*self.nqubits
        return lst1
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        lst1 = self._cal_single_gates()
        for qbit in self.wires:
            
            # temp = MPS[qbit]
            # temp = temp.permute(1,2,0).unsqueeze(-1) #2421
            # temp = torch.squeeze(lst1[qbit] @ temp, dim=3) #242 在指定维度squeeze
            # MPS[qbit] = temp.permute(2,0,1)
            
            MPS[qbit] = torch.einsum('ab,bcd->acd',[lst1[qbit],MPS[qbit]])

        return MPS
    
    #def TN_operation_mp(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        # lst1 = self._cal_single_gates()
        # pool = mp.Pool(processes=8)
        # p_lst = []
        # for qbit in self.wires:
            
        #     #MPS[qbit] = torch.einsum('ab,bcd->acd',[lst1[qbit],MPS[qbit]])
        #     p_lst.append( pool.apply_async(torch.einsum,args=('ab,bcd->acd',[lst1[qbit],MPS[qbit]])) )
        #     #p = mp.Process(target=torch.einsum,args=('ab,bcd->acd',[lst1[qbit],MPS[qbit]]))
        # pool.close()
        # pool.join()
        # for i,each in enumerate( p_lst ):
        #     qbit = self.wires[i]
        #     MPS[qbit] = each.get()
        
        # p_lst = []
        # for qbit in self.wires:
        #     p = mp.Process(target=torch.einsum,args=('ab,bcd->acd',[lst1[qbit],MPS[qbit]]))
        #     p_lst.append(p)
        #     p.start()
        # for p in p_lst:
        #     p.join()
        # for i,p in enumerate(p_lst):
        #     qbit = self.wires[i]
        #     MPS[qbit] = p.get()
        #     p.close()
        # return MPS
    
    

class TwoQbitGateLayer(Operation):
    '''
    两比特层的父类
    '''
    def __init__(self):
        pass

    
    

#===============================================================================

class XYZLayer(SingleGateLayer):
    #label = "XYZLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("XYZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XYZLayer: number of wires must less than N")
        self.label = "XYZLayer"
        
        self.nqubits = N
        self.wires = wires
        #如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True
    
    def _cal_single_gates(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            xm = rx(self.params[3*i+0]).matrix
            ym = ry(self.params[3*i+1]).matrix
            zm = rz(self.params[3*i+2]).matrix
            
            lst1[qbit] = zm @ ym @ xm
        return lst1
    
    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j
    
    def operation_dagger(self):
        params_tensor = -1*self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy( params_tensor[3*i+0] )
            params_tensor[3*i+0] = params_tensor[3*i+2]
            params_tensor[3*i+2] = temp
        return ZYXLayer(self.nqubits,self.wires,params_tensor)
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)
    

class ZYXLayer(SingleGateLayer):
    #label = "ZYXLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("ZYXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("ZYXLayer: number of wires must less than N")
        self.label = "ZYXLayer"
        
        self.nqubits = N
        self.wires = wires
        #如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True
    
    def _cal_single_gates(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            zm = rz(self.params[3*i+0]).matrix
            ym = ry(self.params[3*i+1]).matrix
            xm = rx(self.params[3*i+2]).matrix
            
            lst1[qbit] = xm @ ym @ zm
        return lst1
    
    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j
    
    def operation_dagger(self):
        params_tensor = -1*self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy( params_tensor[3*i+0] )
            params_tensor[3*i+0] = params_tensor[3*i+2]
            params_tensor[3*i+2] = temp
        return XYZLayer(self.nqubits,self.wires,params_tensor)
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)





class YZYLayer(SingleGateLayer):
    # label = "YZYLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("YZYLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("YZYLayer: number of wires must less than N")
        self.label = "YZYLayer"
        
        self.nqubits = N
        self.wires = wires
        #如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True
        
    def _cal_single_gates(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            y1m = ry(self.params[3*i+0]).matrix
            zm = rz(self.params[3*i+1]).matrix
            y2m = ry(self.params[3*i+2]).matrix
            
            lst1[qbit] = y2m @ zm @ y1m
        return lst1
    
    
    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j
    
    def operation_dagger(self):
        params_tensor = -self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy( params_tensor[3*i+0] )
            params_tensor[3*i+0] = params_tensor[3*i+2]
            params_tensor[3*i+2] = temp
        return YZYLayer(self.nqubits,self.wires,params_tensor)
    
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)










class XZXLayer(SingleGateLayer):
    # label = "XZXLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("XZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZXLayer: number of wires must less than N")
        self.label = "XZXLayer"
        
        self.nqubits = N
        self.wires = wires
        #如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True
        
    def _cal_single_gates(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            x1m = rx(self.params[3*i+0]).matrix
            zm = rz(self.params[3*i+1]).matrix
            x2m = rx(self.params[3*i+2]).matrix
            
            lst1[qbit] = x2m @ zm @ x1m
        return lst1
    
    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j
      
    def operation_dagger(self):
        params_tensor = -1*self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy( params_tensor[3*i+0] )
            params_tensor[3*i+0] = params_tensor[3*i+2]
            params_tensor[3*i+2] = temp
        return XZXLayer(self.nqubits,self.wires,params_tensor)
    
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)









class XZLayer(SingleGateLayer):
    # label = "XZLayer"
    
    def __init__(self,N,wires,params_lst):
        if 2*len(wires) != len(params_lst):
            raise ValueError("XZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZLayer: number of wires must less than N")
        self.label = "XZLayer"
        
        self.nqubits = N
        self.wires = wires
        #如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True
        
    def _cal_single_gates(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            xm = rx(self.params[2*i+0]).matrix
            zm = rz(self.params[2*i+1]).matrix

            lst1[qbit] = zm @ xm
        return lst1
    
    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j
    
    
    def operation_dagger(self):
        params_tensor = -1*self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy( params_tensor[2*i+0] )
            params_tensor[2*i+0] = params_tensor[2*i+1]
            params_tensor[2*i+1] = temp
        return ZXLayer(self.nqubits,self.wires,params_tensor)
    
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)












class ZXLayer(SingleGateLayer):
    # label = "ZXLayer"
    
    def __init__(self,N,wires,params_lst):
        if 2*len(wires) != len(params_lst):
            raise ValueError("ZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("ZXLayer: number of wires must less than N")
        self.label = "ZXLayer"
        
        self.nqubits = N
        self.wires = wires
        #如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True
        
    def _cal_single_gates(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            zm = rz(self.params[2*i+0]).matrix
            xm = rx(self.params[2*i+1]).matrix

            lst1[qbit] = xm @ zm
        return lst1
    
    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j
    
    def operation_dagger(self):
        params_tensor = -1*self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy( params_tensor[2*i+0] )
            params_tensor[2*i+0] = params_tensor[2*i+1]
            params_tensor[2*i+1] = temp
        return XZLayer(self.nqubits,self.wires,params_tensor)
    
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)






class HLayer(SingleGateLayer):
    # label = "HadamardLayer"
    
    def __init__(self,N,wires):
        if len(wires) > N:
            raise ValueError("HadamardLayer: number of wires must less than N")
        
        self.label = "HadamardLayer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = True
     
    def _cal_single_gates(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):

            lst1[qbit] = Hadamard().matrix
        return lst1
        
    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j
    
    def operation_dagger(self):
        return self
    
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':None}
        return info
    
    def params_update(self,params_lst):
        pass





#==============================================================================




class ring_of_cnot(TwoQbitGateLayer):
    # label = "ring_of_cnot_Layer"
    
    def __init__(self,N,wires):
        #ladderdown=True表示用下降的阶梯式排列ring of cnot
        
        if len(wires) > N:
            raise ValueError("ring_of_cnotLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnotLayer: number of wires must >= 2")
        self.label = "ring_of_cnot_Layer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = True
    
    def _gate_fusion_U_expand(self,N):
        if N < 3:
            raise ValueError('ring of cnot : gate_fusion error! N must be >= 3')
        I = torch.eye(2,2) + 0j
        rst = cnot( 2,[0,1] ).U_expand()
        for i in range( 1, N ):
            cur_M = cnot( min(2+i,N),[i, (i+1)%N] ).U_expand()
            if i == N-1:
                rst = cur_M @ rst
            else:
                rst = cur_M @ torch.kron(rst,I)
        return rst
        
    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot( self.nqubits,[ self.wires[0],self.wires[1] ]).U_expand()
        
        if self.wires == list( range(self.nqubits) ):
            return self._gate_fusion_U_expand(self.nqubits)
        
        rst = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        for i,qbit in enumerate( self.wires ):
            # if i == L-1: #临时加的
            #     break
            #rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+1)%L] ]).U_expand() @ I
            rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+1)%L] ]).U_expand() @ rst

        return rst
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        '''
        只支持自上而下的cnot，即上方的qbit一定是control，下方的一定是target
        '''
        if self.wires != list( range(self.nqubits) ):
            raise ValueError('ring_of_cnot,TN_operation error')
        L = len(self.wires)
        for i in range(L-1):
            MPS = cnot(self.nqubits,[i,i+1]).TN_operation(MPS)
            # temp1 = MPS[self.wires[i]] #control bit
            # temp2 = MPS[self.wires[(i+1)%L]] #target bit
            # temp = (temp1.unsqueeze(1) @ temp2.unsqueeze(0) ).permute(2,3,0,1)
            # shape = temp.shape
            # #print(shape)
            # temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
            # temp = cnot().matrix @ temp
            # temp = temp.view(shape[0],shape[1],shape[2],shape[3])
            # temp = temp.permute(2,3,0,1)
            # #融合后的张量恢复成两个张量
            # MPS[self.wires[i]],MPS[self.wires[(i+1)%L]] = TensorDecompAfterTwoQbitGate(temp)
        
        if self.nqubits == 2:
            return MPS
        #======================================================================
        for i in range(L-1):
            # temp1 = MPS[self.wires[i]] #control bit
            # temp2 = MPS[self.wires[(i+1)%L]] #target bit
            
            # temp = (temp1.unsqueeze(1) @ temp2.unsqueeze(0) ).permute(2,3,0,1)
            # shape = temp.shape
            # temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
            '''
            试着直接用非邻近cnot门，不要用SWAP了
            '''
            if i != L-2:
                MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
                #temp = SWAP().matrix @ temp
            else:
                MPS = cnot(self.nqubits,[i+1,i]).TN_operation(MPS)
                #temp = cnot(2,[1,0]).U_expand() @ temp
            # temp = temp.view(shape[0],shape[1],shape[2],shape[3])
            # temp = temp.permute(2,3,0,1)
            # #融合后的张量恢复成两个张量
            # MPS[self.wires[i]],MPS[self.wires[(i+1)%L]] = TensorDecompAfterTwoQbitGate(temp)
        #======================================================================
        for i in range(L-3,-1,-1):
            MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
            # temp1 = MPS[self.wires[i]] #control bit
            # temp2 = MPS[self.wires[(i+1)%L]] #target bit
            
            # temp = (temp1.unsqueeze(1) @ temp2.unsqueeze(0) ).permute(2,3,0,1)
            # shape = temp.shape
            # temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
            # temp = SWAP().matrix @ temp
            # temp = temp.view(shape[0],shape[1],shape[2],shape[3])
            # temp = temp.permute(2,3,0,1)
            # #融合后的张量恢复成两个张量
            # MPS[self.wires[i]],MPS[self.wires[(i+1)%L]] = TensorDecompAfterTwoQbitGate(temp)
        #======================================================================
        return MPS
    
    def operation_dagger(self):
        return ring_of_cnot_dagger(self.nqubits,self.wires)
    
    
    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i+1)%L] for i in range(L)]
        if L == 2:
            info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        else:
            info = {'label':self.label, 'contral_lst':self.wires, 'target_lst':target_lst,'params':None}
        return info
    
    def params_update(self,params_lst):
        pass


class ring_of_cnot_dagger(TwoQbitGateLayer):
    '''
    这是上面ring_of_cnot layer的转置共轭算符，比如一个5qubit线路的ring_of_cnot_dagger
    本质就是4控0,3控4，2控3，1控2，0控1依次的五个cnot
    '''
    
    def __init__(self,N,wires):
        #ladderdown=True表示用下降的阶梯式排列ring of cnot
        
        if len(wires) > N:
            raise ValueError("ring_of_cnot_dagger: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnot_dagger: number of wires must >= 2")
        self.label = "ring_of_cnot_dagger"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = True
    
    def _gate_fusion_U_expand(self,N):
        if N < 3:
            raise ValueError('ring_of_cnot_dagger : gate_fusion error! N must be >= 3')
        I = torch.eye(2,2) + 0j
        rst = cnot( 2,[0,1] ).U_expand()
        for i in range( 1, N ):
            cur_M = cnot( min(2+i,N),[i, (i+1)%N] ).U_expand()
            if i == N-1:
                # rst = cur_M @ rst
                rst = rst @ cur_M
            else:
                # rst = cur_M @ torch.kron(rst,I)
                rst = torch.kron(rst,I) @ cur_M 
        return rst
        
    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot( self.nqubits,[ self.wires[0],self.wires[1] ]).U_expand()
        
        if self.wires == list( range(self.nqubits) ):
            return self._gate_fusion_U_expand(self.nqubits)
        
        rst = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        for i,qbit in enumerate( self.wires ):
            rst = rst @ cnot(self.nqubits,[ self.wires[i],self.wires[(i+1)%L] ]).U_expand() 

        return rst
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        
        if self.wires != list( range(self.nqubits) ):
            raise ValueError('ring_of_cnot_dagger,TN_operation error')
        
        L = len(self.wires)
        
        if self.nqubits == 2:
            MPS = cnot(2,[0,1]).TN_operation(MPS)
            return MPS

        for i in range(L-1):
            if i != L-2:
                MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
            else:
                MPS = cnot(self.nqubits,[i+1,i]).TN_operation(MPS)
        for i in range(L-3,-1,-1):
            MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
        
        for i in range(L-2,-1,-1):
            MPS = cnot(self.nqubits,[i,i+1]).TN_operation(MPS)
            
        return MPS
    
    def operation_dagger(self):
        return ring_of_cnot(self.nqubits, self.wires)
    
    
    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i+1)%L] for i in range(L)]
        if L == 2:
            info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        else:
            info = {'label':self.label, 'contral_lst':self.wires, 'target_lst':target_lst,'params':None}
        return info
    
    def params_update(self,params_lst):
        pass





class ring_of_cnot2(TwoQbitGateLayer):
    # label = "ring_of_cnot2_Layer"
    
    def __init__(self,N,wires):
        
        if len(wires) > N:
            raise ValueError("ring_of_cnot2Layer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnot2Layer: number of wires must >= 2")
        self.label = "ring_of_cnot2_Layer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = False
    
    
    def _gate_fusion_U_expand(self,N):
        if N < 3:
            raise ValueError('ring of cnot : gate_fusion error! N must be >= 3')
        I = torch.eye(2,2) + 0j
        rst = cnot( 3,[0,2] ).U_expand()
        for i in range( 1, N ):
            cur_M = cnot( min(3+i,N),[i, (i+2)%N] ).U_expand()
            if i == N-2 or i == N-1:
                rst = cur_M @ rst
            else:
                rst = cur_M @ torch.kron(rst,I)
        return rst    

    
    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot( self.nqubits,[ self.wires[0],self.wires[1] ]).U_expand()
        
        if self.wires == list( range(self.nqubits) ):
            return self._gate_fusion_U_expand(self.nqubits)
        
        #I = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        rst = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        for i,qbit in enumerate( self.wires ):
            
            rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+2)%L] ]).U_expand() @ rst

        return rst
    
    def operation_dagger(self):
        pass
    
    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i+2)%L] for i in range(L)]
        if L == 2:
            info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        else:
            info = {'label':self.label, 'contral_lst':self.wires, 'target_lst':target_lst,'params':None}
        return info
    
    def params_update(self,params_lst):
        pass







#=========================================================================================




class BasicEntangleLayer(TwoQbitGateLayer):
    # label = "BasicEntangleLayer"
    
    def __init__(self, N, wires, params_lst, repeat=1):
        
        if 3*len(wires)*repeat != len(params_lst):
            raise ValueError("BasicEntangleLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("BasicEntangleLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("BasicEntangleLayer: number of wires must >= 2")
        if repeat < 1:
            raise ValueError("BasicEntangleLayer: number of repeat must >= 1")
        self.label = "BasicEntangleLayer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = len(params_lst)
        self.params = params_lst
        self.repeat = repeat
        
        self.part1_lst, self.part2_lst = [], []
        for i in range(int(self.repeat)):
            self.part1_lst.append( YZYLayer(self.nqubits, self.wires, self.params[i*3*len(wires):(i+1)*3*len(wires)]) )
            self.part2_lst.append( ring_of_cnot(self.nqubits, self.wires) )
        
        self.supportTN = False
        
    def U_expand(self):
        rst = torch.eye(2**self.nqubits) + 0j
        cnot_ring = self.part2_lst[0].U_expand()
        for i in range(self.repeat):
            #rst = self.part2_lst[i].U_expand() @ self.part1_lst[i].U_expand() @ rst
            rst = cnot_ring @ self.part1_lst[i].U_expand() @ rst
        return rst
    
    def operation_dagger(self):
        pass
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires, 'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.num_params = len(params_lst)
        self.params = params_lst
        self.part1_lst, self.part2_lst = [], []
        L = 3*len(self.wires)
        for i in range(self.repeat):
            self.part1_lst.append( YZYLayer(self.nqubits, self.wires, self.params[i*L:(i+1)*L]) )
            self.part2_lst.append( ring_of_cnot(self.nqubits, self.wires) )
        













if __name__ == '__main__':
    print('start')
    N = 4
    wires = list(range(N))
    #wires = [0,1,2,3]
    '''
    验证两比特门MPS作用的正确性与效率提升
    '''
    r1 = ring_of_cnot(N, wires)
    
    psi = torch.zeros(1,2**N)+0.0j
    psi[0,0] = 1.0+0j;psi[0,-1] = 0.50+0j
    psi = nn.functional.normalize( psi,p=2,dim=1 )
    #psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    
    psif = ( r1.U_expand() @ psi.permute(1,0) ).permute(1,0)
    
    MPS = StateVec2MPS(psi,N)
    MPS = r1.TN_operation(MPS)
    psi1 = MPS2StateVec( MPS ).view(1,-1)
    print(psif)
    print(psi1)
    
    
    
    '''
    验证MPS技术的正确性与效率提升
    '''
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;#psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    # psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    
    # psi0 = psi
    # hhh = XYZLayer(N, wires, torch.rand(3*N))
    # t1 = time.time()
    # psif = (hhh.U_expand() @ psi0.permute(1,0)).permute(1,0)
    # t2 = time.time()
    # MPS = StateVec2MPS(psi0,N)
    # MPS = hhh.TT_operation(MPS)
    # t3 = time.time()
    # print(psif)
    # print(t2-t1)
    # print( MPS2StateVec( MPS ) )
    # print(t3-t2)
    
    '''
    测试gate_fusion技术的时间优化效果
    '''
    # roc = ring_of_cnot2(N,wires)
    # t1 = time.time()
    # for i in range(10):
    #     r1 = roc.U_expand()
    # t2 = time.time()
    # for i in range(10):
    #     r2 = roc._gate_fusion_U_expand(roc.nqubits)
    # t3 = time.time()
    # print('r1-r2:',r1-r2)
    # #print('r2:',r2)
    # print('old:',t2-t1)
    # print('new:',t3-t2)
    # print('耗时比：',(t3-t2)/(t2-t1))
    
    
    # N = 2
    # p = torch.rand(3*N)
    # a = ring_of_cnot(N,list(range(N)))
    # print(a.label)
    # print(a.U_expand())
    # print(a.info())
    input('')