# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:43:13 2021

@author: shish
"""
import torch
from typing import List
import copy

from deepquantum.gates.qmath import  IsUnitary, IsHermitian, multi_kron
from deepquantum.gates.qtensornetwork import TensorDecompAfterTwoQbitGate, TensorDecompAfterThreeQbitGate
from deepquantum.gates.qTN_contract import _SingleGateOperation_TN_contract, _SingleGateOperation_TN_contract_Rho, _TwoQbitGate_TN_contract_Rho,\
    _rxx_TN_contract, _ryy_TN_contract, _rzz_TN_contract, _cnot_TN_contract,\
    _cz_TN_contract, _cphase_TN_contract, _cu3_TN_contract, _SWAP_TN_contract,\
    _toffoli_TN_contract, _toffoli_TN_contract_Rho

class Operator(object):
    
    @staticmethod #为了让该函数既可以实例化调用，也可以不实例化直接Operator.gate_expand_1toN()调用
    def gate_expand_1toN(gate, N, index):
        '''
        不要直接用这个函数
        '''
        if N < 1:
            raise ValueError("number of qubits N must be >= 1")
        if index < 0 or index > N - 1:
            raise ValueError("index must between 0~N-1")
        lst1 = [torch.eye(2, 2)]*N
        lst1[index] = gate
        return multi_kron(lst1) + 0j
    
    pass

class Operation(Operator):
    
    @staticmethod
    def two_qubit_control_gate(U,N,control,target):
        '''
        不建议直接使用该函数
        two_qubit_control_gate该函数可实现任意两比特受控门
        代码照抄田泽卉的，但建议用我这个函数名，注意这里的U是controlled-U里的U，而非controlled-U整体
        比如想实现cnot门，cnot表示controlled-not gate，那么U就是not门，即sigma_x(paulix)
        比如想实现cz门，cnot表示controlled-z gate，那么U就是z门，即sigma_z(pauliz)
        '''
        if N < 1:
            raise ValueError("number of qubits(interger N) must be >= 1")
        if max(control,target) > N-1:
            raise ValueError("control&target must <= number of qubits - 1")
        if min(control,target) < 0:
            raise ValueError("control&target must >= 0")
        if control == target:
            raise ValueError("control cannot be equal to target")
        
        zero_zero = torch.tensor( [[1,0],[0,0]] ) + 0j
        one_one = torch.tensor( [[0,0],[0,1]] ) + 0j
        
        lst1 = [torch.eye(2,2)] * N
        lst1[control] = zero_zero
        
        lst2 = [torch.eye(2,2)] * N
        lst2[control] = one_one
        lst2[target] = U
        return multi_kron(lst1) + multi_kron(lst2) + 0j
    
    
    
    @staticmethod
    def multi_control_gate(U,N,control_lst,target):
        '''
        多控制比特受控门，比如典型的toffoli gate就是2个控制1个受控
        control_lst:一个列表，内部是控制比特的索引号
        '''
        if N < 1:
            raise ValueError("number of qubits(interger N) must be >= 1")
            
        if max(max(control_lst),target) > N-1:
            raise ValueError("control&target must <= number of qubits - 1")
            
        if min(min(control_lst),target) < 0:
            raise ValueError("control&target must >= 0")
            
        for each in control_lst:
            if each == target:
                raise ValueError("control cannot be equal to target")
        
        U = U + 0j
        one_one = torch.tensor( [[0,0],[0,1]] ) + 0j
        
        lst1 = [torch.eye(2,2)] * N
        for each in control_lst:
            lst1[each] = one_one
        lst1[target] = U
        
        lst2 = [torch.eye(2,2)] * N
        
        lst3 = [torch.eye(2,2)] * N
        for each in control_lst:
            lst3[each] = one_one
        #multi_kron(lst2) - multi_kron(lst3)对应不做操作的哪些情况
        return multi_kron(lst2) - multi_kron(lst3) + multi_kron(lst1) + 0j
    
 
    
    def IsUnitary(matrix):
        return IsUnitary(matrix)
    
    pass

class SingleGateOperation(Operation):
    def __init__(self,N=-1,wires=-1):
        self.label = "SingleGateOperation"
        self.num_wires = 1               
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.eye(2)+0j
        self.supportTN = True
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        if self.nqubits == -1 or self.wires == -1:
            raise ValueError("SingleGateOperation input error! cannot TN_operation")
        # print(self.wires)
        # temp = MPS[self.wires]
        # temp =  torch.squeeze(self.matrix @ temp.permute(1,2,0).unsqueeze(-1),dim=3)
        # MPS[self.wires] = temp.permute(2,0,1)
        
        MPS[self.wires] = torch.einsum('ab,bcd->acd',[self.matrix,MPS[self.wires]])
        return MPS
    
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _SingleGateOperation_TN_contract(self.nqubits, self.wires, self.matrix,\
                                         MPS, batch_mod=batch_mod)  
        return MPS
    
    def TN_contract_Rho(self, MPDO:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPDO = _SingleGateOperation_TN_contract_Rho(self.nqubits, self.wires, self.matrix,\
                                                    MPDO, batch_mod=batch_mod)
        # print("TN::",MPDO)
        return MPDO

class TwoQbitGateOperation(Operation):
    
    def TN_contract_Rho(self, MPDO:torch.Tensor, batch_mod:bool=False):
        MPDO = _TwoQbitGate_TN_contract_Rho(self.nqubits, self.wires, self.matrix,\
                                             MPDO, batch_mod=batch_mod)
        # print("asdasdasd",MPDO.reshape([2**self.nqubits, 2**self.nqubits]))
        return MPDO


class Observable(Operator):
    
    def IsHermitian(matrix):
        #判断一个矩阵是否厄米
        return IsHermitian(matrix)
    
    pass

class DiagonalOperation(Operation):
    pass


#==============================无参数单比特门==================================
'''
所有的gates、layers必须有self.num_params属性
必须有params_update方法
'''
class Hadamard(SingleGateOperation):
    #没有可调参数
    #只作用在1个qubit上
    #自己是自己的逆操作
    #以下属于类的属性，而非实例的属性
    # label = "Hadamard"
    # num_params = 0
    # num_wires = 1
    # self_inverse = True
    #matrix = torch.sqrt( torch.tensor(0.5) ) * torch.tensor([[1,1],[1,-1]]) + 0j
    
    def __init__(self,N=-1,wires=-1):
        self.label = "Hadamard"
        self.num_params = 0
        self.num_wires = 1               
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.sqrt( torch.tensor(0.5) ) * torch.tensor([[1,1],[1,-1]]) + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Hadamard gate input error! cannot expand")
    
    def operation_dagger(self):
        return self
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass
        
    

class PauliX(SingleGateOperation):
    # label = "PauliX"
    # num_params = 0
    # num_wires = 1               
    # self_inverse = True
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,N=-1,wires=-1):
        self.label = "PauliX"
        self.num_params = 0
        self.num_wires = 1   
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.tensor([[0,1],[1,0]]) + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("PauliX gate input error! cannot expand")
    
    def operation_dagger(self):
        return self
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass
    

class PauliY(SingleGateOperation):
    # label = "PauliY"
    # num_params = 0
    # num_wires = 1
    # self_inverse = True
    #matrix = torch.tensor([[0,-1j],[1j,0]]) + 0j
    
    def __init__(self,N=-1,wires=-1):
        self.label = "PauliY"
        self.num_params = 0
        self.num_wires = 1
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.tensor([[0j,-1j],[1j,0j]])
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
            #return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("PauliY gate input error! cannot expand")
    
    def operation_dagger(self):
        return self
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass


class PauliZ(SingleGateOperation):
    # label = "PauliZ"
    # num_params = 0
    # num_wires = 1               
    # self_inverse = True
    #matrix = torch.tensor([[1,0],[0,-1]]) + 0j
    
    def __init__(self,N=-1,wires=-1):
        self.label = "PauliZ"
        self.num_params = 0
        self.num_wires = 1               
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.tensor([[1,0],[0,-1]]) + 0j
        self.supportTN = True
        self.diagonal = True
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("PauliZ gate input error! cannot expand")
    
    def operation_dagger(self):
        return self
    
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass

#==============================带参数单比特门==================================

class rx(SingleGateOperation):
    # label = "Rx"
    # num_params = 1
    # num_wires = 1            
    # self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=-1,wires=-1):
        self.label = "Rx"
        self.num_params = 1
        self.num_wires = 1            
        self.self_inverse = False
        
        
        theta = theta + torch.tensor(0.0)
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(theta/2.0)*PauliX().matrix + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Rx gate input error! cannot expand")
    
    
    def operation_dagger(self):
        return rx(-1*self.params,self.nqubits,self.wires)
    
    
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(self.params/2.0)*PauliX().matrix + 0j







class ry(SingleGateOperation):
    # label = "Ry"
    # num_params = 1
    # num_wires = 1            
    # self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=-1,wires=-1):
        self.label = "Ry"
        self.num_params = 1
        self.num_wires = 1            
        self.self_inverse = False
        
        theta = theta + torch.tensor(0.0)
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(theta/2.0)*PauliY().matrix + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Ry gate input error! cannot expand")
    
    
    def operation_dagger(self):
        return ry(-1*self.params,self.nqubits,self.wires)
    
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(self.params/2.0)*PauliY().matrix + 0j







class rz(SingleGateOperation):
    # label = "Rz"
    # num_params = 1
    # num_wires = 1
    # self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=-1,wires=-1):
        self.label = "Rz"
        self.num_params = 1
        self.num_wires = 1
        self.self_inverse = False
        
        theta = theta + torch.tensor(0.0)
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(theta/2.0)*PauliZ().matrix + 0j
        self.supportTN = True
        self.diagonal = True
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Rz gate input error! cannot expand")
    
    def operation_dagger(self):
        return rz(-1*self.params,self.nqubits,self.wires)

    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(self.params/2.0)*PauliZ().matrix + 0j





class u1(SingleGateOperation):
    
    '''
    参考：
    U1(λ)=[[1,0],
            [0,exp(iλ)]]
    '''
    def __init__(self,theta,N=-1,wires=-1):
        self.label = "u1"
        self.num_params = 1
        self.num_wires = 1
        self.self_inverse = False
        
        self.nqubits = N
        self.wires = wires
        theta = theta + torch.tensor(0.0)
        self.params = theta
        
        # exp_itheta = torch.cos(theta) + 1j * torch.sin(theta)
        # self.matrix = torch.tensor([[1,0],[0,exp_itheta]]) + 0j
        
        c = torch.cos(0.5*theta)
        s = torch.sin(0.5*theta)
        e = c+1j*s
        self.matrix = e*( c*torch.eye(2,2) - 1j*s*torch.tensor([[1,0],[0,-1]]) )
        
        self.supportTN = True
        self.diagonal = True
        
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("u1 gate input error! cannot expand")
    
    
    def operation_dagger(self):
        return u1(-1*self.params,self.nqubits,self.wires)

    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,theta):
        theta = theta + torch.tensor(0.0)
        self.params = theta
        
        exp_itheta = torch.cos(theta) + 1j * torch.sin(theta)
        self.matrix = torch.tensor([[1,0],[0,exp_itheta]]) + 0j







class u3(SingleGateOperation):
    
    '''
    U3(θ,ϕ,λ)=Rz(ϕ)Rx(−π/2)Rz(θ)Rx(π/2)Rz(λ)
    '''
    def __init__(self,theta_lst,N=-1,wires=-1):
        self.label = "u3"
        self.num_params = 3
        self.num_wires = 1
        self.self_inverse = False
        
        if type(theta_lst) == type([1]):
            theta_lst = torch.tensor(theta_lst)
        self.nqubits = N
        self.wires = wires
        self.params = theta_lst
        
        theta = theta_lst[0]
        phi = theta_lst[1]
        lambd = theta_lst[2]
        self.matrix = \
            rz(phi).matrix \
            @ rx(-0.5*torch.pi).matrix \
            @ rz(theta).matrix \
            @ rx(0.5*torch.pi).matrix \
            @ rz(lambd).matrix
        
        self.supportTN = True
        self.diagonal = False
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            return Operator.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("u3 gate input error! cannot expand")
    
    
    def operation_dagger(self):
        theta, phi, lambd = self.params[0],self.params[1],self.params[2]
        return u3([-theta,-lambd,-phi],self.nqubits,self.wires)
    
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,theta_lst):
        
        if type(theta_lst) == type([1]):
            theta_lst = torch.tensor(theta_lst)
        self.params = theta_lst
        
        theta = theta_lst[0]
        phi = theta_lst[1]
        lambd = theta_lst[2]
        self.matrix = \
            rz(phi).matrix \
            @ rx(-0.5*torch.pi).matrix \
            @ rz(theta).matrix \
            @ rx(0.5*torch.pi).matrix \
            @ rz(lambd).matrix
#==============================带参数两比特门==================================




class rxx(TwoQbitGateOperation):
    # label = "Rxx"
    # num_params = 1
    # num_wires = 2           
    # self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=-1,wires=-1):#wires以list形式输入
        self.label = "Rxx"
        self.num_params = 1
        self.num_wires = 2           
        self.self_inverse = False
    
        theta = theta + torch.tensor(0.0)
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(theta/2.0)*torch.kron(PauliX().matrix,PauliX().matrix) + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            if self.nqubits < 1:
                raise ValueError("number of qubits N must be >= 1")
            if self.wires[0] < 0 or self.wires[0] > self.nqubits - 1 or self.wires[1] < 0  or self.wires[0] > self.nqubits - 1:
                raise ValueError("qbit index must between 0~N-1")
            if self.wires[0] == self.wires[1]:
                raise ValueError("qbit1 cannot be equal to qbit2")
            lst1 = [torch.eye(2,2)]*self.nqubits
            lst2 = [torch.eye(2,2)]*self.nqubits
            lst2[self.wires[0]] =  PauliX().matrix
            lst2[self.wires[1]] =  PauliX().matrix
            rst = torch.cos(self.params/2.0)*multi_kron(lst1) - 1j*torch.sin(self.params/2.0)*multi_kron(lst2)
            return rst + 0j
        else:
            raise ValueError("Rxx gate input error! cannot expand")
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _rxx_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
        
    
    
    def operation_dagger(self):
        return rxx(-1*self.params, self.nqubits, self.wires)
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':list(self.wires),'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(self.params/2.0)*torch.kron(PauliX().matrix,PauliX().matrix) + 0j










class ryy(TwoQbitGateOperation):
    # label = "Ryy"
    # num_params = 1
    # num_wires = 2           
    # self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=-1,wires=-1):
        #wires以list形式输入
        self.label = "Ryy"
        self.num_params = 1
        self.num_wires = 2           
        self.self_inverse = False
        
        theta = theta + torch.tensor(0.0)
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(theta/2.0)*torch.kron(PauliY().matrix,PauliY().matrix) + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            if self.nqubits < 1:
                raise ValueError("number of qubits N must be >= 1")
            if self.wires[0] < 0 or self.wires[0] > self.nqubits - 1 or self.wires[1] < 0  or self.wires[0] > self.nqubits - 1:
                raise ValueError("qbit index must between 0~N-1")
            if self.wires[0] == self.wires[1]:
                raise ValueError("qbit1 cannot be equal to qbit2")
            lst1 = [torch.eye(2,2)]*self.nqubits
            lst2 = [torch.eye(2,2)]*self.nqubits
            lst2[self.wires[0]] =  PauliY().matrix
            lst2[self.wires[1]] =  PauliY().matrix
            rst = torch.cos(self.params/2.0)*multi_kron(lst1) - 1j*torch.sin(self.params/2.0)*multi_kron(lst2)
            return rst + 0j
        else:
            raise ValueError("Ryy gate input error! cannot expand")
    
    
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _ryy_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
        
    
    def operation_dagger(self):
        return ryy(-1*self.params, self.nqubits, self.wires)
    
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':list(self.wires),'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(self.params/2.0)*torch.kron(PauliY().matrix,PauliY().matrix) + 0j










class rzz(TwoQbitGateOperation):
    # label = "Rzz"
    # num_params = 1
    # num_wires = 2           
    # self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=-1,wires=-1):
        #wires以list形式输入
        self.label = "Rzz"
        self.num_params = 1
        self.num_wires = 2           
        self.self_inverse = False
        
        theta = theta + torch.tensor(0.0)
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(theta/2.0)*torch.kron(PauliZ().matrix,PauliZ().matrix) + 0j
        self.supportTN = True
        self.diagonal = True
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            if self.nqubits < 1:
                raise ValueError("number of qubits N must be >= 1")
            if self.wires[0] < 0 or self.wires[0] > self.nqubits - 1 or self.wires[1] < 0  or self.wires[0] > self.nqubits - 1:
                raise ValueError("qbit index must between 0~N-1")
            if self.wires[0] == self.wires[1]:
                raise ValueError("qbit1 cannot be equal to qbit2")
            lst1 = [torch.eye(2,2)]*self.nqubits
            lst2 = [torch.eye(2,2)]*self.nqubits
            lst2[self.wires[0]] =  PauliZ().matrix
            lst2[self.wires[1]] =  PauliZ().matrix
            rst = torch.cos(self.params/2.0)*multi_kron(lst1) - 1j*torch.sin(self.params/2.0)*multi_kron(lst2)
            return rst + 0j
        else:
            raise ValueError("Rzz gate input error! cannot expand")
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _rzz_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
        
    
    
    def operation_dagger(self):
        return rzz(-1*self.params, self.nqubits, self.wires)
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':list(self.wires),'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(self.params/2.0)*torch.kron(PauliZ().matrix,PauliZ().matrix) + 0j



#==============================无参数两比特门==================================



class cnot(TwoQbitGateOperation):
    
    def __init__(self,N=-1,wires=-1):
        #wires以list形式输入
        self.label = "cnot"
        self.num_params = 0
        self.num_wires = 2          
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.tensor([[1,0,0,0],\
                                   [0,1,0,0],\
                                   [0,0,0,1],\
                                   [0,0,1,0]]) + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            sigma_x = torch.tensor( [[0,1],[1,0]] ) + 0j
            control = self.wires[0]
            target = self.wires[1]
            return Operation.two_qubit_control_gate( sigma_x, self.nqubits, control, target )
        else:
            raise ValueError("cnot gate input error! cannot expand")
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        if self.nqubits == -1 or self.wires == -1:
            raise ValueError("cnot gate input error! cannot TN_operation")
        # control = self.wires[0]
        # target = self.wires[1]
        upqbit = min(self.wires)
        downqbit = max(self.wires)
        if upqbit + 1 == downqbit: 
            # temp1 = MPS[upqbit] #control qubit
            # temp2 = MPS[downqbit]  #target qubit
            
            temp = (MPS[upqbit].unsqueeze(1) @ MPS[downqbit].unsqueeze(0) ).permute(2,3,0,1)
            shape = temp.shape
            temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
            if self.wires[0] == upqbit:
                temp = cnot().matrix @ temp
            else:
                temp = cnot(2,[1,0]).U_expand() @ temp
            temp = temp.view(shape[0],shape[1],shape[2],shape[3])
            temp = temp.permute(2,3,0,1)
            #融合后的张量恢复成两个张量
            MPS[upqbit],MPS[downqbit] = TensorDecompAfterTwoQbitGate(temp)
            
            
        else:
            #当cnot门横跨多个量子比特时，需要用SWAP将控制、目标比特搬运至最近邻
            for i in range(upqbit,downqbit-1):
                
                MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
            
            temp = (MPS[downqbit-1].unsqueeze(1) @ MPS[downqbit].unsqueeze(0) ).permute(2,3,0,1)
            shape = temp.shape
            temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
            if self.wires[0] == upqbit:
                temp = cnot().matrix @ temp
            else:
                temp = cnot(2,[1,0]).U_expand() @ temp
            temp = temp.view(shape[0],shape[1],shape[2],shape[3])
            temp = temp.permute(2,3,0,1)
            #融合后的张量恢复成两个张量
            MPS[downqbit-1],MPS[downqbit] = TensorDecompAfterTwoQbitGate(temp)
           
            
            
            for i in range(downqbit-1,upqbit,-1):
                
                MPS = SWAP(self.nqubits,[i-1,i]).TN_operation(MPS)
        return MPS
    
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _cnot_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
        
    
    
    def operation_dagger(self):
        return self

    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        return info
    
    def params_update(self,params):
        pass
    





class cz(TwoQbitGateOperation):
    
    def __init__(self,N=-1,wires=-1):
        #wires以list形式输入
        self.label = "cz"
        self.num_params = 0
        self.num_wires = 2          
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.tensor([[1,0,0,0],\
                                   [0,1,0,0],\
                                   [0,0,1,0],\
                                   [0,0,0,-1]]) + 0j
        self.supportTN = True
        self.diagonal = True
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            sigma_z = torch.tensor( [[1,0],[0,-1]] ) + 0j
            control = self.wires[0]
            target = self.wires[1]
            return Operation.two_qubit_control_gate( sigma_z, self.nqubits, control, target )
        else:
            raise ValueError("cz gate input error! cannot expand")
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        if self.nqubits == -1 or self.wires == -1:
            raise ValueError("TN_operation:cz gate input error! cannot expand")
        if len(self.wires) != 2:
            raise ValueError("TN_operation:cz gate must be applied on 2 qbits")
        upqbit = min(self.wires)
        downqbit = max(self.wires)
        
        for i in range(upqbit,downqbit-1):
            MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
        
        temp = (MPS[downqbit-1].unsqueeze(1) @ MPS[downqbit].unsqueeze(0) ).permute(2,3,0,1)
        shape = temp.shape
        temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
        if self.wires[0] == upqbit:
            #temp = cz().matrix @ temp
            temp = self.matrix @ temp
        else:
            temp = cz(2,[1,0]).U_expand() @ temp
        temp = temp.view(shape[0],shape[1],shape[2],shape[3])
        temp = temp.permute(2,3,0,1)
        #融合后的张量恢复成两个张量
        MPS[downqbit-1],MPS[downqbit] = TensorDecompAfterTwoQbitGate(temp)
        
        for i in range(downqbit-1,upqbit,-1):
            MPS = SWAP(self.nqubits,[i-1,i]).TN_operation(MPS)
        
        return MPS
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _cz_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
        
    
    
    
    def operation_dagger(self):
        return self
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        return info
    
    def params_update(self,params):
        pass
    





class cphase(TwoQbitGateOperation):
    
    def __init__(self, theta, N=-1, wires=-1):
        #wires以list形式输入，包含控制比特和受控比特
        self.label = "cphase"
        self.num_params = 1
        self.num_wires = 2          
        self.self_inverse = False
        
        theta = theta + torch.tensor(0.0)
        self.nqubits = N
        self.wires = wires
        self.params = theta
        
        exp_itheta = torch.cos(theta) + 1j * torch.sin(theta)
        mt = torch.tensor([[0,0,0,0],\
                           [0,0,0,0],\
                           [0,0,0,0],\
                           [0,0,0,1]])
        # self.matrix = torch.tensor([[1,0,0,0],\
        #                             [0,1,0,0],\
        #                             [0,0,1,0],\
        #                             [0,0,0,exp_itheta]]) + 0j
        self.matrix = torch.eye(4,4) + (exp_itheta-1)*mt
        
        self.supportTN = True
        self.diagonal = True
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            exp_itheta = torch.cos(self.params) + 1j * torch.sin(self.params)
            phase_gate = torch.tensor( [[1,0],[0,exp_itheta]] ) + 0j
            control = self.wires[0]
            target = self.wires[1]
            return Operation.two_qubit_control_gate( phase_gate, self.nqubits, control, target )
        else:
            raise ValueError("cphase gate input error! cannot expand")
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        if self.nqubits == -1 or self.wires == -1:
            raise ValueError("TN_operation:cz gate input error! cannot expand")
        if len(self.wires) != 2:
            raise ValueError("TN_operation:cz gate must be applied on 2 qbits")
        upqbit = min(self.wires)
        downqbit = max(self.wires)
        
        for i in range(upqbit,downqbit-1):
            MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
        #=====================================================================
        temp = (MPS[downqbit-1].unsqueeze(1) @ MPS[downqbit].unsqueeze(0) ).permute(2,3,0,1)
        shape = temp.shape
        temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
        if self.wires[0] == upqbit:
            #temp = cphase(self.params).matrix @ temp
            temp = self.matrix @ temp
        else:
            temp = cphase(self.params,2,[1,0]).U_expand() @ temp
        temp = temp.view(shape[0],shape[1],shape[2],shape[3])
        temp = temp.permute(2,3,0,1)
        #融合后的张量恢复成两个张量
        MPS[downqbit-1],MPS[downqbit] = TensorDecompAfterTwoQbitGate(temp)
        #=====================================================================
        for i in range(downqbit-1,upqbit,-1):
            MPS = SWAP(self.nqubits,[i-1,i]).TN_operation(MPS)
        
        return MPS
    
    def TN_contract(self,MPS:torch.Tensor, batch_mod:bool=False):
        MPS = _cphase_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS

    
    def operation_dagger(self):
        return cphase(-1*self.params,self.nqubits,self.wires)
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':self.params}
        return info
    
    def params_update(self,theta):
        theta = theta + torch.tensor(0.0)
        self.params = theta
        exp_itheta = torch.cos(theta) + 1j * torch.sin(theta)
        self.matrix = torch.tensor([[1,0,0,0],\
                                    [0,1,0,0],\
                                    [0,0,1,0],\
                                    [0,0,0,exp_itheta]]) + 0j




class cu3(TwoQbitGateOperation):
    def __init__(self,theta_lst,N=-1,wires=-1):
        #wires以list形式输入
        self.label = "cu3"
        self.num_params = 3
        self.num_wires = 2          
        self.self_inverse = False
        
        self.nqubits = N
        self.wires = wires
        
        if type(theta_lst) == type([1]):
            theta_lst = torch.tensor(theta_lst)
        self.params = theta_lst
        
        theta = theta_lst[0]
        phi = theta_lst[1]
        lambd = theta_lst[2]
        self.u_matrix = \
            rz(phi).matrix \
            @ rx(-0.5*torch.pi).matrix \
            @ rz(theta).matrix \
            @ rx(0.5*torch.pi).matrix \
            @ rz(lambd).matrix
        self.matrix = Operation.two_qubit_control_gate( self.u_matrix, 2, 0, 1 )
        
        self.supportTN = True
        self.diagonal = False
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            control = self.wires[0]
            target = self.wires[1]
            return Operation.two_qubit_control_gate( self.u_matrix, self.nqubits, control, target )
        else:
            raise ValueError("cu3 gate input error! cannot expand")
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        if self.nqubits == -1 or self.wires == -1:
            raise ValueError("TN_operation: cu3 gate input error! cannot expand")
        if len(self.wires) != 2:
            raise ValueError("TN_operation: cu3 gate must be applied on 2 qbits")
        upqbit = min(self.wires)
        downqbit = max(self.wires)
        
        for i in range(upqbit,downqbit-1):
            MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
        #=====================================================================
        temp = (MPS[downqbit-1].unsqueeze(1) @ MPS[downqbit].unsqueeze(0) ).permute(2,3,0,1)
        shape = temp.shape
        temp = temp.view(shape[0],shape[1],shape[2]*shape[3],1)
        if self.wires[0] == upqbit:
            temp = self.matrix @ temp
        else:
            #temp = cu3(self.params,2,[1,0]).U_expand() @ temp
            temp = Operation.two_qubit_control_gate( self.u_matrix, 2, 1, 0 ) @ temp
        temp = temp.view(shape[0],shape[1],shape[2],shape[3])
        temp = temp.permute(2,3,0,1)
        #融合后的张量恢复成两个张量
        MPS[downqbit-1],MPS[downqbit] = TensorDecompAfterTwoQbitGate(temp)
        #=====================================================================
        for i in range(downqbit-1,upqbit,-1):
            MPS = SWAP(self.nqubits,[i-1,i]).TN_operation(MPS)
        
        return MPS
    
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _cu3_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
        
            
    
    def operation_dagger(self):
        theta, phi, lambd = self.params[0],self.params[1],self.params[2]
        return cu3([-theta,-lambd,-phi],self.nqubits,self.wires)
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':self.params}
        return info
    
    def params_update(self,theta_lst):
        if type(theta_lst) == type([1]):
            theta_lst = torch.tensor(theta_lst)
        self.params = theta_lst
        
        theta = theta_lst[0]
        phi = theta_lst[1]
        lambd = theta_lst[2]
        self.u_matrix = \
            rz(phi).matrix \
            @ rx(-0.5*torch.pi).matrix \
            @ rz(theta).matrix \
            @ rx(0.5*torch.pi).matrix \
            @ rz(lambd).matrix
        self.matrix = Operation.two_qubit_control_gate( self.u_matrix, 2, 0, 1 )







class SWAP(TwoQbitGateOperation):
    
    def __init__(self,N=-1,wires=-1):
        #wires以list形式输入
        self.label = "SWAP"
        self.num_params = 0
        self.num_wires = 2          
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        self.matrix = torch.tensor([[1,0,0,0],\
                                   [0,0,1,0],\
                                   [0,1,0,0],\
                                   [0,0,0,1]]) + 0j
        self.supportTN = True
        self.diagonal = False
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            qbit1 = self.wires[0]
            qbit2 = self.wires[1]
            
            zero_zero = torch.tensor( [[1,0],[0,0]] ) + 0j
            one_one = torch.tensor( [[0,0],[0,1]] ) + 0j
            zero_one = torch.tensor( [[0,1],[0,0]] ) + 0j
            one_zero = torch.tensor( [[0,0],[1,0]] ) + 0j
            
            lst1 = [torch.eye(2,2)] * self.nqubits
            lst1[qbit1] = zero_zero
            lst1[qbit2] = zero_zero
            
            lst2 = [torch.eye(2,2)] * self.nqubits
            lst2[qbit1] = one_zero
            lst2[qbit2] = zero_one
            
            lst3 = [torch.eye(2,2)] * self.nqubits
            lst3[qbit1] = zero_one
            lst3[qbit2] = one_zero
            
            lst4 = [torch.eye(2,2)] * self.nqubits
            lst4[qbit1] = one_one
            lst4[qbit2] = one_one
            
            return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4) + 0j
        else:
            raise ValueError("SWAP gate input error! cannot expand")
    
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        '''
        SWAP门是tensor network的核心，很多跨度很大的多比特门的分解需要一堆SWAP
        '''
        if self.nqubits == -1 or self.wires == -1:
            raise ValueError("TN_operation: SWAP gate input error! cannot expand")
        if len(self.wires) != 2:
            raise ValueError("TN_operation: SWAP gate must be applied on 2 qbits")
        upqbit = min(self.wires)
        downqbit = max(self.wires)
        for i in range(upqbit,downqbit):
            temp = (MPS[i].unsqueeze(1) @ MPS[i+1].unsqueeze(0) ).permute(2,3,0,1)
            s = temp.shape
            temp = temp.view(s[0],s[1],s[2]*s[3],1)
            temp = (self.matrix @ temp).view(s[0],s[1],s[2],s[3])
            temp = temp.permute(2,3,0,1)
            MPS[i],MPS[i+1] = TensorDecompAfterTwoQbitGate(temp)
        for i in range(downqbit-1,upqbit,-1):
            temp = (MPS[i-1].unsqueeze(1) @ MPS[i].unsqueeze(0) ).permute(2,3,0,1)
            s = temp.shape
            temp = temp.view(s[0],s[1],s[2]*s[3],1)
            temp = (self.matrix @ temp).view(s[0],s[1],s[2],s[3])
            temp = temp.permute(2,3,0,1)
            MPS[i-1],MPS[i] = TensorDecompAfterTwoQbitGate(temp)
            
           
        return MPS
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _SWAP_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
        
            
    
    def operation_dagger(self):
        return self
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':[], 'target_lst':list(self.wires),'params':None}
        return info
    
    def params_update(self,params):
        pass





#==============================无参数多比特门==================================

class toffoli(Operation):
    
    def __init__(self,N=-1,wires=-1):
        #wires以list形式输入
        self.label = "toffoli"
        self.num_params = 0
        self.num_wires = 3       
        self.self_inverse = True
        
        self.nqubits = N
        self.wires = wires
        if wires != -1:
            if len(wires) != 3:
                raise ValueError("toffoli gate must be applied on 3 qubits")
            
            self.control_lst = [ wires[0], wires[1] ]
            self.target_lst = [ wires[2] ]
        self.matrix = torch.tensor([[1,0,0,0,0,0,0,0],\
                                   [0,1,0,0,0,0,0,0],\
                                   [0,0,1,0,0,0,0,0],\
                                   [0,0,0,1,0,0,0,0],\
                                   [0,0,0,0,1,0,0,0],\
                                   [0,0,0,0,0,1,0,0],\
                                   [0,0,0,0,0,0,0,1],\
                                   [0,0,0,0,0,0,1,0]]) + 0j
        self.supportTN = True
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        
        if self.nqubits != -1 and self.wires != -1:
            sigma_x = torch.tensor( [[0,1],[1,0]] ) + 0j
            
            return Operation.multi_control_gate( sigma_x, self.nqubits, self.control_lst, self.target_lst[0] )
        else:
            raise ValueError("toffoli gate input error! cannot expand")
    
    def TN_operation(self,MPS:List[torch.Tensor])->List[torch.Tensor]:
        if self.nqubits == -1 or self.wires == -1:
            raise ValueError("toffoli gate input error! cannot TN_operation")
        sw = sorted(self.wires)
        upqbit = sw[0]
        midqbit = sw[1]
        downqbit = sw[2]
        #通过一堆SWAP把upqbit和downqbit移动到midqbit的近邻
        for i in range(upqbit,midqbit-1):
            MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
            
        for i in range(downqbit,midqbit+1,-1):
            MPS = SWAP(self.nqubits,[i-1,i]).TN_operation(MPS)
            
        
        
        
        temp = (MPS[midqbit-1].unsqueeze(1).unsqueeze(1) \
                @ MPS[midqbit].unsqueeze(1).unsqueeze(0) \
                @ MPS[midqbit+1].unsqueeze(0).unsqueeze(0) ).permute(3,4,0,1,2)
        shape = temp.shape
        temp = temp.view(shape[0],shape[1],shape[2]*shape[3]*shape[4],1)
        if downqbit == self.wires[2]:
            temp = toffoli().matrix @ temp
        elif midqbit == self.wires[2]:
            temp = toffoli(3,[0,2,1]).U_expand() @ temp
        else:
            temp = toffoli(3,[1,2,0]).U_expand() @ temp
        temp = temp.view(shape[0],shape[1],shape[2],shape[3],shape[4])
        temp = temp.permute(2,3,4,0,1)
        #融合后的张量恢复成两个张量
        MPS[midqbit-1], MPS[midqbit], MPS[midqbit+1] = TensorDecompAfterThreeQbitGate(temp)
        
        
        
        #通过一堆SWAP把upqbit和downqbit移动到midqbit的近邻
        for i in range(midqbit-1,upqbit,-1):
            MPS = SWAP(self.nqubits,[i-1,i]).TN_operation(MPS)
            
        for i in range(midqbit+1,downqbit):
            MPS = SWAP(self.nqubits,[i,i+1]).TN_operation(MPS)
            
        return MPS
    
    
    def TN_contract(self, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPS = _toffoli_TN_contract(self.nqubits, self.wires, self.matrix, MPS, batch_mod=batch_mod)
        return MPS
    
    def TN_contract_Rho(self, MPDO:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
        MPDO = _toffoli_TN_contract_Rho(self.nqubits, self.wires, self.matrix, MPDO, batch_mod=batch_mod)
        return MPDO
        
    
    def operation_dagger(self):
        return self
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':self.control_lst, 'target_lst':self.target_lst,'params':None}
        return info
    
    def params_update(self,params):
        pass







class multi_control_cnot(Operation):
    
    
    def __init__(self,N=-1,wires=-1):
        #wires以list形式输入
        self.num_params = 0
        self.self_inverse = True
        
        self.label = str(len(wires)-1)+"_control_cnot"
        self.num_wires = len(wires)
        
        self.nqubits = N
        self.wires = wires
        self.control_lst =  list(wires[0:len(wires)-1]) 
        self.target_lst = [ wires[-1] ]
        self.supportTN = False
        self.diagonal = False
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != -1 and self.wires != -1:
            sigma_x = torch.tensor( [[0,1],[1,0]] ) + 0j
            return Operation.multi_control_gate( sigma_x, self.nqubits, self.control_lst, self.target_lst[0] )
        else:
            raise ValueError(self.label+" input error! cannot expand")
    
    def operation_dagger(self):
        return self
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'label':self.label, 'contral_lst':self.control_lst, 'target_lst':self.target_lst,'params':None}
        return info
    
    def params_update(self,params):
        pass










if __name__ == "__main__":
    # print('start')
    # h0 = rx(torch.tensor(0.5),3,1)
    # print(h0.matrix)
    # #h = Hadamard(1,0)
    # print(h0.U_expand())
    # #print(h.label,' ',h.self_inverse)
    #c1 = cnot(5,[3,0])
    c1 = multi_control_cnot(6,[0,1,2,3])
    #print(c1.matrix)
    print(c1.U_expand())
    print(c1.info())
    input("")
    