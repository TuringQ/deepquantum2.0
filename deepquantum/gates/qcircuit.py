# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:16:17 2021

@author: shish
"""
import torch
from collections.abc import Iterable
from deepquantum.layers.qlayers import HLayer,XYZLayer,YZYLayer,XZXLayer,XZLayer,ZXLayer,\
    ring_of_cnot,ring_of_cnot2,BasicEntangleLayer
from deepquantum.gates.qoperator import Hadamard,PauliX,PauliY,PauliZ,rx,ry,rz,u1,u3,\
    rxx,ryy,rzz,cnot,cz,cphase,cu3,SWAP,toffoli,multi_control_cnot
    
from deepquantum.gates.qmath import multi_kron
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec

from typing import List
import copy
#import multiprocessing as mp


class Circuit(object):
    def __init__(self, N):
        self.nqubits = N  # 总QuBit的个数
        self.gate = []  # 顺序添加各类门
        #self._U = torch.tensor(1.0) + 0j     # 线路酉矩阵，初始为1
        self.cir_params = {} #记录线路中所有门、层的参数
        self.cir_num_params = 0
        
        
        
    def state_init(self):
        '''
        返回线路的初始态，默认全为|0>态,避免内存浪费，需要时再调用
        '''
        state_init = torch.zeros(2**self.nqubits)
        state_init[0] = 1.0
        state_init = state_init + 0j
        return state_init

    def add(self, gate):
        self.gate.append(gate)
        if gate.num_params != 0:
            self.cir_num_params += gate.num_params
            self.cir_params[len(self.gate)-1] = gate.params
        else:
            self.cir_params[len(self.gate)-1] = None

    
    def U(self, left_to_right=True,cuda=False):
        
        U_overall = torch.eye(2 ** self.nqubits, 2 ** self.nqubits) + 0j
        for i, each_oper in enumerate( self.gate ):
            u_matrix = each_oper.U_expand()        
            if left_to_right:
                U_overall = u_matrix @ U_overall
            else:
                U_overall = U_overall @ u_matrix
        #self._U = U_overall
       
        return U_overall
    
    def TN_evolution(self, MPS:List[torch.Tensor])->List[torch.Tensor]:
        if len(MPS) != self.nqubits:
            raise ValueError('TN_evolution:MPS tensor list must have N elements!')
        for idx, oper in enumerate(self.gate):
            #print(idx)
            if oper.supportTN == True:
                MPS = oper.TN_operation(MPS)
            else:
                raise ValueError(str(oper.info()['label'])
                                 +'-TN_evolution:some part of circuit do not support Tensor Network')
        return MPS
    
    
    
    def cir_expectation(self,init_state,M,TN=True):
        if init_state.shape[0] != 1:
            raise ValueError('cir_expectation init_state shape error')
        if init_state.shape[1] != int(2**self.nqubits):
            raise ValueError('cir_expectation init_state shape error')
        if M.shape[0] != M.shape[1]:
            raise ValueError('cir_expectation M shape error')
        
        if self.TN_check() == True and TN == True:
            MPS = StateVec2MPS( init_state.view(1,-1), self.nqubits )
            psi_f = MPS2StateVec( self.TN_evolution( MPS ) )
            expec = ( psi_f.view(1,-1).conj() @ M @ psi_f.view(-1,1) ).real.squeeze()
        else:
            psi_f = ( self.U() @ init_state.view(-1,1) )
            expec = ( psi_f.view(1,-1).conj() @ M @ psi_f.view(-1,1) ).real.squeeze()
        return expec
    
    
    
    # def circuit_check(self):
    #     print('qubit数目：',self.nqubits,'  gate&layer数目：',len(self.gate))
    #     unsupportTN = 0
    #     for idx,g in enumerate(self.gate):
    #         if g.nqubits != self.nqubits:
    #             raise ValueError('circuit_check ERROR:gate&layers nqubits must equal to circuit nqubits')
    #         if g.supportTN == False:
    #             unsupportTN += 1
    #     print('number of gate&layers do not support TN:',unsupportTN)
    
    
    def TN_check(self)->bool:
        '''
        判断线路中是否所有的门都支持tensor network操作
        '''
        for idx,g in enumerate(self.gate):
            if g.supportTN == False:
                return False
        return True
    
    
    # def num_params(self)->int:
    #     '''
    #     返回整个线路所需的参数数量
    #     '''
    #     num_params = 0
    #     for oper in self.gate:
    #         num_params += oper.num_params
    #     return num_params
    
    
    def draw(self):
        pass
    
    def clear(self):
        self.gate = []
        #self._U = torch.tensor(1.0) + 0j

        



#==============================================================================







    def Hadamard(self, wires):
        if isinstance(wires, Iterable):
            self.add( HLayer(self.nqubits, wires) )
        else:
            self.add( Hadamard(self.nqubits, wires) )
    
    def PauliX(self, wires):
        self.add( PauliX(self.nqubits, wires) )
    
    def PauliY(self, wires):
        self.add( PauliY(self.nqubits, wires) )
    
    def PauliZ(self, wires):
        self.add( PauliZ(self.nqubits, wires) )
    
    def rx(self, theta, wires):
        self.add( rx(theta, self.nqubits, wires) )
    
    def ry(self, theta, wires):
        self.add( ry(theta, self.nqubits, wires) )
    
    def rz(self, theta, wires):
        self.add( rz(theta, self.nqubits, wires) )
    
    def u1(self, theta, wires):
        self.add( u1(theta, self.nqubits, wires) )
    
    def u3(self, theta_lst, wires):
        self.add( u3(theta_lst, self.nqubits, wires) )
    
    def rxx(self, theta, wires):
        self.add( rxx(theta, self.nqubits, wires) )
    
    def ryy(self, theta, wires):
        self.add( ryy(theta, self.nqubits, wires) )
    
    def rzz(self, theta, wires):
        self.add( rzz(theta, self.nqubits, wires) )
    
    def cnot(self, wires):
        self.add( cnot(self.nqubits, wires) )
    
    def cz(self, wires):
        self.add( cz(self.nqubits, wires) )
    
    def cphase(self, theta, wires):
        self.add( cphase(theta, self.nqubits, wires) )
    
    def cu3(self, theta_lst, wires):
        self.add( cu3(theta_lst, self.nqubits, wires) )
    
    def SWAP(self, wires):
        self.add( SWAP(self.nqubits, wires) )
    
    def toffoli(self, wires):
        self.add( toffoli(self.nqubits, wires) )
    
    def multi_control_cnot(self, wires):
        self.add( multi_control_cnot(self.nqubits, wires) )
    #====================================================================    
    def XYZLayer(self, wires, params_lst):
        self.add( XYZLayer(self.nqubits, wires, params_lst) )
    
    def YZYLayer(self, wires, params_lst):
        self.add( YZYLayer(self.nqubits, wires, params_lst) )
    
    def XZXLayer(self, wires, params_lst):
        self.add( XZXLayer(self.nqubits, wires, params_lst) )
    
    def XZLayer(self, wires, params_lst):
        self.add( XZLayer(self.nqubits, wires, params_lst) )
    
    def ZXLayer(self, wires, params_lst):
        self.add( ZXLayer(self.nqubits, wires, params_lst) )
    
    def ring_of_cnot(self, wires):
        self.add( ring_of_cnot(self.nqubits, wires) )
    
    def ring_of_cnot2(self, wires):
        self.add( ring_of_cnot2(self.nqubits, wires) )
    
    def BasicEntangleLayer(self, wires, params_lst, repeat=1):
        self.add( BasicEntangleLayer(self.nqubits, wires, params_lst, repeat) )
    


class parameter_shift(object):
    '''
    线路中设计的门、层必须有更新参数的method：def params_update(self,theta_lst)
    '''
    def __init__(self, cir, psi_init, M):
        self.cir = cir
        self.init_state = psi_init
        self.M = M
    
    def cal_params_grad(self):
        grad_lst = []
        for idx,gate in enumerate( self.cir.gate ):
            #遍历线路中的每个gate、layer
            if gate.num_params == 0:
                #首先判断该gate是否有参数，没参数直接跳过
                continue
                
            p = copy.deepcopy( self.cir.cir_params[idx] )
            if len(p.shape) == 0:
                #只有单个参数
                self.cir.gate[idx].params_update(p + 0.5*torch.pi)
                e1 = self.cir.cir_expectation(self.init_state, self.M)
                self.cir.gate[idx].params_update(p - 0.5*torch.pi)
                e2 = self.cir.cir_expectation(self.init_state, self.M)
                self.cir.gate[idx].params_update(p) #最后别忘了把参数恢复
                grad_lst.append( 0.5*(e1 - e2) )
            
            elif len(p.shape) == 1:
                #参数是个一维张量
                for i,each_p in enumerate(p):
                    p[i] = p[i] + 0.5*torch.pi
                    self.cir.gate[idx].params_update(p)
                    e1 = self.cir.cir_expectation(self.init_state, self.M)
                    
                    p[i] = p[i] - 1.0*torch.pi
                    self.cir.gate[idx].params_update(p)
                    e2 = self.cir.cir_expectation(self.init_state, self.M)
                    
                    p[i] = p[i] + 0.5*torch.pi
                    self.cir.gate[idx].params_update(p) 
                    #最后别忘了把参数恢复
                    grad_lst.append( 0.5*(e1 - e2) )
            else:
                raise ValueError('cal_params_grad: error about params shape')
                        
        assert len(grad_lst) == self.cir.cir_num_params
        return torch.tensor(grad_lst)
                
                
        




if __name__ == '__main__':
    
    cir = Circuit(3)
    
    cir.cphase(1.234, [1,2])
    
    cir.Hadamard([0,1])
    
    cir.PauliX(0)
    
    cir.cnot([1,0])
    
    cir.rx(torch.pi/5.0, 1)
    
    cir.BasicEntangleLayer([0,1,2], torch.rand(9))
    
    cir.SWAP([0,2])
    
    for each in cir.gate:
        print(each.info()['label'])
    
    print('\n',cir.U())
    lst1 = [torch.eye(2)]*3
    multi_kron(lst1)
    #print(type(cir))
    print(cir.cir_params)
    
    I = torch.eye(2)
    lst = [I]*cir.nqubits
    lst[0] = PauliZ().matrix
    M = multi_kron(lst)
    
    ps = parameter_shift(cir, cir.state_init().view(1,-1), M)
    grad = ps.cal_params_grad()
    print(grad)
    input('qcircuit.py END')
        
        
