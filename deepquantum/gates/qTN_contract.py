# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 08:49:16 2021

@author: shish
"""

'''
此脚本需被封装成PYD文件
'''
import torch
import torch.nn as nn
import random
import math
import time
from typing import List
print('this is .py file qTN_contract.py')
def _SingleGateOperation_TN_contract(nqubits, wires, matrix,\
                                     MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("SingleGateOperation input error! cannot TN_contract")
    if batch_mod==False:
        permute_shape = list( range(nqubits) )
        permute_shape[wires] = nqubits - 1
        permute_shape[nqubits - 1] = wires
        MPS = MPS.permute(permute_shape).unsqueeze(-1)
        MPS = ( matrix @ MPS ).squeeze(-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        permute_shape = list( range(nqubits+1) )
        permute_shape[wires+1] = nqubits
        permute_shape[nqubits] = wires+1
        MPS = MPS.permute(permute_shape).unsqueeze(-1)
        MPS = ( matrix @ MPS ).squeeze(-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS

def _rxx_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("rxx gate input error! cannot TN_contract")
    cqbit = wires[0]
    tqbit = wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        permute_shape = list( range(nqubits) )
        permute_shape.remove(cqbit)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit,nqubits-2)
            permute_shape.insert(tqbit,nqubits-1)
        else:
            permute_shape.insert(tqbit,nqubits-1)
            permute_shape.insert(cqbit,nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        permute_shape = list( range(nqubits+1) )
        permute_shape.remove(cqbit+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([s[0]]+[2]*nqubits)
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit+1,nqubits-1)
            permute_shape.insert(tqbit+1,nqubits)
        else:
            permute_shape.insert(tqbit+1,nqubits)
            permute_shape.insert(cqbit+1,nqubits-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS

def _ryy_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("rxx gate input error! cannot TN_contract")
    cqbit = wires[0]
    tqbit = wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        permute_shape = list( range(nqubits) )
        permute_shape.remove(cqbit)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit,nqubits-2)
            permute_shape.insert(tqbit,nqubits-1)
        else:
            permute_shape.insert(tqbit,nqubits-1)
            permute_shape.insert(cqbit,nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        permute_shape = list( range(nqubits+1) )
        permute_shape.remove(cqbit+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([s[0]]+[2]*nqubits)
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit+1,nqubits-1)
            permute_shape.insert(tqbit+1,nqubits)
        else:
            permute_shape.insert(tqbit+1,nqubits)
            permute_shape.insert(cqbit+1,nqubits-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS

def _rzz_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("rxx gate input error! cannot TN_contract")
    cqbit = wires[0]
    tqbit = wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        permute_shape = list( range(nqubits) )
        permute_shape.remove(cqbit)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit,nqubits-2)
            permute_shape.insert(tqbit,nqubits-1)
        else:
            permute_shape.insert(tqbit,nqubits-1)
            permute_shape.insert(cqbit,nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        permute_shape = list( range(nqubits+1) )
        permute_shape.remove(cqbit+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([s[0]]+[2]*nqubits)
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit+1, nqubits-1)
            permute_shape.insert(tqbit+1, nqubits)
        else:
            permute_shape.insert(tqbit+1, nqubits)
            permute_shape.insert(cqbit+1, nqubits-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS


def _cnot_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("cnot gate input error! cannot TN_contract")
    cqbit = wires[0]
    tqbit = wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        permute_shape = list( range(nqubits) )
        permute_shape.remove(cqbit)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit, nqubits-2)
            permute_shape.insert(tqbit, nqubits-1)
        else:
            permute_shape.insert(tqbit, nqubits-1)
            permute_shape.insert(cqbit, nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        permute_shape = list( range(nqubits+1) )
        permute_shape.remove(cqbit+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([s[0]]+[2]*nqubits)
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit+1, nqubits-1)
            permute_shape.insert(tqbit+1, nqubits)
        else:
            permute_shape.insert(tqbit+1, nqubits)
            permute_shape.insert(cqbit+1, nqubits-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS

def _cz_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("cz gate input error! cannot TN_contract")
    cqbit = wires[0]
    tqbit = wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        permute_shape = list( range(nqubits) )
        permute_shape.remove(cqbit)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        if cqbit < tqbit:
            permute_shape.insert(cqbit,nqubits-2)
            permute_shape.insert(tqbit,nqubits-1)
        else:
            permute_shape.insert(tqbit, nqubits-1)
            permute_shape.insert(cqbit, nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        permute_shape = list( range(nqubits+1) )
        permute_shape.remove(cqbit+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        #MPS = MPS.view([2]*self.nqubits)
        MPS = MPS.view([s[0]]+[2]* nqubits)
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        if cqbit < tqbit:
            permute_shape.insert(cqbit+1, nqubits-1)
            permute_shape.insert(tqbit+1, nqubits)
        else:
            permute_shape.insert(tqbit+1, nqubits)
            permute_shape.insert(cqbit+1, nqubits-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS

def _cphase_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or  wires == -1:
        raise ValueError("cphase gate input error! cannot TN_contract")
    cqbit =  wires[0]
    tqbit =  wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) ==  nqubits
        permute_shape = list( range( nqubits) )
        permute_shape.remove(cqbit)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        if cqbit < tqbit:
            permute_shape.insert(cqbit, nqubits-2)
            permute_shape.insert(tqbit, nqubits-1)
        else:
            permute_shape.insert(tqbit, nqubits-1)
            permute_shape.insert(cqbit, nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        permute_shape = list( range( nqubits+1) )
        permute_shape.remove(cqbit+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([s[0]]+[2]*nqubits)
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit+1, nqubits-1)
            permute_shape.insert(tqbit+1, nqubits)
        else:
            permute_shape.insert(tqbit+1, nqubits)
            permute_shape.insert(cqbit+1, nqubits-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS

def _cu3_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("cu3 gate input error! cannot TN_contract")
    cqbit = wires[0]
    tqbit = wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        permute_shape = list( range(nqubits) )
        permute_shape.remove(cqbit)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        if cqbit < tqbit:
            permute_shape.insert(cqbit, nqubits-2)
            permute_shape.insert(tqbit, nqubits-1)
        else:
            permute_shape.insert(tqbit, nqubits-1)
            permute_shape.insert(cqbit, nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        permute_shape = list( range(nqubits+1) )
        permute_shape.remove(cqbit+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-2] = s[-2]*s[-1]
        s[-1] = 1
        MPS = MPS.reshape(s)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([s[0]]+[2]*nqubits)
        
        permute_shape = list( range( nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        #permute_shape = permute_shape[:-2]
        if cqbit < tqbit:
            permute_shape.insert(cqbit+1, nqubits-1)
            permute_shape.insert(tqbit+1, nqubits)
        else:
            permute_shape.insert(tqbit+1, nqubits)
            permute_shape.insert(cqbit+1, nqubits-1)
        MPS = MPS.permute(permute_shape)
        
        return MPS

def _SWAP_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("SWAP gate input error! cannot TN_contract")
    cqbit = wires[0]
    tqbit = wires[1]
    assert cqbit != tqbit
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        
        permute_shape = list( range(nqubits) )
        permute_shape[cqbit] = tqbit
        permute_shape[tqbit] = cqbit
        MPS = MPS.permute(permute_shape)
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        
        permute_shape = list( range(nqubits+1) )
        permute_shape[cqbit+1] = tqbit+1
        permute_shape[tqbit+1] = cqbit+1
        MPS = MPS.permute(permute_shape)
        return MPS

def _toffoli_TN_contract(nqubits, wires, matrix, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    if nqubits == -1 or wires == -1:
        raise ValueError("toffoli gate input error! cannot TN_contract")
    cqbit1 = wires[0]
    cqbit2 = wires[1]
    tqbit = wires[2]
    assert cqbit1 != tqbit and cqbit2 != tqbit and cqbit1 != cqbit2
    
    if batch_mod == False:
        assert len(MPS.shape) == nqubits
        
        permute_shape = list( range(nqubits) )
        permute_shape.remove(cqbit1)
        permute_shape.remove(cqbit2)
        permute_shape.remove(tqbit)
        permute_shape = permute_shape + [cqbit1, cqbit2, tqbit]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-3] = s[-3]*s[-2]*s[-1]
        s[-2],s[-1] = 1,1
        MPS = MPS.reshape(s)
        MPS = MPS.squeeze(-1)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([2]*nqubits)
        
        permute_shape = list( range(nqubits) )
        permute_shape.pop()
        permute_shape.pop()
        permute_shape.pop()
        if cqbit1 < cqbit2 and cqbit2 < tqbit:
            permute_shape.insert(cqbit1, nqubits-3)
            permute_shape.insert(cqbit2, nqubits-2)
            permute_shape.insert(tqbit,  nqubits-1)
        elif cqbit1 < tqbit and tqbit < cqbit2:
            permute_shape.insert(cqbit1, nqubits-3)
            permute_shape.insert(tqbit,  nqubits-1)
            permute_shape.insert(cqbit2, nqubits-2)
        elif cqbit2 < tqbit and tqbit < cqbit1:
            permute_shape.insert(cqbit2, nqubits-2)
            permute_shape.insert(tqbit,  nqubits-1)
            permute_shape.insert(cqbit1, nqubits-3)
        elif cqbit2 < cqbit1 and cqbit1 < tqbit:
            permute_shape.insert(cqbit2, nqubits-2)
            permute_shape.insert(cqbit1, nqubits-3)
            permute_shape.insert(tqbit,  nqubits-1)
        elif tqbit < cqbit1 and cqbit1 < cqbit2:
            permute_shape.insert(tqbit,  nqubits-1)
            permute_shape.insert(cqbit1, nqubits-3)
            permute_shape.insert(cqbit2, nqubits-2)
        elif tqbit < cqbit2 and cqbit2 < cqbit1:
            permute_shape.insert(tqbit,  nqubits-1)
            permute_shape.insert(cqbit2, nqubits-2)
            permute_shape.insert(cqbit1, nqubits-3)
        MPS = MPS.permute(permute_shape)
        
        return MPS
    else:
        assert len(MPS.shape) == nqubits+1
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.remove(cqbit1+1)
        permute_shape.remove(cqbit2+1)
        permute_shape.remove(tqbit+1)
        permute_shape = permute_shape + [cqbit1+1, cqbit2+1, tqbit+1]
            
        MPS = MPS.permute(permute_shape)
        
        s = list( MPS.shape )
        s[-3] = s[-3]*s[-2]*s[-1]
        s[-2],s[-1] = 1,1
        MPS = MPS.reshape(s)
        MPS = MPS.squeeze(-1)
        
        MPS = matrix @ MPS
        
        MPS = MPS.view([s[0]]+[2]*nqubits)
        
        permute_shape = list( range(nqubits+1) )
        permute_shape.pop()
        permute_shape.pop()
        permute_shape.pop()
        if cqbit1 < cqbit2 and cqbit2 < tqbit:
            permute_shape.insert(cqbit1+1, nqubits-2)
            permute_shape.insert(cqbit2+1, nqubits-1)
            permute_shape.insert(tqbit+1,  nqubits)
        elif cqbit1 < tqbit and tqbit < cqbit2:
            permute_shape.insert(cqbit1+1, nqubits-2)
            permute_shape.insert(tqbit+1,  nqubits)
            permute_shape.insert(cqbit2+1, nqubits-1)
        elif cqbit2 < tqbit and tqbit < cqbit1:
            permute_shape.insert(cqbit2+1, nqubits-1)
            permute_shape.insert(tqbit+1,  nqubits)
            permute_shape.insert(cqbit1+1, nqubits-2)
        elif cqbit2 < cqbit1 and cqbit1 < tqbit:
            permute_shape.insert(cqbit2+1, nqubits-1)
            permute_shape.insert(cqbit1+1, nqubits-2)
            permute_shape.insert(tqbit+1,  nqubits)
        elif tqbit < cqbit1 and cqbit1 < cqbit2:
            permute_shape.insert(tqbit+1,  nqubits)
            permute_shape.insert(cqbit1+1, nqubits-2)
            permute_shape.insert(cqbit2+1, nqubits-1)
        elif tqbit < cqbit2 and cqbit2 < cqbit1:
            permute_shape.insert(tqbit+1,  nqubits)
            permute_shape.insert(cqbit2+1, nqubits-1)
            permute_shape.insert(cqbit1+1, nqubits-2)
        MPS = MPS.permute(permute_shape)
        
        return MPS

#==============================================================================

def _SingleGateLayer_TN_contract(nqubits, wires, lst1, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
    #lst1 = self._cal_single_gates()
    if batch_mod==False:
        for qbit in wires:
            permute_shape = list( range(nqubits) )
            permute_shape[qbit] = nqubits - 1
            permute_shape[nqubits - 1] = qbit
            MPS = MPS.permute(permute_shape).unsqueeze(-1)
            MPS = ( lst1[qbit] @ MPS ).squeeze(-1)
            MPS = MPS.permute(permute_shape)
        return MPS
    else:
        for qbit in wires:
            permute_shape = list( range(nqubits+1) )
            permute_shape[qbit+1] = nqubits
            permute_shape[nqubits] = qbit+1
            MPS = MPS.permute(permute_shape).unsqueeze(-1)
            MPS = ( lst1[qbit] @ MPS ).squeeze(-1)
            MPS = MPS.permute(permute_shape)
        return MPS

# def _ring_of_cnot_TN_contract(nqubits, wires, MPS:torch.Tensor, batch_mod:bool=False)->torch.Tensor:
#     if wires != list( range(nqubits) ):
#         raise ValueError('ring_of_cnot,TN_contract error')
#     L = len(wires)
    
#     if nqubits == 2:
#         MPS = cnot( 2,[wires[0], wires[1]] ).TN_contract(MPS, batch_mod=batch_mod)
#         return MPS
    
#     for i in range(L):
#         cqbit = wires[i]
#         tqbit = wires[(i+1)%L]
#         MPS = cnot(nqubits,[cqbit,tqbit]).TN_contract(MPS, batch_mod=batch_mod)
#     return MPS

# def _ring_of_cnot_dagger_TN_contract():
#     pass

#==============================================================================
def _StateVec2MPS(psi:torch.Tensor, N:int, d:int=2)->List[torch.Tensor]:
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


def _MatrixProductState(psi:torch.Tensor, N:int)->torch.Tensor:
    #t1 = time.time()
    #输入合法性检测：输入的态矢必须是1行2^N列的张量
    if len(psi.shape) != 2:
        raise ValueError('StateVec2MPS:input dimension error!')
    if psi.shape[0] != 1 or psi.shape[1] != 2**N:
        raise ValueError('StateVec2MPS:input shape must be 1 ROW 2^N COLUMN')
    
    MPS = psi.view([2]*N)
    return MPS


def _MPS2StateVec(tensor_lst:List[torch.Tensor],return_sv=True)->torch.Tensor:
    #t1 = time.time()
    N = len(tensor_lst)
    if return_sv == True:
        for i in range(N):
            temp = tensor_lst[i].unsqueeze(0)
            if i == 0:
                c_tensor = tensor_lst[i]
            else:
                c_tensor = c_tensor.unsqueeze(1) @ temp
                shape = c_tensor.shape
                c_tensor = c_tensor.view(shape[0]*shape[1],shape[2],shape[3])
    
        c_tensor = c_tensor.view(-1).view(1,-1)
        return c_tensor #返回1行2^N列的张量，表示态矢的系数
    else:
        for i in range(N):
            temp = tensor_lst[i].unsqueeze(0)
            if i == 0:
                c_tensor = tensor_lst[i]
            else:
                c_tensor = c_tensor.unsqueeze(i) @ temp
        c_tensor = c_tensor.squeeze()
        assert len(c_tensor.shape) == N
        return c_tensor #rank-N张量


def _TensorDecompAfterTwoQbitGate(tensor:torch.Tensor):
    block1 = torch.cat((tensor[0,0],tensor[0,1]),dim=1)
    block2 = torch.cat((tensor[1,0],tensor[1,1]),dim=1)
    tensor = torch.cat((block1,block2),dim=0)
    
    U,S,V = torch.svd( tensor )
    V_d = V.permute(1,0).conj()
    D = len( (S[torch.abs(S)>1e-8]).view(-1) ) #D：bond dimension
    #print('D: ',D)
    # if D==0:
    #     print(S)
    S = torch.diag(S) + 0j
    #默认情况下（不人为设置bond dimension）根据算得的bond dimension对张量进行缩减
    #if bond_dim == -1 and D < S.shape[0]:
    if D < S.shape[0]:
        U = torch.index_select(U, 1, torch.tensor(list(range(D))))
        S = torch.index_select(S, 0, torch.tensor(list(range(D))))
    # elif bond_dim > 0:
    #     #根据人为设置的bond_dim来做分解
    #     U = torch.index_select(U, 1, torch.tensor(list(range(bond_dim))))
    #     S = torch.index_select(S, 0, torch.tensor(list(range(bond_dim))))
    
    rst1 = (U.view(2,-1,U.shape[1]))
    
    rst2 = S @ V_d
    rst2_block = torch.chunk(rst2, chunks=2, dim=1)
    rst2 = torch.cat((rst2_block[0], rst2_block[1]), dim=0)
    rst2 = rst2.view(2,-1,rst2.shape[1])
    
    return rst1,rst2


def _TensorDecompAfterThreeQbitGate(tensor:torch.Tensor):
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


def _MPS_inner_product(ketMPS,braMPS):
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


def _MPS_expec(MPS,wires:List[int],local_obserable:List[torch.Tensor]):
    '''
    local_obserable是一个包含局部力学量(针对单个qubit的力学量，一个2X2矩阵)的list
    wires也是一个list，表明每个局部力学量作用在哪个qubit上
    '''
    SV0 = _MPS2StateVec(MPS).view(1,-1)
    
    for i,qbit in enumerate(wires):
        temp = MPS[qbit]
        temp = temp.permute(1,2,0).unsqueeze(-1) #2421
        temp = torch.squeeze(local_obserable[i] @ temp, dim=3) #242 在指定维度squeeze
        MPS[qbit] = temp.permute(2,0,1)
    expec = SV0.conj() @ _MPS2StateVec(MPS).view(-1,1)
    return expec.squeeze().real
    
#============================================================================

def _Rho2MPS(Rho:torch.Tensor,N:int)->List[torch.Tensor]:
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


def _MPS2Rho(MPS:List[torch.Tensor])->torch.Tensor:
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


def _MatrixProductDensityOperator(rho:torch.Tensor, N:int)->torch.Tensor:
    '''
    以4qubit密度矩阵为例
    按照j1 j2 j3 j4 j1' j2' j3' j4'的对应关系
    把rho变成2X2X2X2X2X2X2X2的高阶张量
    '''
    if len(rho.shape) != 2:
        raise ValueError('MPDO: rho must be matrix(rank-2 tensor)')
    if rho.shape[0] != 2**N or rho.shape[1] != 2**N:
        raise ValueError('MPDO: dimension of rho must be [2^N, 2^N]')
    rho_tensor = rho.view([2]*2*N)
    assert len(rho_tensor.shape) == 2*N
    return rho_tensor