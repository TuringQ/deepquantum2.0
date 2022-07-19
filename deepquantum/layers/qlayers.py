# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:06:10 2021

@author: shish
"""
import torch
import torch.nn as nn
import time
from typing import List
import copy

from deepquantum.gates import multi_kron
from deepquantum.gates.qoperator import Hadamard, rx, ry, rz, rxx, ryy, rzz, cnot, cz, SWAP, Operation
from deepquantum.gates.qtensornetwork import StateVec2MPS, MPS2StateVec, TensorDecompAfterTwoQbitGate
from deepquantum.gates.qTN_contract import _SingleGateLayer_TN_contract, _SingleGateLayer_TN_contract_Rho,  \
    _cnot_TN_contract, _TwoQbitGate_TN_contract_Rho


'''
所有layer必须有label标签，nqubits比特数，wires涉及哪几个比特，num_params参数数目，是否支持张量网络
self.label = "HadamardLayer"
self.nqubits = N
self.wires = wires
self.num_params = 0
self.supportTN = True
'''


class SingleGateLayer(Operation):
    '''
    单比特层的父类
    '''

    def __init__(self):
        self.nqubits = 4
        self.wires = []
        self.supportTN = True
        self.dev = 'cpu'

    def _cal_single_gates(self) -> List[torch.Tensor]:
        lst1 = [torch.eye(2, 2)] * self.nqubits
        return lst1

    def TN_operation(self, MPS: List[torch.Tensor]) -> List[torch.Tensor]:
        lst1 = self._cal_single_gates()
        for qbit in self.wires:
            # temp = MPS[qbit]
            # temp = temp.permute(1,2,0).unsqueeze(-1) #2421
            # temp = torch.squeeze(lst1[qbit] @ temp, dim=3) #242 在指定维度squeeze
            # MPS[qbit] = temp.permute(2,0,1)

            MPS[qbit] = torch.einsum('ab,bcd->acd', [lst1[qbit], MPS[qbit]])

        return MPS

    def TN_contract(self, MPS: torch.Tensor, batch_mod: bool = False) -> torch.Tensor:
        if self.dev == "gpu" or self.dev == "cuda":
            assert MPS.is_cuda, "------MPS must be on-cuda-----"
        lst1 = self._cal_single_gates()
        # if batch_mod==False:
        #     for qbit in self.wires:
        #         permute_shape = list( range(self.nqubits) )
        #         permute_shape[qbit] = self.nqubits - 1
        #         permute_shape[self.nqubits - 1] = qbit
        #         MPS = MPS.permute(permute_shape).unsqueeze(-1)
        #         MPS = ( lst1[qbit] @ MPS ).squeeze(-1)
        #         MPS = MPS.permute(permute_shape)
        #     return MPS
        # else:
        #     for qbit in self.wires:
        #         permute_shape = list( range(self.nqubits+1) )
        #         permute_shape[qbit+1] = self.nqubits
        #         permute_shape[self.nqubits] = qbit+1
        #         MPS = MPS.permute(permute_shape).unsqueeze(-1)
        #         MPS = ( lst1[qbit] @ MPS ).squeeze(-1)
        #         MPS = MPS.permute(permute_shape)
        #     return MPS
        MPS = _SingleGateLayer_TN_contract(self.nqubits, self.wires, lst1, MPS, batch_mod=batch_mod)
        return MPS

    def TN_contract_Rho(self, MPDO: torch.Tensor, batch_mod: bool = False) -> torch.Tensor:
        if self.dev == "gpu" or self.dev == "cuda":
            assert MPDO.is_cuda, "------MPDO must be on-cuda-----"
        lst1 = self._cal_single_gates()
        MPDO = _SingleGateLayer_TN_contract_Rho(self.nqubits, self.wires, lst1, MPDO, batch_mod=batch_mod)
        return MPDO


class TwoQbitGateLayer(Operation):
    '''
    两比特层的父类
    '''

    def __init__(self):
        pass


class XYZLayer(SingleGateLayer):
    # label = "XYZLayer"

    def __init__(self, N, wires, params_lst, dev='cpu'):
        if 3 * len(wires) != len(params_lst):
            raise ValueError("XYZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XYZLayer: number of wires must less than N")
        self.label = "XYZLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.n_wires = len(wires)
        # 如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True

    def _cal_single_gates(self):
        lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        X = torch.tensor([[0, 1], [1, 0]]).to(self.device)
        Y = torch.tensor([[0, -1j], [1j, 0]]).to(self.device)
        Z = torch.tensor([[1, 0], [0, -1]]).to(self.device)
        I = torch.eye(2, 2).to(self.device)

        # c = torch.cos(0.5 * self.params).to(self.device).reshape(-1, 3).permute(1, 0)
        # s = torch.sin(0.5 * self.params).to(self.device).reshape(-1, 3).permute(1, 0)
        # cx, cy, cz = c[0], c[1], c[2]
        # sx, sy, sz = s[0], s[1], s[2]

        c = torch.cos(0.5 * self.params).to(self.device)
        s = torch.sin(0.5 * self.params).to(self.device)
        cx, cy, cz = c[:self.n_wires], c[self.n_wires:2*self.n_wires], c[2*self.n_wires:]
        sx, sy, sz = s[:self.n_wires], s[self.n_wires:2*self.n_wires], s[2*self.n_wires:]

        xms = torch.kron(cx, I) - 1j * torch.kron(sx, X)
        yms = torch.kron(cy, I) - 1j * torch.kron(sy, Y)
        zms = torch.kron(cz, I) - 1j * torch.kron(sz, Z)
        xms = xms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        yms = yms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        zms = zms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)

        rst = zms @ yms @ xms
        lst1[:] = rst[:]                # 暂只适用于 n_wires == n_qubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        params_tensor = -1 * self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy(params_tensor[3 * i + 0])
            params_tensor[3 * i + 0] = params_tensor[3 * i + 2]
            params_tensor[3 * i + 2] = temp
        return ZYXLayer(self.nqubits, self.wires, params_tensor)

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)


class ZYXLayer(SingleGateLayer):
    # label = "ZYXLayer"

    def __init__(self, N, wires, params_lst, dev='cpu'):
        if 3 * len(wires) != len(params_lst):
            raise ValueError("ZYXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("ZYXLayer: number of wires must less than N")
        self.label = "ZYXLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.n_wires = len(wires)
        # 如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True

    def _cal_single_gates(self):
        lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        X = torch.tensor([[0, 1], [1, 0]]).to(self.device)
        Y = torch.tensor([[0, -1j], [1j, 0]]).to(self.device)
        Z = torch.tensor([[1, 0], [0, -1]]).to(self.device)
        I = torch.eye(2, 2).to(self.device)

        c = torch.cos(0.5 * self.params).to(self.device)
        s = torch.sin(0.5 * self.params).to(self.device)
        cz, cy, cx = c[:self.n_wires], c[self.n_wires:2*self.n_wires], c[2*self.n_wires:]
        sz, sy, sx = s[:self.n_wires], s[self.n_wires:2*self.n_wires], s[2*self.n_wires:]

        zms = torch.kron(cz, I) - 1j * torch.kron(sz, Z)
        yms = torch.kron(cy, I) - 1j * torch.kron(sy, Y)
        xms = torch.kron(cx, I) - 1j * torch.kron(sx, X)
        zms = zms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        yms = yms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        xms = xms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)

        rst = xms @ yms @ zms
        lst1[:] = rst[:]                # 暂只适用于 n_wires == n_qubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        params_tensor = -1 * self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy(params_tensor[3 * i + 0])
            params_tensor[3 * i + 0] = params_tensor[3 * i + 2]
            params_tensor[3 * i + 2] = temp
        return XYZLayer(self.nqubits, self.wires, params_tensor)

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)


class ZYZLayer(SingleGateLayer):
    """
    zyz layer
    """
    def __init__(self, N, wires, params_lst, dev='cpu'):
        if 3 * len(wires) != len(params_lst):
            raise ValueError("ZYZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("ZYZLayer: number of wires must less than N")
        self.label = "ZYZLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.n_wires = len(wires)
        # 如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True

    def _cal_single_gates(self):
        lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        Y = torch.tensor([[0, -1j], [1j, 0]]).to(self.device)
        Z = torch.tensor([[1, 0], [0, -1]]).to(self.device)
        I = torch.eye(2, 2).to(self.device)

        c = torch.cos(0.5 * self.params).to(self.device)
        s = torch.sin(0.5 * self.params).to(self.device)
        cz1, cy, cz2 = c[:self.n_wires], c[self.n_wires:2*self.n_wires], c[2*self.n_wires:]
        sz1, sy, sz2 = s[:self.n_wires], s[self.n_wires:2*self.n_wires], s[2*self.n_wires:]

        z1ms = torch.kron(cz1, I) - 1j * torch.kron(sz1, Z)
        yms = torch.kron(cy, I) - 1j * torch.kron(sy, Y)
        z2ms = torch.kron(cz2, I) - 1j * torch.kron(sz2, Z)
        z1ms = z1ms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        yms = yms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        z2ms = z2ms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)

        rst = z2ms @ yms @ z1ms
        lst1[:] = rst[:]                # 暂只适用于 n_wires == n_qubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        pass

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        pass


class YZYLayer(SingleGateLayer):
    # label = "YZYLayer"

    def __init__(self, N, wires, params_lst, dev='cpu'):
        if 3 * len(wires) != len(params_lst):
            raise ValueError("YZYLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("YZYLayer: number of wires must less than N")
        self.label = "YZYLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.n_wires = len(wires)
        # 如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True

    def _cal_single_gates(self):
        lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        Y = torch.tensor([[0, -1j], [1j, 0]]).to(self.device)
        Z = torch.tensor([[1, 0], [0, -1]]).to(self.device)
        I = torch.eye(2, 2).to(self.device)

        c = torch.cos(0.5 * self.params).to(self.device)
        s = torch.sin(0.5 * self.params).to(self.device)
        cy1, cz, cy2 = c[:self.n_wires], c[self.n_wires:2*self.n_wires], c[2*self.n_wires:]
        sy1, sz, sy2 = s[:self.n_wires], s[self.n_wires:2*self.n_wires], s[2*self.n_wires:]

        y1ms = torch.kron(cy1, I) - 1j * torch.kron(sy1, Y)
        zms = torch.kron(cz, I) - 1j * torch.kron(sz, Z)
        y2ms = torch.kron(cy2, I) - 1j * torch.kron(sy2, Y)
        y1ms = y1ms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        zms = zms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        y2ms = y2ms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)

        rst = y2ms @ zms @ y1ms
        lst1[:] = rst[:]                # 暂只适用于 n_wires == n_qubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        params_tensor = -self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy(params_tensor[3 * i + 0])
            params_tensor[3 * i + 0] = params_tensor[3 * i + 2]
            params_tensor[3 * i + 2] = temp
        return YZYLayer(self.nqubits, self.wires, params_tensor)

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)


class XZXLayer(SingleGateLayer):
    # label = "XZXLayer"

    def __init__(self, N, wires, params_lst, dev='cpu'):
        if 3 * len(wires) != len(params_lst):
            raise ValueError("XZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZXLayer: number of wires must less than N")
        self.label = "XZXLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.n_wires = len(wires)
        # 如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True

    def _cal_single_gates(self):
        lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        X = torch.tensor([[0, 1], [1, 0]]).to(self.device)
        Z = torch.tensor([[1, 0], [0, -1]]).to(self.device)
        I = torch.eye(2, 2).to(self.device)

        c = torch.cos(0.5 * self.params).to(self.device)
        s = torch.sin(0.5 * self.params).to(self.device)
        cx1, cz, cx2 = c[:self.n_wires], c[self.n_wires:2*self.n_wires], c[2*self.n_wires:]
        sx1, sz, sx2 = s[:self.n_wires], s[self.n_wires:2*self.n_wires], s[2*self.n_wires:]

        x1ms = torch.kron(cx1, I) - 1j * torch.kron(sx1, X)
        x1ms = x1ms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        zms = torch.kron(cz, I) - 1j * torch.kron(sz, Z)
        zms = zms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        x2ms = torch.kron(cx2, I) - 1j * torch.kron(sx2, X)
        x2ms = x2ms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)

        rst = x2ms @ zms @ x1ms
        lst1[:] = rst[:]                # 暂只适用于 n_wires == n_qubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        params_tensor = -1 * self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy(params_tensor[3 * i + 0])
            params_tensor[3 * i + 0] = params_tensor[3 * i + 2]
            params_tensor[3 * i + 2] = temp
        return XZXLayer(self.nqubits, self.wires, params_tensor)

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)


class XZLayer(SingleGateLayer):
    # label = "XZLayer"

    def __init__(self, N, wires, params_lst, dev='cpu'):
        if 2 * len(wires) != len(params_lst):
            raise ValueError("XZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZLayer: number of wires must less than N")
        self.label = "XZLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.n_wires = len(wires)
        # 如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True

    def _cal_single_gates(self):
        lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        X = torch.tensor([[0, 1], [1, 0]]).to(self.device)
        Z = torch.tensor([[1, 0], [0, -1]]).to(self.device)
        I = torch.eye(2, 2).to(self.device)

        c = torch.cos(0.5 * self.params).to(self.device)
        s = torch.sin(0.5 * self.params).to(self.device)
        cx, cz = c[:self.n_wires], c[self.n_wires:]
        sx, sz = s[:self.n_wires], s[self.n_wires:]

        xms = torch.kron(cx, I) - 1j * torch.kron(sx, X)
        xms = xms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        zms = torch.kron(cz, I) - 1j * torch.kron(sz, Z)
        zms = zms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)

        rst = zms @ xms
        lst1[:] = rst[:]                # 暂只适用于 n_wires == n_qubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        params_tensor = -1 * self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy(params_tensor[2 * i + 0])
            params_tensor[2 * i + 0] = params_tensor[2 * i + 1]
            params_tensor[2 * i + 1] = temp
        return ZXLayer(self.nqubits, self.wires, params_tensor)

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)


class ZXLayer(SingleGateLayer):
    # label = "ZXLayer"

    def __init__(self, N, wires, params_lst, dev='cpu'):
        if 2 * len(wires) != len(params_lst):
            raise ValueError("ZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("ZXLayer: number of wires must less than N")
        self.label = "ZXLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.n_wires = len(wires)
        # 如果是列表，先转成tensor
        if type(params_lst) == type([1]):
            params_lst = torch.tensor(params_lst)
        self.params = params_lst
        self.num_params = len(params_lst)
        self.supportTN = True

    def _cal_single_gates(self):
        lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        X = torch.tensor([[0, 1], [1, 0]]).to(self.device)
        Z = torch.tensor([[1, 0], [0, -1]]).to(self.device)
        I = torch.eye(2, 2).to(self.device)

        c = torch.cos(0.5 * self.params).to(self.device)
        s = torch.sin(0.5 * self.params).to(self.device)
        cz, cx = c[:self.n_wires], c[self.n_wires:]
        sz, sx = s[:self.n_wires], s[self.n_wires:]

        zms = torch.kron(cz, I) - 1j * torch.kron(sz, Z)
        zms = zms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)
        xms = torch.kron(cx, I) - 1j * torch.kron(sx, X)
        xms = xms.permute(1, 0).reshape(-1, 2, 2).permute(0, 2, 1)

        rst = xms @ zms
        lst1[:] = rst[:]                # 暂只适用于 n_wires == n_qubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        params_tensor = -1 * self.params
        for i in range(len(self.wires)):
            temp = copy.deepcopy(params_tensor[2 * i + 0])
            params_tensor[2 * i + 0] = params_tensor[2 * i + 1]
            params_tensor[2 * i + 1] = temp
        return XZLayer(self.nqubits, self.wires, params_tensor)

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        self.params = params_lst
        self.num_params = len(params_lst)


class HLayer(SingleGateLayer):
    # label = "HadamardLayer"

    def __init__(self, N, wires, dev='cpu'):
        if len(wires) > N:
            raise ValueError("HadamardLayer: number of wires must less than N")

        self.label = "HadamardLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = True

    def _cal_single_gates(self):
        # lst1 = [torch.eye(2, 2).to(self.device)] * self.nqubits
        # H = (torch.sqrt(torch.tensor(0.5)) * torch.tensor([[1, 1], [1, -1]])).to(self.device) + 0j
        # for i, qbit in enumerate(self.wires):
        #     # lst1[qbit] = Hadamard().matrix
        #     lst1[qbit] = H

        H = torch.sqrt(torch.tensor(0.5)) * torch.tensor([[1, 1], [1, -1]]).to(self.device) + 0j
        lst1 = [H] * self.nqubits
        return lst1

    def U_expand(self):
        lst1 = self._cal_single_gates()
        return multi_kron(lst1) + 0j

    def operation_dagger(self):
        return self

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': None}
        return info

    def params_update(self, params_lst):
        pass


# ==============================================================================


class ring_of_cnot(TwoQbitGateLayer):
    # label = "ring_of_cnot_Layer"

    def __init__(self, N, wires, dev='cpu'):
        # ladderdown=True表示用下降的阶梯式排列ring of cnot

        if len(wires) > N:
            raise ValueError("ring_of_cnotLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnotLayer: number of wires must >= 2")
        self.label = "ring_of_cnot_Layer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = True
        self.matrix = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).to(self.device) + 0j

    def _gate_fusion_U_expand(self, N):
        if N < 3:
            raise ValueError('ring of cnot : gate_fusion error! N must be >= 3')
        I = torch.eye(2, 2).to(self.device) + 0j
        rst = cnot(2, [0, 1], self.dev).U_expand()
        for i in range(1, N):
            cur_M = cnot(min(2 + i, N), [i, (i + 1) % N], self.dev).U_expand()
            if i == N - 1:
                rst = cur_M @ rst
            else:
                rst = cur_M @ torch.kron(rst, I)
        return rst

    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot(self.nqubits, [self.wires[0], self.wires[1]], self.dev).U_expand()

        if self.wires == list(range(self.nqubits)):
            return self._gate_fusion_U_expand(self.nqubits)

        rst = torch.eye(2 ** self.nqubits, 2 ** self.nqubits).to(self.device) + 0j
        for i, qbit in enumerate(self.wires):
            # if i == L-1: #临时加的
            #     break
            # rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+1)%L] ]).U_expand() @ I
            rst = cnot(self.nqubits, [self.wires[i], self.wires[(i + 1) % L]], self.dev).U_expand() @ rst

        return rst

    def TN_operation(self, MPS: List[torch.Tensor]) -> List[torch.Tensor]:
        '''
        只支持自上而下的cnot，即上方的qbit一定是control，下方的一定是target
        '''
        if self.wires != list(range(self.nqubits)):
            raise ValueError('ring_of_cnot,TN_operation error')
        L = len(self.wires)
        for i in range(L - 1):
            MPS = cnot(self.nqubits, [i, i + 1]).TN_operation(MPS)
        if self.nqubits == 2:
            return MPS
        # ======================================================================
        for i in range(L - 1):
            '''
            试着直接用非邻近cnot门，不要用SWAP了
            '''
            if i != L - 2:
                MPS = SWAP(self.nqubits, [i, i + 1]).TN_operation(MPS)
                # temp = SWAP().matrix @ temp
            else:
                MPS = cnot(self.nqubits, [i + 1, i]).TN_operation(MPS)
        for i in range(L - 3, -1, -1):
            MPS = SWAP(self.nqubits, [i, i + 1]).TN_operation(MPS)
        return MPS

    def TN_contract(self, MPS: torch.Tensor, batch_mod: bool = False) -> torch.Tensor:
        if self.dev == "gpu" or self.dev == "cuda":
            assert MPS.is_cuda, "------MPS must be on-cuda-----"

        if self.wires != list(range(self.nqubits)):
            raise ValueError('ring_of_cnot, TN_contract error')

        L = len(self.wires)

        if self.nqubits == 2:
            # MPS = cnot(2, [self.wires[0], self.wires[1]], self.dev).TN_contract(MPS, batch_mod=batch_mod)
            MPS = _cnot_TN_contract(2, [self.wires[0], self.wires[1]], self.matrix, MPS, batch_mod)
            return MPS

        for i in range(L):
            cqbit = self.wires[i]
            tqbit = self.wires[(i + 1) % L]
            # MPS = cnot(self.nqubits, [cqbit, tqbit], self.dev).TN_contract(MPS, batch_mod=batch_mod)
            MPS = _cnot_TN_contract(self.nqubits, [cqbit, tqbit], self.matrix, MPS, batch_mod)

        return MPS

    def TN_contract_Rho(self, MPDO: torch.Tensor, batch_mod: bool = False) -> torch.Tensor:
        if self.dev == "gpu" or self.dev == "cuda":
            assert MPDO.is_cuda, "------MPDO must be on-cuda-----"

        if self.wires != list(range(self.nqubits)):
            raise ValueError('ring_of_cnot,TN_contract_Rho error')

        L = len(self.wires)

        if self.nqubits == 2:
            # MPDO = cnot(2, [self.wires[0], self.wires[1]], self.dev).TN_contract_Rho(MPDO, batch_mod=batch_mod)
            MPDO = _TwoQbitGate_TN_contract_Rho(2, [self.wires[0], self.wires[1]], self.matrix, MPDO, batch_mod)
            return MPDO

        for i in range(L):
            cqbit = self.wires[i]
            tqbit = self.wires[(i + 1) % L]
            # MPDO = cnot(self.nqubits, [cqbit, tqbit], self.dev).TN_contract_Rho(MPDO, batch_mod=batch_mod)
            MPDO = _TwoQbitGate_TN_contract_Rho(self.nqubits, [cqbit, tqbit], self.matrix, MPDO, batch_mod)
        return MPDO

    def operation_dagger(self):
        return ring_of_cnot_dagger(self.nqubits, self.wires)

    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i + 1) % L] for i in range(L)]
        if L == 2:
            info = {'label': self.label, 'contral_lst': [self.wires[0]], 'target_lst': [self.wires[1]], 'params': None}
        else:
            info = {'label': self.label, 'contral_lst': self.wires, 'target_lst': target_lst, 'params': None}
        return info

    def params_update(self, params_lst):
        pass


class ring_of_cnot_dagger(TwoQbitGateLayer):
    '''
    这是上面ring_of_cnot layer的转置共轭算符，比如一个5qubit线路的ring_of_cnot_dagger
    本质就是4控0,3控4，2控3，1控2，0控1依次的五个cnot
    '''

    def __init__(self, N, wires, dev='cpu'):
        # ladderdown=True表示用下降的阶梯式排列ring of cnot

        if len(wires) > N:
            raise ValueError("ring_of_cnot_dagger: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnot_dagger: number of wires must >= 2")
        self.label = "ring_of_cnot_dagger"
        self.dev = dev
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = True
        device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.matrix = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).to(device) + 0j

    def _gate_fusion_U_expand(self, N):
        if N < 3:
            raise ValueError('ring_of_cnot_dagger : gate_fusion error! N must be >= 3')
        I = torch.eye(2, 2) + 0j
        rst = cnot(2, [0, 1]).U_expand()
        for i in range(1, N):
            cur_M = cnot(min(2 + i, N), [i, (i + 1) % N]).U_expand()
            if i == N - 1:
                # rst = cur_M @ rst
                rst = rst @ cur_M
            else:
                # rst = cur_M @ torch.kron(rst,I)
                rst = torch.kron(rst, I) @ cur_M
        return rst

    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot(self.nqubits, [self.wires[0], self.wires[1]]).U_expand()

        if self.wires == list(range(self.nqubits)):
            return self._gate_fusion_U_expand(self.nqubits)

        rst = torch.eye(2 ** self.nqubits, 2 ** self.nqubits) + 0j
        for i, qbit in enumerate(self.wires):
            rst = rst @ cnot(self.nqubits, [self.wires[i], self.wires[(i + 1) % L]]).U_expand()

        return rst

    def TN_operation(self, MPS: List[torch.Tensor]) -> List[torch.Tensor]:

        if self.wires != list(range(self.nqubits)):
            raise ValueError('ring_of_cnot_dagger,TN_operation error')

        L = len(self.wires)

        if self.nqubits == 2:
            MPS = cnot(2, [0, 1]).TN_operation(MPS)
            return MPS

        for i in range(L - 1):
            if i != L - 2:
                MPS = SWAP(self.nqubits, [i, i + 1]).TN_operation(MPS)
            else:
                MPS = cnot(self.nqubits, [i + 1, i]).TN_operation(MPS)
        for i in range(L - 3, -1, -1):
            MPS = SWAP(self.nqubits, [i, i + 1]).TN_operation(MPS)

        for i in range(L - 2, -1, -1):
            MPS = cnot(self.nqubits, [i, i + 1]).TN_operation(MPS)

        return MPS

    def TN_contract(self, MPS: torch.Tensor, batch_mod: bool = False) -> torch.Tensor:
        if self.dev == "gpu" or self.dev == "cuda":
            assert MPS.is_cuda, "------MPS must be on-cuda-----"

        if self.wires != list(range(self.nqubits)):
            raise ValueError('ring_of_cnot_dagger,TN_contract error')
        L = len(self.wires)

        if self.nqubits == 2:
            # MPS = cnot(2, [self.wires[0], self.wires[1]], self.dev).TN_contract(MPS, batch_mod=batch_mod)
            MPS = _cnot_TN_contract(2, [self.wires[0], self.wires[1]], self.matrix, MPS, batch_mod)
            return MPS

        for i in range(L - 1, -1, -1):
            cqbit = self.wires[i]
            tqbit = self.wires[(i + 1) % L]
            # MPS = cnot(self.nqubits, [cqbit, tqbit], self.dev).TN_contract(MPS, batch_mod=batch_mod)
            MPS = _cnot_TN_contract(self.nqubits, [cqbit, tqbit], self.matrix, MPS, batch_mod)
        return MPS

    def TN_contract_Rho(self, MPDO: torch.Tensor, batch_mod: bool = False) -> torch.Tensor:
        if self.dev == "gpu" or self.dev == "cuda":
            assert MPDO.is_cuda, "------MPS must be on-cuda-----"

        if self.wires != list(range(self.nqubits)):
            raise ValueError('ring_of_cnot_dagger,TN_contract_Rho error')

        L = len(self.wires)

        if self.nqubits == 2:
            # MPDO = cnot(2, [self.wires[0], self.wires[1]], self.dev).TN_contract_Rho(MPDO, batch_mod=batch_mod)
            MPDO = _TwoQbitGate_TN_contract_Rho(2, [self.wires[0], self.wires[1]], self.matrix, MPDO, batch_mod)
            return MPDO

        for i in range(L - 1, -1, -1):
            cqbit = self.wires[i]
            tqbit = self.wires[(i + 1) % L]
            # MPDO = cnot(self.nqubits, [cqbit, tqbit], self.dev).TN_contract_Rho(MPDO, batch_mod=batch_mod)
            MPDO = _TwoQbitGate_TN_contract_Rho(self.nqubits, [cqbit, tqbit], self.matrix, MPDO, batch_mod)
        return MPDO

    def operation_dagger(self):
        return ring_of_cnot(self.nqubits, self.wires)

    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i + 1) % L] for i in range(L)]
        if L == 2:
            info = {'label': self.label, 'contral_lst': [self.wires[0]], 'target_lst': [self.wires[1]], 'params': None}
        else:
            info = {'label': self.label, 'contral_lst': self.wires, 'target_lst': target_lst, 'params': None}
        return info

    def params_update(self, params_lst):
        pass


class ring_of_cnot2(TwoQbitGateLayer):
    # label = "ring_of_cnot2_Layer"

    def __init__(self, N, wires, dev='cpu'):

        if len(wires) > N:
            raise ValueError("ring_of_cnot2Layer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnot2Layer: number of wires must >= 2")
        self.label = "ring_of_cnot2_Layer"
        self.dev = dev
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        self.supportTN = False

    def _gate_fusion_U_expand(self, N):
        if N < 3:
            raise ValueError('ring of cnot : gate_fusion error! N must be >= 3')
        I = torch.eye(2, 2) + 0j
        rst = cnot(3, [0, 2]).U_expand()
        for i in range(1, N):
            cur_M = cnot(min(3 + i, N), [i, (i + 2) % N]).U_expand()
            if i == N - 2 or i == N - 1:
                rst = cur_M @ rst
            else:
                rst = cur_M @ torch.kron(rst, I)
        return rst

    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot(self.nqubits, [self.wires[0], self.wires[1]]).U_expand()

        if self.wires == list(range(self.nqubits)):
            return self._gate_fusion_U_expand(self.nqubits)

        # I = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        rst = torch.eye(2 ** self.nqubits, 2 ** self.nqubits) + 0j
        for i, qbit in enumerate(self.wires):
            rst = cnot(self.nqubits, [self.wires[i], self.wires[(i + 2) % L]]).U_expand() @ rst

        return rst

    def operation_dagger(self):
        pass

    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i + 2) % L] for i in range(L)]
        if L == 2:
            info = {'label': self.label, 'contral_lst': [self.wires[0]], 'target_lst': [self.wires[1]], 'params': None}
        else:
            info = {'label': self.label, 'contral_lst': self.wires, 'target_lst': target_lst, 'params': None}
        return info

    def params_update(self, params_lst):
        pass


# =========================================================================================


class BasicEntangleLayer(TwoQbitGateLayer):
    # label = "BasicEntangleLayer"

    def __init__(self, N, wires, params_lst, repeat=1, dev='cpu'):

        if 3 * len(wires) * repeat != len(params_lst):
            raise ValueError("BasicEntangleLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("BasicEntangleLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("BasicEntangleLayer: number of wires must >= 2")
        if repeat < 1:
            raise ValueError("BasicEntangleLayer: number of repeat must >= 1")
        self.label = "BasicEntangleLayer"
        self.dev = dev
        self.device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
        self.nqubits = N
        self.wires = wires
        self.num_params = len(params_lst)
        self.params = params_lst
        self.repeat = repeat

        self.part1_lst, self.part2_lst = [], []
        for i in range(int(self.repeat)):
            self.part1_lst.append(
                YZYLayer(self.nqubits, self.wires, self.params[i * 3 * len(wires):(i + 1) * 3 * len(wires)], self.dev))
            self.part2_lst.append(ring_of_cnot(self.nqubits, self.wires, self.dev))

        self.supportTN = False

    def U_expand(self):
        rst = torch.eye(2 ** self.nqubits).to(self.device) + 0j
        cnot_ring = self.part2_lst[0].U_expand()
        for i in range(self.repeat):
            # rst = self.part2_lst[i].U_expand() @ self.part1_lst[i].U_expand() @ rst
            rst = cnot_ring @ self.part1_lst[i].U_expand() @ rst
        return rst

    def operation_dagger(self):
        pass

    def info(self):
        info = {'label': self.label, 'contral_lst': [], 'target_lst': self.wires, 'params': self.params}
        return info

    def params_update(self, params_lst):
        self.num_params = len(params_lst)
        self.params = params_lst
        self.part1_lst, self.part2_lst = [], []
        L = 3 * len(self.wires)
        for i in range(self.repeat):
            self.part1_lst.append(YZYLayer(self.nqubits, self.wires, self.params[i * L:(i + 1) * L]))
            self.part2_lst.append(ring_of_cnot(self.nqubits, self.wires))


if __name__ == '__main__':
    print('start')
    N = 4
    wires = list(range(N))
    # wires = [0,1,2,3]
    '''
    验证两比特门MPS作用的正确性与效率提升
    '''
    r1 = ring_of_cnot(N, wires)

    psi = torch.zeros(1, 2 ** N) + 0.0j
    psi[0, 0] = 1.0 + 0j;
    psi[0, -1] = 0.50 + 0j
    psi = nn.functional.normalize(psi, p=2, dim=1)
    # psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )

    psif = (r1.U_expand() @ psi.permute(1, 0)).permute(1, 0)

    MPS = StateVec2MPS(psi, N)
    MPS = r1.TN_operation(MPS)
    psi1 = MPS2StateVec(MPS).view(1, -1)
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
