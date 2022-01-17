# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 16:40:23 2022

@author: shish
"""

from deepquantum.gates.qtensornetwork import MatrixProductState
from deepquantum.gates.qcircuit import Circuit

n_qubits = 4
cir = Circuit(n_qubits)

psi0 = cir.state_init().view(1,-1)
MPS = MatrixProductState(psi0, cir.nqubits)

cir.rx(0.1, 0)
cir.ry(0.1, 2)
cir.PauliX(3)
cir.cnot([2,1])
cir.rz(0.85, 3)
cir.ring_of_cnot( list(range(cir.nqubits)) )

MPS = cir.TN_contract_evolution(MPS)
psif = MPS.reshape(1,-1)
# print(psif)
# print( (cir.U()@psi0.view(-1,1) ).reshape(1,-1))
# input('END')

