# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 09:18:23 2021

@author: shish
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

from deepquantum.gates.qmath import multi_kron, measure, IsUnitary, IsNormalized
import deepquantum.gates.qoperator as op
from deepquantum.gates.qcircuit import Circuit
#from deepquantum.embeddings.qembedding import PauliEncoding
#from deepquantum.layers.qlayers import YZYLayer, ZXLayer,ring_of_cnot, ring_of_cnot2, BasicEntangleLayer
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,MPS_expec,Rho2MPS,MPS2Rho

# a0 = 1*random.random()+0.5 #产生一个0~1之间的随机数
# a1 = 6*random.random() - 2
# a2 = 6*random.random() - 3
# a3 = 6*random.random() - 3
# a4 = -a1-a2-a3
# print(a0,a1,a2,a3,a4)
a0 = 1.327193070989003
a1 = 1.4274476074643232
a2 = 0.869014090579638
a3 = -2.3608178731399994
a4 = 0.06435617509603819

def rand_func(x,a0=a0,a1=a1,a2=a2,a3=a3,a4=a4):    
    #x = 2*(x - 0.5)
    return 1.5 + a1*(x) + 1*math.sin(4*a0*math.pi*x) + 1 + a2*(x)**2 + a3*(x)**3 + a4*(x)**4

N = 6
epsilon = 1.0/(2**N)
xx = np.arange(epsilon,1+epsilon,epsilon)
yy = [rand_func(x) for x in xx]
assert len(yy) == 2**N
summ = sum(yy)
target = [1.0*y/summ for y in yy]
assert abs( sum(target) - 1.0 ) < 1e-6
for each in target:
    assert each > 0.0
target = torch.tensor(target)

plt.cla()
plt.plot(xx,target,'d--')
plt.show()

#==============================================================================
class qcir(torch.jit.ScriptModule):
    def __init__(self,nqubits):
        super().__init__()
        self.nqubits = nqubits
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(18*self.nqubits), a=0.0, b=2*torch.pi) )
    
    def forward(self):
        wires_lst = [i for i in range(self.nqubits)]
        L = len(wires_lst)
        
        c1 = Circuit(self.nqubits) 
        c1.BasicEntangleLayer(wires_lst, self.weight[0:15*self.nqubits], repeat=5)
        c1.YZYLayer(wires_lst, self.weight[15*self.nqubits:18*self.nqubits])
        # for i in wires_lst:
        #     c1.ry(self.weight[3*i+0], i)
        #     c1.rz(self.weight[3*i+1], i)
        #     c1.ry(self.weight[3*i+2], i)
        # for i in wires_lst:
        #     c1.cnot([i,(i+1)%L])
        # for i in wires_lst:
        #     c1.ry(self.weight[3*self.nqubits+3*i+0], i)
        #     c1.rz(self.weight[3*self.nqubits+3*i+1], i)
        #     c1.ry(self.weight[3*self.nqubits+3*i+2], i)
        # for i in wires_lst:
        #     c1.cnot([i,(i+1)%L])
        # for i in wires_lst:
        #     c1.ry(self.weight[6*self.nqubits+3*i+0], i)
        #     c1.rz(self.weight[6*self.nqubits+3*i+1], i)
        #     c1.ry(self.weight[6*self.nqubits+3*i+2], i)
        # for i in wires_lst:
        #     c1.cnot([i,(i+1)%L])
        # for i in wires_lst:
        #     c1.ry(self.weight[9*self.nqubits+3*i+0], i)
        #     c1.rz(self.weight[9*self.nqubits+3*i+1], i)
        #     c1.ry(self.weight[9*self.nqubits+3*i+2], i)
        psi0 = c1.state_init().view(-1,1)
        
        # MPS = StateVec2MPS(psi0,self.nqubits)
        # MPS = c1.TN_evolution(MPS)
        # sv_f = MPS2StateVec(MPS)
        # print(sv_f)
        
        sv_f = c1.U() @ psi0.view(-1,1)
        sv_f = sv_f.view(1,-1)
        return sv_f

class qnet(torch.jit.ScriptModule):
    
    def __init__(self,nqubits):
        super().__init__()
        self.nqubits = nqubits
        self.circuit = qcir(self.nqubits)
        #self.FC1 = nn.Linear(1, 2**self.nqubits)
    
    def forward(self):
        cir_out = self.circuit()
        m = torch.abs(cir_out.view(-1)).double()
        rst = m*m
        assert len(rst)==2**(self.nqubits)
        return rst
        
net1 = qnet(N)      #构建训练模型
#loss = nn.MSELoss() #平方损失函数
loss = nn.KLDivLoss(reduction='sum')

# print('start producing torchscript file')
# scripted_modeule = torch.jit.script(qnet(N))
# torch.jit.save(scripted_modeule, 'test_torchscript.pt')
# print('completed!')


#定义优化器，也就是选择优化器，选择Adam梯度下降，还是随机梯度下降，或者其他什么
#optimizer = optim.SGD(net1.parameters(), lr=0.001) #lr为学习率
optimizer = optim.Adam(net1.parameters(), lr=0.5) #lr为学习率
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=\
                                              [50,100,150,250,350,500,800,1000,2300], gamma=0.8)

num_epochs = 3000;
#记录loss随着epoch的变化，用于后续绘图
epoch_lst = [i+1 for i in range(num_epochs)]
loss_lst = []

for epoch in range(1,num_epochs+1):
    t1 = time.time()
    
    output = net1()
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    注意KL散度中模型预测值要先取log再输入
    '''
    l = loss(torch.log( output.squeeze() ),target.squeeze())
    #梯度清0
    optimizer.zero_grad() 
    l.backward()
   
    #nn.utils.clip_grad_norm_(net1.circuit.weight,max_norm=1,norm_type=2)
    #print('loss: ',l.item())
    #print("weights_grad2:",net1.circuit.weight.grad,'  weight is leaf?:',net1.circuit.weight.is_leaf)
    # grad = net1.circuit.weight.grad
    # net1.circuit.weight.grad \
    #     = torch.where(torch.isnan(grad),torch.full_like(grad, 0),grad)
    optimizer.step() 
    lr_scheduler.step() #进行学习率的更新
    t2 = time.time()
    loss_lst.append(l.item())
    print("epoch:%d, loss:%f" % (epoch,l.item()),\
          ';current lr:', optimizer.state_dict()["param_groups"][0]["lr"],\
              '   耗时：',t2-t1)
    


print('训练完毕，开始绘图：')
output = net1()
output = [float(o) for o in output]
#print(output)
assert len(output)==len(xx)
plt.cla()
plt.subplot(121)
plt.plot(xx,output,'m^--',linewidth=1, markersize=2)
plt.plot(xx,target,'g^--',linewidth=1, markersize=0.5)

plt.subplot(122)
plt.plot(epoch_lst,loss_lst,'r^--',linewidth=1, markersize=1.5)
plt.show()



input('test_born_machine.py END')