# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 18:39:10 2022

@author: shish
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import math
import time
import copy
#import onnx

import deepquantum as dq
from deepquantum.gates.qmath import multi_kron, measure, IsUnitary, IsNormalized
import deepquantum.gates.qoperator as op
from deepquantum.gates.qcircuit import Circuit,parameter_shift
from deepquantum.embeddings.qembedding import PauliEncoding
from deepquantum.layers.qlayers import YZYLayer, ZXLayer,ring_of_cnot, ring_of_cnot2, BasicEntangleLayer
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,\
    TensorDecompAfterTwoQbitGate,TensorDecompAfterThreeQbitGate


'''
6qubit时，耗时一样，但是TN占用CPU仅40%，对比70%
8qubit时，耗时只有一半
10qubit时，耗时仅1/30
'''



#==============================================================================
class qcir(torch.jit.ScriptModule):
    def __init__(self,nqubits):
        super().__init__()
        #属性：量子线路qubit数目，随机初始化的线路参数，测量力学量列表
        self.nqubits = nqubits
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(21*self.nqubits), a=0.0, b=2*torch.pi) )
        
        self.M_lst = self.Zmeasure()

        
    def Zmeasure(self):
        #生成测量力学量的列表
        M_lst = []
        for i in range(self.nqubits):
            Mi = op.PauliZ(self.nqubits,i).U_expand()
            M_lst.append( Mi )
        
        return M_lst
            
    
    def forward2(self,input_lst_batch):
        '''
        构建变分量子线路的演化矩阵
        '''
        
        wires_lst = list( range(self.nqubits) )
        #创建线路
        c1 = Circuit(self.nqubits) 
        psi0 = c1.state_init().view(1,-1)
        
        MPS = StateVec2MPS(psi0,self.nqubits)
        MPSc = MPS2StateVec(MPS,return_sv=False)
        #encoding编码部分
        MPS_batch = []
        for i in range(len(input_lst_batch)):
            PE = PauliEncoding(self.nqubits, input_lst_batch[i], wires_lst)
            MPS_f = PE.TN_contract(copy.deepcopy(MPSc))
            MPS_batch.append(MPS_f)
            
        c1 = Circuit(self.nqubits)
        c1.YZYLayer(wires_lst, self.weight[0:3*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[3*self.nqubits:6*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[6*self.nqubits:9*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[9*self.nqubits:12*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[12*self.nqubits:15*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[15*self.nqubits:18*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[18*self.nqubits:21*self.nqubits])
        
        for i in range(len(MPS_batch)):
            MPSt = c1.TN_contract_evolution(MPS_batch[i])
            temp = torch.clone(MPSt)
            #print(MPSt.requires_grad)
            MPStz = op.PauliZ(self.nqubits,0).TN_contract(MPSt)
            expec = (temp.reshape(1,-1).conj() @ MPStz.reshape(-1,1) ).real
            
            if i == 0:
                rst = expec.view(1,-1)
            else:
                rst = torch.cat( (rst, expec.view(1,-1)), dim=0 )
            #print(rst.requires_grad)
        return rst
        #======================================================================
    
    def forward3(self,input_lst_batch):
        '''
        构建变分量子线路的演化矩阵
        '''
        
        wires_lst = list( range(self.nqubits) )
        #创建线路
        c1 = Circuit(self.nqubits) 
        psi0 = c1.state_init().view(1,-1)
        
        #encoding编码部分
        psi_batch = []
        for i in range(len(input_lst_batch)):
            PE = PauliEncoding(self.nqubits, input_lst_batch[i], wires_lst)
            psi_e = (PE.U_expand() @ psi0.view(-1,1)).view(1,-1)
            psi_batch.append(psi_e)
            
        c1 = Circuit(self.nqubits)
        c1.YZYLayer(wires_lst, self.weight[0:3*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[3*self.nqubits:6*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[6*self.nqubits:9*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[9*self.nqubits:12*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[12*self.nqubits:15*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[15*self.nqubits:18*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[18*self.nqubits:21*self.nqubits])
        
        
        for i in range(len(psi_batch)):
            psif = c1.U() @ psi_batch[i].view(-1,1)
            expec = psif.view(1,-1).conj() @ op.PauliZ(self.nqubits,0).U_expand() @ psif
            expec = expec.real
            if i == 0:
                rst = expec.view(1,-1)
            else:
                rst = torch.cat( (rst, expec.view(1,-1)), dim=0 )
            #print(rst.requires_grad)
        return rst
    
    def forward1(self,input_lst_batch):
        
        wires_lst = list( range(self.nqubits) )
        
        c1 = Circuit(self.nqubits) 
        psi0 = c1.state_init().view(1,-1)
        psi_batch = []
        for i in range(len(input_lst_batch)):
            PE = PauliEncoding(self.nqubits, input_lst_batch[i], wires_lst)
            psi_e = (PE.U_expand() @ psi0.view(-1,1)).view(1,-1)
            psi_batch.append(psi_e)
            
        c1 = Circuit(self.nqubits)
        c1.BasicEntangleLayer(wires_lst, self.weight[0:18*self.nqubits],repeat=6)
        c1.YZYLayer(wires_lst, self.weight[18*self.nqubits:21*self.nqubits])
        
        
        for i in range(len(psi_batch)):
            psif = c1.U() @ psi_batch[i].view(-1,1)
            expec = psif.view(1,-1).conj() @ op.PauliZ(self.nqubits,0).U_expand() @ psif
            expec = expec.real
            if i == 0:
                rst = expec.view(1,-1)
            else:
                rst = torch.cat( (rst, expec.view(1,-1)), dim=0 )
        return rst
        
    
    
    def forward(self,input_lst_batch):
        '''
        1-采用可训练的TN_contract正向演化（10qubit时~16倍）
        2-让TN_contract支持batch操作，充分利用torch并行（~16倍）
        3-打包成.pyd文件，加速调用
        4-上GPU（长远规划）
        '''
        
        wires_lst = list( range(self.nqubits) )
        #创建线路
        c1 = Circuit(self.nqubits) 
        psi0 = c1.state_init().view(1,-1)
        
        MPS = StateVec2MPS(psi0,self.nqubits)
        MPSc = MPS2StateVec(MPS,return_sv=False)
        MPS_batch = torch.cat( tuple( [MPSc.unsqueeze(0)]*len(input_lst_batch) ),dim=0 )
        #encoding编码部分
        
        for i in range(len(input_lst_batch)):
            PE = PauliEncoding(self.nqubits, input_lst_batch[i], wires_lst)
            MPS_batch[i] = PE.TN_contract(MPS_batch[i])
        # s = MPS_batch.shape
        # print(MPS_batch.view(s[0],1,-1))
            
        c1 = Circuit(self.nqubits)
        c1.YZYLayer(wires_lst, self.weight[0:3*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[3*self.nqubits:6*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[6*self.nqubits:9*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[9*self.nqubits:12*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[12*self.nqubits:15*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[15*self.nqubits:18*self.nqubits])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, self.weight[18*self.nqubits:21*self.nqubits])
        
        # for i in range(len(MPS_batch)):
        #     MPSt = c1.TN_contract_evolution(MPS_batch[i])
        #     temp = torch.clone(MPSt)
        #     #print(MPSt.requires_grad)
        #     MPStz = op.PauliZ(self.nqubits,0).TN_contract(MPSt)
        #     expec = (temp.reshape(1,-1).conj() @ MPStz.reshape(-1,1) ).real
            
        #     if i == 0:
        #         rst = expec.view(1,-1)
        #     else:
        #         rst = torch.cat( (rst, expec.view(1,-1)), dim=0 )
        #     #print(rst.requires_grad)
        MPS_batch = c1.TN_contract_evolution(MPS_batch,batch_mod=True)
        MPS_batch0 = torch.clone(MPS_batch)
        MPS_batch = op.PauliZ(self.nqubits,0).TN_contract(MPS_batch,batch_mod=True)
        s = MPS_batch.shape
        MPS_batch = MPS_batch.reshape(s[0],-1,1)
        MPS_batch0 = MPS_batch0.reshape(s[0],1,-1)
        rst = (MPS_batch0 @ MPS_batch).real
        rst = rst.squeeze(-1)
        #rst = rst.squeeze(-1)
        return rst
        #======================================================================
        
        
        
        
        

class qnet(torch.jit.ScriptModule):
    
    def __init__(self,nqubits):
        super().__init__()
        
        self.nqubits = nqubits
        self.circuit = qcir(self.nqubits)
        self.FC1 = nn.Linear(len(self.circuit.M_lst),8)
        self.FC2 = nn.Linear(8,1)
        # self.FC3 = nn.Linear(8,1)
      
   
    
    def forward(self,x_batch):
        
        #输入数据的非线性预处理
        #pre_batch = torch.sqrt( 0.5*(1 + torch.sigmoid(x_batch)) )
        pre_batch = x_batch
        #print('pre_batch: ',pre_batch.requires_grad)
        cir_out = 6*self.circuit ( pre_batch )
        #print(cir_out)
        #print('cir_out: ',cir_out.requires_grad)
        return cir_out[:,0]
        # out = nn.functional.leaky_relu(self.FC1(cir_out))
        # out = torch.sigmoid(self.FC1(cir_out))
        # print('out: ',out.requires_grad)
        # out = nn.functional.leaky_relu(self.FC2(out))
        # out = nn.functional.leaky_relu(self.FC3(out))
        # return out





def foo(x1):
    y = 2*math.sin(2*x1+1.9)
    return y




if __name__ == "__main__":
    
    N = 10
    num_examples = 512
    num_inputs = 1
    num_outputs = 1
    
    features = torch.empty( num_examples,num_inputs )
    labels = torch.empty( num_examples,num_outputs )
    for i in range(num_examples):
        features[i] = torch.rand(num_inputs)*2*math.pi

    for i in range(num_examples):
        labels[i] = foo( features[i][0] ) + 1e-3*random.random()
    
    def data_iter(batch_size, features, labels):
        #输入batch_size，输入训练集地数据features+标签labels
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices) #把indices列表顺序随机打乱
        for i in range(0,num_examples,batch_size):
            #每次取batch_size个训练样本,j代表索引
            j = torch.LongTensor( indices[i:min(i+batch_size,num_examples)] ) 
            #print(features.index_select(0,j), labels.index_select(0,j))
            yield features.index_select(0,j), labels.index_select(0,j)
            #把张量沿着0维，只保留取出索引号对应的元素
    
#=============================================================================
    
    net1 = qnet(N)      #构建训练模型
    loss = nn.MSELoss() #平方损失函数
    
    print('start producing torchscript file')
    scripted_modeule = torch.jit.script(qnet(N))
    torch.jit.save(scripted_modeule, 'test_torchscript.pt')
    print('completed!')
    
    
    #定义优化器，也就是选择优化器，选择Adam梯度下降，还是随机梯度下降，或者其他什么
    #optimizer = optim.SGD(net1.parameters(), lr=0.001) #lr为学习率
    optimizer = optim.Adam(net1.parameters(), lr=0.3) #lr为学习率
    '''
    #注意，batch_size增大，学习率也要增大（一般增大相同的倍数）
    #注意，一般来说batch_size增大，最终输出的scaling也要增大
    '''
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100], gamma=0.1)
    
    num_epochs = 20;
    batch_size = 64;
    
    #记录loss随着epoch的变化，用于后续绘图
    epoch_lst = [i+1 for i in range(num_epochs)]
    loss_lst = []
    
    for epoch in range(1,num_epochs+1):
        t1 = time.time()
        T_forward = 0.0
        for x,y in data_iter(batch_size,features,labels):
            t8 = time.time()
            output = net1(x);
            t9 = time.time()
            T_forward = T_forward + t9 - t8
            #squeeze是为了把y维度从1x3变成3
            #print(output)
            l = loss(output.squeeze(),y.squeeze())
            
            # print(l)
            
            #梯度清0
            optimizer.zero_grad() 
            l.backward()
            #print('output.grad: ',output.requires_grad)
            '''
            parameters：希望实施梯度裁剪的可迭代网络参数
            max_norm：该组网络参数梯度的范数上限
            norm_type：范数类型(一般默认为L2 范数, 即范数类型=2) 
            torch.nn.utils.clipgrad_norm() 的使用应该在loss.backward() 之后，optimizer.step()之前.
            '''
            #nn.utils.clip_grad_norm_(net1.circuit.weight,max_norm=1,norm_type=2)
            #print('loss: ',l.item())
            print("weights_grad2:",net1.circuit.weight.grad,'  weight is leaf?:',net1.circuit.weight.is_leaf)
            # grad = net1.circuit.weight.grad
            # net1.circuit.weight.grad \
            #     = torch.where(torch.isnan(grad),torch.full_like(grad, 0),grad)
            optimizer.step()
            
        lr_scheduler.step() #进行学习率的更新
        loss_lst.append(l.item())
        t2 = time.time()
        print("epoch:%d, loss:%f" % (epoch,l.item()),\
              ';current lr:', optimizer.state_dict()["param_groups"][0]["lr"],\
                  '   耗时：',str(t2-t1)[:6],' forward耗时：',str(T_forward)[:6])
        
    
    
    print('开始绘图：')
    plt.cla()
    plt.subplot(121)
    xx = list(features[:num_examples,0])
    
    #yy = [float(each) for each in net1( features[:num_examples,:] ).squeeze() ]
    yy = []
    for i in range(num_examples):
        yy.append( float( net1(features[i:i+1,:]).squeeze() ) )
    #print(yy)
    xx = [float( xi ) for xi in xx]
    yy_t = [foo(xi) for xi in xx]
    plt.plot(xx,yy,'m^',linewidth=1, markersize=2)
    plt.plot(xx,yy_t,'g^',linewidth=1, markersize=0.5)
    
    plt.subplot(122)
    plt.plot(epoch_lst,loss_lst,'r^--',linewidth=1, markersize=1.5)
    plt.show()
    
    
    
input('test_TN_train.py END')

