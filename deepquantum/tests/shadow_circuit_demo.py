# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:02:05 2021

@author: shish
"""
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

import random
import time

from deepquantum import Circuit
import deepquantum as dp

class qcircuit2(nn.Module):
    
    def __init__(self, N, n_sqc, device):
        super().__init__()
        self.nqubits = N
        self.n_sqc = n_sqc
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(9*self.n_sqc), a=0.0, b=2*torch.pi) )
            
        
        self.device = device
        
        self.M = dp.multi_kron([dp.PauliX().matrix]*self.n_sqc) 
        pass
    
    def normlization(self,vector_batch):
        '''
        将输入batch的每一行都归一化
        '''
        #num_row = vector_batch.shape[0]
        #num_col = vector_batch.shape[1]
        #assert vector_batch.shape[1] == 784;
        # 对二维数组vector_batch按行归一化
        # p=2表示二范式，也就是模平方和开根号，dim=1表示按行归一化
        rst_vector_batch = nn.functional.normalize(vector_batch, p=2, dim=1)
        # assert rst_vector_batch.shape[0] == vector_batch.shape[0]
        # assert rst_vector_batch.shape[1] == vector_batch.shape[1]
        # rst_vector_batch = torch.empty(num_row, num_col)
        # for i in range(num_row):
        #     summ = 0.0
        #     for j in range(num_col):
        #         summ += torch.abs( vector_batch[i,j] )**2
        #     for j in range(num_col):
        #         rst_vector_batch[i,j] = ( 1.0/torch.sqrt(summ) ) * vector_batch[i,j]
        
        return ( rst_vector_batch + 0j ).to(self.device)
    
    def amplitude_encoding(self, norm_vector_batch):
        '''
        N表示circuit的qubit总数，norm_vector_batch这是已经归一化的输入矢量
        '''
        N = self.nqubits
        num_row = norm_vector_batch.shape[0]
        num_col = norm_vector_batch.shape[1]
        #assert norm_vector_batch.shape[1] == 784;
        #print(num_col.dtype)
        if num_col > 2**N:
            raise ValueError('amplitude_encoding error: vector dimension too large! qubits not enough!')
        
        zero_m = ( torch.zeros(num_row, 2**N-num_col) + 0j ).to(self.device)
        rst = torch.cat( (norm_vector_batch, zero_m), 1 )
        return rst
    
    def build_circuit(self):
        '''
        n_sqc表示shadow circuit的qubit数目，必须满足n_sqc < N
        n_sqc一般不超过4
        '''
        n_sqc = self.n_sqc
        
        cir = Circuit(n_sqc)
        
        #wires = list( range(start_qbit, start_qbit+n_sqc, 1) )
        wires = list( range(n_sqc) )
        
        repeat = 2
        cir.BasicEntangleLayer(wires, self.weight[0:3*n_sqc*repeat], repeat=repeat)
        
        cir.YZYLayer(wires, self.weight[3*n_sqc*repeat:3*n_sqc*(repeat+1)])
        
        U = cir.U()
        
        return U.to(self.device)
    
    def cal_o_lst(self,state):
        #rho =  state.view(-1,1) @ dp.dag(state.view(-1,1))
        o_lst = torch.zeros(self.nqubits - self.n_sqc + 1)
        U = self.build_circuit()
        temp = dp.dag(U) @ self.M @ U
        for start_qbit in range(0,self.nqubits - self.n_sqc + 1):
            
            temp_lst = [torch.eye(2).to(self.device)+0j]*start_qbit \
                + [temp] \
                + [torch.eye(2).to(self.device)+0j]*( self.nqubits - self.n_sqc - start_qbit )
            
            #o = torch.trace( rho @ dp.multi_kron(temp_lst) ).real
            o = ( dp.dag(state.view(-1,1)) @ dp.multi_kron(temp_lst) @ state.view(-1,1)).real
            o_lst[start_qbit] = o
        return o_lst
        
    
    def forward(self,vector_batch):
        #batch_size = vector_batch.shape[0]
        state_batch = self.amplitude_encoding( self.normlization(vector_batch) ) 
        #o_batch = torch.empty( batch_size, self.nqubits - self.n_sqc + 1 )
        #assert len(state_batch) == batch_size
        #x_lst = []
        for idx,state in enumerate( state_batch ):
            # x = self.cal_o_lst(state)
            # x_lst.append(x)
            if idx == 0:
                o_batch = self.cal_o_lst(state).view(1,-1)
            else:
                o_batch = torch.cat( ( o_batch, self.cal_o_lst(state).view(1,-1) ) )
            #o_batch[idx] = self.cal_o_lst(state)
        
        return o_batch
        #return torch.stack(x_lst)
        
class qcircuit(torch.jit.ScriptModule):
    
    def __init__(self, N, n_sqc, device):
        super().__init__()
        self.nqubits = N
        self.n_sqc = n_sqc
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(15*self.n_sqc), a=0.0, b=2*torch.pi) )
            
        self.device = device
        self.M = dp.multi_kron([dp.PauliX().matrix]*self.n_sqc)
        pass
    
    def normlization(self,vector_batch):
        '''
        将输入batch的每一行都归一化
        '''  
        rst_vector_batch = nn.functional.normalize(vector_batch, p=2, dim=1)
        return ( rst_vector_batch + 0j ).to(self.device)
    
    def amplitude_encoding(self, norm_vector_batch):
        '''
        N表示circuit的qubit总数，norm_vector_batch这是已经归一化的输入矢量
        '''
        N = self.nqubits
        num_row = norm_vector_batch.shape[0]
        num_col = norm_vector_batch.shape[1]
        if num_col > 2**N:
            raise ValueError('amplitude_encoding error: vector dimension too large! qubits not enough!')
        
        zero_m = ( torch.zeros(num_row, 2**N-num_col) + 0j ).to(self.device)
        rst = torch.cat( (norm_vector_batch, zero_m), 1 )
        return rst
    
    def build_circuit(self):
        '''
        n_sqc表示shadow circuit的qubit数目，必须满足n_sqc < N
        n_sqc一般不超过4
        '''
        n_sqc = self.n_sqc
        
        cir = Circuit(n_sqc)
        wires = list( range(n_sqc) )
        repeat = 4
        cir.BasicEntangleLayer(wires, self.weight[0:3*n_sqc*repeat], repeat=repeat)
        
        cir.YZYLayer(wires, self.weight[3*n_sqc*repeat:3*n_sqc*(repeat+1)])
        
        U = cir.U()
        
        return U.to(self.device)
    
    def cal_o_lst(self,state_batch):
        batchsize = state_batch.shape[0]
        Hdim = state_batch.shape[1]
        state_batch = state_batch.view(batchsize,1,Hdim)
        state_batch_l = state_batch.conj()
        state_batch_r = state_batch.permute(0,2,1)
        #print(state_batch_l.shape,state_batch_r.shape)
        #U_lst = torch.zeros(self.nqubits - self.n_sqc + 1,1,2**self.nqubits,2**self.nqubits) + 0j   
        U = self.build_circuit()
        temp = dp.dag(U) @ self.M @ U
        shadows = self.nqubits - self.n_sqc + 1
        Hdim = 2**self.nqubits
        for start_qbit in range(0,shadows):
            temp_lst = [torch.eye(2**start_qbit).to(self.device)+0j] \
                     + [temp] \
                     + [torch.eye(2**( self.nqubits - self.n_sqc - start_qbit )).to(self.device)+0j]
            cur_M = dp.multi_kron(temp_lst).view(1,Hdim,Hdim)
            if start_qbit == 0:
                U_lst = cur_M
            else:
                U_lst = torch.cat( (U_lst,cur_M) , dim=0)
            #print(U_lst.shape)
        rst = state_batch_l @ U_lst.view(shadows,1,Hdim,Hdim) @ state_batch_r
        rst = rst.real
        rst = rst.squeeze(dim=-1).squeeze(dim=-1)
        
        # if len(rst.shape) == 1:
        #     rst = rst.unsqueeze(1)
        
        return rst.permute(1,0)
        
    
    def forward(self,vector_batch):
        
        state_batch = self.amplitude_encoding( self.normlization(vector_batch) ) 

        return self.cal_o_lst( state_batch )
        
                
        
        
#============================================================


class qnet(torch.jit.ScriptModule):
    
    def __init__(self,nqubits,n_sqc,device):
        super().__init__()
        
        self.nqubits = nqubits
        self.n_sqc = n_sqc
        self.circuit = qcircuit( self.nqubits, self.n_sqc , device )
        #self.circuit = qcircuit2( self.nqubits, self.n_sqc , device )
        
        num_qout = self.circuit.nqubits - self.circuit.n_sqc + 1
        self.FC1 = nn.Linear(num_qout,10)
        self.device = device
      
   
    
    def forward(self,x_batch):
        
        #输入数据的非线性预处理
        #pre_batch = torch.sqrt( 0.5*(1 + torch.sigmoid(x_batch)) )
        #pre_batch = torch.arctan( x_batch )
        #pre_batch = torch.arcsin( x_batch )
        #pre_batch = torch.arcsin( x_batch*x_batch )
        pre_batch = x_batch
        
        cir_out = self.circuit ( pre_batch )
        
        #return torch.sigmoid( cir_out[:,0] )
        #out = nn.functional.softmax(self.FC1(cir_out),dim=1) #dim=1同一行的元素做softmax
        out = self.FC1(cir_out)  #无需在此处做softmax，交叉熵loss计算中会自动帮你计算
        #print(out)
        # out = nn.functional.leaky_relu(self.FC2(out))
        # out = nn.functional.leaky_relu(self.FC3(out))
        #mmax,idx = torch.max( out, dim=1 )
        #idx = idx.view(-1,1) + torch.tensor(0.0)
        
        return out




if __name__ == "__main__":
    
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #40.7%，30epoch，512example，32batch，36pqc参数，lr=0.2
    #45.1%，100epoch，512example，32batch，36pqc参数，lr=0.2
    #36.8%:29.2%
    num_examples = 64
    num_inputs = 784
    num_outputs = 10
    
    num_epochs = 5;
    batch_size = 16;
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307),(0.3081))])
    
    
    
    
    
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                               train=True, 
                                               transform=transforms.ToTensor(),  
                                               download=True)
    n_train = num_examples
    train_dataset.data = train_dataset.data[:n_train]
    train_dataset.targets = train_dataset.targets[:n_train]
    
    
    
    
    
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=False, 
                                              transform=transforms.ToTensor())
    n_test = 200
    n_bias = 1000
    test_dataset.data = test_dataset.data[n_bias:n_bias+n_test]
    test_dataset.targets = test_dataset.targets[n_bias:n_bias+n_test]
    
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
     
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=1, 
                                              shuffle=False)
   
 
    device = 'cpu'
    net1 = qnet(10,4,device)      #构建训练模型
    
    net1 = net1.to(device)
    #loss = nn.MSELoss() #平方损失函数
    loss = nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer = optim.Adam(net1.parameters(), lr=0.2) #lr为学习率   
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10,15], gamma=0.8)
    

    #记录loss随着epoch的变化，用于后续绘图
    epoch_lst = [i+1 for i in range(num_epochs)]
    loss_lst = []
    
    print('start')
    for epoch in range(1,num_epochs+1):
        t1 = time.time()
        #for x,y in data_iter(batch_size,features,labels):
        for i,data in enumerate(train_loader):
            x,y = data
            #print(x.shape)
            #print('y :',y)
            #x = x.to(device);y = y.to(device);
            x = x.squeeze().view(-1,784)
            
            
            x.requires_grad_(True)
            output = net1(x);
            #print('yp:',output)
            #squeeze是为了把y维度从1x3变成3
            
            # 参考：
            # Pytorch中的CrossEntropyLoss()函数案例解读和结合one-hot编码计算Loss
            print(output.shape,'  ',y.shape)
            l = loss(output, y);
            optimizer.zero_grad()
            l.requires_grad_(True)
            l.backward()
            #print("weights_grad2:",net1.circuit.weight.grad,'  weight is leaf?:',net1.circuit.weight.is_leaf)
            optimizer.step()
            
        #lr_scheduler.step()
        t2 = time.time()
        loss_lst.append(l.item())
        
        print("epoch:%d, loss:%f" % (epoch,l.item()),\
              ';current lr:', optimizer.state_dict()["param_groups"][0]["lr"],'time:',t2-t1)
    #torch.save(net1,'shadow_circuit_module.pth')
#================================================================================
    score = 0
    for i,data in enumerate(test_loader):
        #print(i)
        x,y = data 
        #x = x.to(device); y = y.to(device)
        x = x.squeeze().view(-1,784)
        x.requires_grad_(True)
        output = net1(x);
        out = nn.functional.softmax(output, dim=1)
        mmax,idx = torch.max( out, dim=1 )
        #print('prediciton: ',int(idx),'   label: ',int(y))
        if int(idx) == int(y):
            score = score + 1
    print('测试集准确率： ',100.0*score/n_test,'%')
    
    
    input('')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    