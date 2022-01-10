"""
    更新门后版本
"""

#导入库文件
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from door_lxy import Circuit
# from calculate import dag, measure, IsUnitary
from deepquantum import qgate, qmath
from deepquantum.gates.qmath import dag
from deepquantum.gates.qcircuit import Circuit
from deepquantum.nn.modules.quanv import QuanConv2D
from deepquantum.embeddings.qembedding import PauliEncoding
from deepquantum.gates.qtensornetwork import MatrixProductState
import deepquantum.gates.qoperator as op
import time

BATCH_SIZE = 128
EPOCHS = 1     # Number of optimization epochs
n_train = 512    # Size of the train dataset
n_test = 128   # Size of the test dataset

SAVE_PATH = "./"  # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
seed = 42
np.random.seed(seed)        # Seed for NumPy random number generator
torch.manual_seed(seed)     # Seed for TensorFlow random number generator


DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root="./data",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

train_dataset.data = train_dataset.data[:n_train]
train_dataset.targets = train_dataset.targets[:n_train]

test_dataset = datasets.MNIST(root="./data",
                              train=False,
                              transform=transforms.ToTensor())

test_dataset.data = test_dataset.data[:n_test]
test_dataset.targets = test_dataset.targets[:n_test]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nqubits = 4
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(15*self.nqubits), a=0.0, b=2*torch.pi) )
            
        self.quan1 = QuanConv2D(n_qubits=4, stride=2, kernel_size=2)
        
        self.BATCH_SIZE = 128
        self.fc1 = nn.Linear(14*14*4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.squeeze(1)
        s = x.shape
        batch_size = s[0]
        assert batch_size == self.BATCH_SIZE
        row = s[1]
        col = s[2]
        L = int( batch_size*row*col*0.25 )
        #================================================================================
        c1 = Circuit(self.nqubits) 
        psi0 = c1.state_init().view(1,-1)
        MPSc = MatrixProductState(psi0,self.nqubits)
        MPS_batch = torch.cat( tuple( [MPSc.unsqueeze(0)]*L ),dim=0 )
        t = time.time()
        for id1 in range(batch_size):
            for i in range(0,row,2):
                for j in range(0,col,2):
                    input_lst = [x[id1][i,j],x[id1][i,j+1],x[id1][i+1,j],x[id1][i+1,j+1]]
                    PE = PauliEncoding( self.nqubits, input_lst, list(range(self.nqubits)) )
                    MPS_batch[i] = PE.TN_contract(MPS_batch[i])
        print('三重循环耗时: ',time.time() - t)
        #=================================================================================
        
        wires_lst = list(range(self.nqubits))
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
        
        MPS_batch = c1.TN_contract_evolution(MPS_batch,batch_mod=True)
        s = MPS_batch.shape
        MPS_batch1 = torch.clone(MPS_batch)
        MPS_batch1 = MPS_batch1.reshape(s[0],1,-1)
        
        for i in range(self.nqubits):
            MPS_batch0 = torch.clone(MPS_batch)
            #print(i,'  ',id(MPS_batch0))
            MPS_batch0 = op.PauliZ(self.nqubits,i).TN_contract(MPS_batch0, batch_mod=True)
            MPS_batch0 = MPS_batch0.reshape(s[0],-1,1)
            rst = (MPS_batch1 @ MPS_batch0).real
            rst = rst.squeeze(-1)
            #print('rst.shape:' ,rst.shape)

            if i == 0:
                rstf = rst
            else:
                rstf = torch.cat( (rstf,rst),dim=1 )
        
        rstf = rstf.view(batch_size, -1)
        #print('rstf.shape: ',rstf.shape)
        r1 = F.relu(self.fc1(rstf))
        #print('r1.shape: ',r1.shape)
        r2 = self.fc2(r1)
        #print('r2.shape: ',r2.shape)
        return r2
        
                    
        #x = self.quan1(x)
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        #return x
        return r2


print('==========================start training=============================')
model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

loss_list = []

model.train().to(DEVICE)
for epoch in range(EPOCHS):
    total_loss = []
    t1 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        print('data.shape: ',data.shape)
        print('target.shape: ',target.shape)
        target = target.to(DEVICE)
        
        data = data.to(torch.float32).to(DEVICE)

        # Forward pass
        output = model(data).to(DEVICE)

        # Calculating loss
        loss = loss_func(output, target).to(DEVICE)
        
        #梯度清零
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()

        # Optimize the weights
        optimizer.step()

        total_loss.append(loss.item())
    t2 = time.time()
    loss_list.append(sum(total_loss) / len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / EPOCHS, loss_list[-1]))
    print('耗时: ',t2-t1)

model.eval()
with torch.no_grad():
    correct = 0
    total_loss = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(torch.float32).to(DEVICE)
        target = target.to(DEVICE)
        output = model(data).to(DEVICE)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_func(output, target)
        total_loss.append(loss.item())
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100 / BATCH_SIZE)
        )
input('END')