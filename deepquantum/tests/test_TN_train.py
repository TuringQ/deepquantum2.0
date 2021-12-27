# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:24:22 2021

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
NOTE:
参考：
https://pytorch.org/docs/master/generated/torch.linalg.svd.html?highlight=svd#torch.linalg.svd
如果forward过程，涉及一个矩阵A通过SVD分解成U,S,V，然后U,V用作下阶段的计算。
这时梯度的反向传播要求，矩阵A不能有重复的奇异值（数值上，两个奇异值的差不能很小），不能有0奇异值。

因为从特征值的角度考虑（我们假设A是方阵），重复的特征值，意味着某两个特征矢量
可以是两个特征基矢的任意线性组合，0特征值，意味着任意矢量都是特征矢量，这都会
导致A分解的U和V不唯一，相当于一个一对多的映射，无法求梯度。

解决方案：
剔除梯度中为nan的成分，把所有nan替换成0，即如果某一轮某一个输入求梯度，
发现对某个参数梯度为nan，那就不更新该参数。

缺点是降低了训练效率。LOSS很难收敛。


l.backward()
grad = net1.circuit.weight.grad
net1.circuit.weight.grad \
    = torch.where(torch.isnan(grad),torch.full_like(grad, 0),grad)
optimizer.step()
'''

'''
NOTE:
1)batch为1就可以收敛，大于1就不能收敛，why？
2)尽量避免使用list列表操作，line 159~line 163的列表操作会导致无法求梯度，使得grad=None
3)自定义的backward必需带有grad_output参数，这表示求导链式法则中，loss相对于该forward输出的导数
backward的返回值必须乘上grad_output，因为backward主体计算的是forward输出对于输入的导数，根据链
式法则，乘上grad_output之后，才表示loss相对输入的导数。
'''
# class HybridFunction(torch.autograd.Function):
#     '''
#     参考：
#     https://blog.csdn.net/winycg/article/details/104410525
#     https://zhuanlan.zhihu.com/p/27783097
#     https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html
#     '''
#     @staticmethod
#     def forward(ctx, params_lst, N,input_state, M):
#         c1 = Circuit(N)
#         wires_lst = list(range(N))
#         for i in range(N):
#             c1.rx(params_lst[i], i)
#         c1.YZYLayer(wires_lst, params_lst[1*N:4*N])
#         c1.ring_of_cnot(wires_lst)
#         c1.YZYLayer(wires_lst, params_lst[4*N:7*N])
        
#         ctx.circuit = c1
#         ctx.save_for_backward(input_state, M)
#         result = c1.cir_expectation(input_state, M)
#         return result
    
#     @staticmethod
#     def backward(ctx,grad_output):
#         input_state, M = ctx.saved_tensors
#         ps = parameter_shift(ctx.circuit, input_state, M)
#         #save_dict = ps.save_mid_MPS(op.PauliZ(ctx.circuit.nqubits,0))
#         grad = ps.cal_params_grad()
#         #grad = ctx.ps.cal_params_grad2(ctx.save_dict)
#         # print('old: ',grad1)
#         # print('new: ',grad)
#         return grad * grad_output,None,None,None


class HybridFunction(torch.autograd.Function):
    '''
    参考：
    https://blog.csdn.net/winycg/article/details/104410525
    https://zhuanlan.zhihu.com/p/27783097
    https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html
    '''
    @staticmethod
    def forward(ctx, params_lst, N,input_state, M):
        c1 = Circuit(N)
        wires_lst = list(range(N))
        for i in range(N):
            c1.rx(params_lst[i], i)
        c1.YZYLayer(wires_lst, params_lst[1*N:4*N])
        c1.ring_of_cnot(wires_lst)
        c1.YZYLayer(wires_lst, params_lst[4*N:7*N])
        c1.cphase(params_lst[7*N], [1,0])
        # c1.ring_of_cnot(wires_lst)
        # c1.YZYLayer(wires_lst, params_lst[6*N:9*N])
        # c1.ring_of_cnot(wires_lst)
        # c1.YZYLayer(wires_lst, params_lst[9*N:12*N])
        #print('params_lst: ',params_lst.requires_grad)
        #ctx.circuit = c1
        #ctx.save_for_backward(input_state, M)
        
        ps = parameter_shift(c1, input_state, M)
        save_dict,sv_f = ps.save_mid_MPS(op.PauliZ(c1.nqubits,0))
        ctx.save_dict = save_dict
        ctx.ps = ps
        result = sv_f.view(1,-1).conj() @ M @ sv_f.view(-1,1)
        result = result.real
        #result = c1.cir_expectation(input_state, M)
        return result
    
    @staticmethod
    def backward(ctx,grad_output):
        grad = ctx.ps.cal_params_grad2(ctx.save_dict)
        # print('old: ',grad1)
        # print('new: ',grad)
        return grad * grad_output,None,None,None



#==============================================================================
class qcir(torch.jit.ScriptModule):
    def __init__(self,nqubits):
        super().__init__()
        #属性：量子线路qubit数目，随机初始化的线路参数，测量力学量列表
        self.nqubits = nqubits
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(7*self.nqubits+1), a=0.0, b=2*torch.pi) )
        
        self.M_lst = self.Zmeasure()

        
    def Zmeasure(self):
        #生成测量力学量的列表
        M_lst = []
        for i in range(self.nqubits):
            Mi = op.PauliZ(self.nqubits,i).U_expand()
            M_lst.append( Mi )
        
        return M_lst
            
    
    def forward(self,input_lst_batch):
        '''
        构建变分量子线路的演化矩阵
        '''
        
        wires_lst = [i for i in range(self.nqubits)]
        #创建线路
        c1 = Circuit(self.nqubits) 
        psi0 = c1.state_init().view(1,-1)
        #psi_err = nn.functional.normalize( torch.rand(1,2**self.nqubits)+torch.rand(1,2**self.nqubits)*1j,p=2,dim=1 )
        #psi0 = psi0 + 0.01*psi_err 
        #print(psi0)
        MPS = StateVec2MPS(psi0,self.nqubits)
        #encoding编码部分
        MPS_batch = []
        for i in range(len(input_lst_batch)):
            PE = PauliEncoding(self.nqubits, input_lst_batch[i], wires_lst)
            #print('PE: ',id(PE))
            MPS_f = PE.TN_operation(copy.deepcopy(MPS))
            
            '''
            子诶调用父类的method：TN_operation返回值赋值后MPS_f地址一直一样
            导致直接append(MPS_f)会导致MPS_batch内部所有元素都链接到一个地址上
            从而全部被覆盖，所以必须用copy.deepcopy()
            '''
            #print("MPS_f",id(MPS_f))
            # print(MPS_f)
            MPS_batch.append(MPS_f)
            #MPS_batch.append( MPS_f )
            #MPS_batch[i] = copy.copy(MPS_f)
            #print("MPS_batch",id(MPS_batch[i]))
            #print(MPS_batch[i])
        # print(0,MPS_batch[0])
        # print(1,MPS_batch[1])
        #print('================================================================')
        # ======================================================================
        for i in range(len(MPS_batch)):
            #MPS_batch[i] = c1.TN_evolution(MPS_batch[i])
            psi_i = MPS2StateVec(MPS_batch[i])
            
            # rst_lst = []
            # for Mi in self.M_lst:
            #     rst_lst.append(\
            #             HybridFunction.apply(self.weight, self.nqubits, psi_i, Mi ) )
            # r = torch.tensor(rst_lst,requires_grad=True)
            
            # for j, Mi in enumerate(self.M_lst):
            #     expec = HybridFunction.apply(self.weight, self.nqubits, psi_i, Mi ).view(1,1)
            #     if j == 0:
            #         r = expec
            #     else:
            #         r = torch.cat((r,expec),dim=1)
            
            
            r = HybridFunction.apply(self.weight, self.nqubits, psi_i, self.M_lst[0] )
            
            if i == 0:
                rst = r.view(1,-1)
            else:
                rst = torch.cat( (rst, r.view(1,-1)), dim=0 )
        #print(rst)
        # print('rst: ',rst.requires_grad)
        # print('self.weight grad: ',self.weight.requires_grad)
        return rst
        #======================================================================
        
        #variation变分部分
        # c1.YZYLayer(wires_lst, self.weight[0*self.nqubits:3*self.nqubits])
        # c1.ring_of_cnot(wires_lst)
        # c1.YZYLayer(wires_lst, self.weight[3*self.nqubits:6*self.nqubits])
        # c1.ring_of_cnot(wires_lst)
        # c1.YZYLayer(wires_lst, self.weight[6*self.nqubits:9*self.nqubits])
        # c1.ring_of_cnot(wires_lst)
        # c1.YZYLayer( wires_lst, self.weight[9*self.nqubits:12*self.nqubits] ) 
        
        # psif_batch = []
        # for i in range(len(MPS_batch)):
        #     MPS_batch[i] = c1.TN_evolution(MPS_batch[i])
        #     if i == 0:
        #         psif_batch = MPS2StateVec(MPS_batch[i]).view(1,-1)
        #     else:
        #         psif_batch = torch.cat((psif_batch,MPS2StateVec(MPS_batch[i]).view(1,-1)),dim=0)
        # #print(psif_batch.shape)
        # return psif_batch
    
    
    # def forward(self,input_lst_batch):
    #     #计算编码后的态和变分线路的演化矩阵
    #     psif = self.build_circuit(input_lst_batch)
        
        
    #     #模拟测量得到各个测量力学量的期望值
    #     measure_rst = []
    #     for Mi in self.M_lst: 
    #         measure_rst.append((psif.conj().unsqueeze(-2) @ Mi @ psif.unsqueeze(-1)).squeeze(dim=2).real)
        
    #     #以4个qubit的线路为例，把3个[batch,1]的矩阵拼接为[batch,4]
    #     rst = torch.cat( tuple(measure_rst),dim=1 ) 
        
    #     #把值域做拉伸，从[-1,1]变为[-4,4]
    #     rst = ( rst + 0 ) * 4
    #     return rst
        
        
        
        
        

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
        cir_out = 4*self.circuit ( pre_batch )
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
    
    N = 3
    num_examples = 256
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
    
    # print('start producing torchscript file')
    # scripted_modeule = torch.jit.script(qnet(4))
    # torch.jit.save(scripted_modeule, 'test_torchscript.pt')
    # print('completed!')
    
    
    #定义优化器，也就是选择优化器，选择Adam梯度下降，还是随机梯度下降，或者其他什么
    #optimizer = optim.SGD(net1.parameters(), lr=0.001) #lr为学习率
    optimizer = optim.Adam(net1.parameters(), lr=0.01) #lr为学习率
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100], gamma=0.1)
    
    num_epochs = 15;
    batch_size = 2;
    
    #记录loss随着epoch的变化，用于后续绘图
    epoch_lst = [i+1 for i in range(num_epochs)]
    loss_lst = []
    
    for epoch in range(1,num_epochs+1):
        t1 = time.time()
        for x,y in data_iter(batch_size,features,labels):
            
            output = net1(x);
            #squeeze是为了把y维度从1x3变成3
            
            l = loss(output.squeeze(),y.squeeze())
            
            # print(l)
            # print(l.requires_grad)
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
            #print("weights_grad2:",net1.circuit.weight.grad,'  weight is leaf?:',net1.circuit.weight.is_leaf)
            # grad = net1.circuit.weight.grad
            # net1.circuit.weight.grad \
            #     = torch.where(torch.isnan(grad),torch.full_like(grad, 0),grad)
            optimizer.step()
            
        lr_scheduler.step() #进行学习率的更新
        loss_lst.append(l.item())
        t2 = time.time()
        print("epoch:%d, loss:%f" % (epoch,l.item()),\
              ';current lr:', optimizer.state_dict()["param_groups"][0]["lr"],\
                  '   耗时：',t2-t1)
        
    
    
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

