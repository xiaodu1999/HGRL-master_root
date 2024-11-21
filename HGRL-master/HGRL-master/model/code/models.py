import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter
from functools import reduce
from utils import dense_tensor_to_sparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from printt import printt
from print import print
class HGRL(nn.Module):
    def __init__(self, nfeat_list, nhid, nclass, dropout,
                 type_attention=True, node_attention=True,
                 gamma=0.1, sigmoid=False, orphan=False, #True
                 write_emb=True
                 ):
        super(HGRL, self).__init__()
        self.sigmoid = sigmoid
        self.type_attention = type_attention
        self.node_attention = node_attention

        self.write_emb = write_emb
        if self.write_emb:
            self.emb = None
            self.emb2 = None

        self.nonlinear = F.relu_

        self.nclass = nclass
        self.ntype = len(nfeat_list)

        dim_1st = nhid
        dim_2nd = nclass
        if orphan:
            dim_2nd += self.ntype - 1
        
        self.gc2 = nn.ModuleList()
        if not self.node_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append( GraphConvolution(nfeat_list[t], dim_1st, bias=False) )
                self.bias1 = Parameter( torch.FloatTensor(dim_1st) )
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(nfeat_list, dim_1st, gamma=gamma)
        self.gc2.append( GraphConvolution(dim_1st, dim_2nd, bias=True) )

        if self.type_attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append( SelfAttention(dim_1st, t, 50) )
                self.at2.append( SelfAttention(dim_2nd, t, 50) )
           
        self.dropout = dropout
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(32 * (50 // 1), 64)  # Assuming output sequence length is the same
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)


    def forward(self, x_list, adj_list, adj_all = None):
        # print('x_list', x_list)#'list' object has no attribute 'shape'

        # print(' adj_list', adj_list)

        x0 = x_list
        x0 = [torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x for x in x_list]
        print('x0',len(x0))
        for i in range(self.ntype):
            if i!=1:     
                #x_i = torch.tensor(x0[i], dtype=torch.float32)
                x_i = x0[i]

                x_i = x_i.unsqueeze(dim=1) 

                # 执行卷积操作
                x_i = self.relu(self.conv1(x_i))  
                print('first conv, x_i shape:', x_i.shape)
                x_i = self.pool(x_i)
                # 进行第二次卷积
                x_i = self.relu(self.conv2(x_i))  
                print('second conv, x_i shape:', x_i.shape)
                x_i = self.pool(x_i)
                x_i=x_i.view(x_i.shape[0],-1)
                # 更新 x0 列表中的特征
                x0[i] = x_i
            else:
                
               # x_i =  torch.tensor(x0[i], dtype=torch.float32)
                x0[i] = x0[i]
                

        if not self.node_attention: #无注意力
            printt('无注意力')
            x1 = [None for _ in range(self.ntype)]
            # First Layer
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    idx = t2
                    print('x0[t2]', x0[t2].shape)
                    #
                   
                    #input()
                    x_t1.append( self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1 )
                if self.type_attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)
                    
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            #print('else')
            x1 = [None for _ in range(self.ntype)]
            print('x1',x1)#[None, None, None]
            x1_in = self.gc1(x0, adj_list)
            print('x1_in', len(x1_in))#3
            print('x1_in',x1_in[0][0].shape)#[30, 512]
            print('x1_in',x1_in[0][1].shape)#[30, 512]
            print('x1_in',x1_in[0][2].shape)#[30, 512])
            print('x1_in',x1_in[1][0].shape)#[2, 512])
            print('x1_in',x1_in[1][1].shape)#[2, 512])  
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.type_attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )##类型注意力
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                print('x_t1',x_t1.shape)#([30, 512]) [2, 512]) 12, 512])
                # input()

                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]        
        
        x2 = [None for _ in range(self.ntype)]
        # Second Layer
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                idx = 0

                #printt('self.gc2[idx](x1[t2], adj_list[t1][t2])',self.gc2[idx](x1[t2], adj_list[t1][t2]).shape)
                # #([30, 5])
                #([30, 5]) ([30, 5]) ([2, 5 ([2, 5 ([2, 5, [12, 5]) [12, 5]) [12, 5])
                #input()
                x_t1.append( self.gc2[idx](x1[t2], adj_list[t1][t2]) )
            if self.type_attention:
                x_t1, weights = self.at2[t1]( torch.stack(x_t1, dim=1) )
                print('x_t1',x_t1.shape)#[30, 5])  [2, 5]) ([12, 5])
                # input()
            else:
                x_t1 = reduce(torch.add, x_t1)

            x2[t1] = x_t1
            if self.write_emb and t1 == 0:
                self.emb2 = x2[t1]

            #printt('x_t1',x_t1.shape)
            # output layer
            if self.sigmoid:
                x2[t1] = torch.sigmoid(x_t1)
            else:
                x2[t1] = F.log_softmax(x_t1, dim=1)
                #printt('x2[t1]',x2[t1].shape)
        return x2

