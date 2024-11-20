# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import random
import torch
import numpy
from print import print
from printt import printt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TripletLoss_ts(torch.nn.modules.loss._Loss):
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss_ts, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        print('batch, train', batch.shape, train.shape)
        #batch [3, 3, 6, 30]),batch,dimension, number of subseries, length of subseries
        #train ([5, 3, 6, 30]),batch,dimension, number of subseries, length of subseries

        B, C, num_subseries, subseries_length = batch.shape
        batch_size = batch.size(0)
        train_size = train.size(0)
       
       # Randomly choose m dimensions from C dimensions
        max_value = C // 3  # 计算最大值
        max_value = max(1, max_value)  # 在 1 和 C//3 之间选择随机整数
        selected_dims = random.sample(range(C), max_value)  # Randomly select m dimensions
        print('selected_dims',selected_dims)
    #==== 锚点和正样本
        anchor_samples = []  # 初始化锚点样本 
        pos_samples = []  # 初始化正样本 

        # 随机选择锚点
        for anchor_dim in selected_dims:
            # 随机选择锚点的起始索引
            anchor_index = random.randint(0, num_subseries - 1 - num_subseries // 3)
            print(f'Anchor dimension: {anchor_dim}, anchor_index: {anchor_index}')
            
            # 确定最大子序列数量
            max_num_subseries = min(num_subseries // 3, num_subseries - anchor_index)
            
            # 定义锚点 T_ref
            anchor = batch[:, anchor_dim, anchor_index:anchor_index + max_num_subseries, :]
            anchor_samples.append(anchor.unsqueeze(1))  # 增加一个维度以便后续拼接
            print('Anchor shape:', anchor.shape)
        
            # 随机选择偏移量 k，确保正样本的起始位置不等于锚点的起始位置
            k = random.randint(1, max_num_subseries - 1)
            
            # 正样本选择逻辑
            if anchor_index + k + max_num_subseries <= num_subseries:  # 确保不越界
                pos_sample = batch[:, anchor_dim, anchor_index + k:anchor_index + k + max_num_subseries, :]
                pos_samples.append(pos_sample.unsqueeze(1))
                print('Positive sample within bounds:', pos_sample.shape)
            else:
                print('else')
                # 如果超出边界，则填充为和锚点子序列数量一样
                fill_length = anchor_index + k + max_num_subseries - num_subseries  # 计算需要填充的长度
                # 创建零填充张量
                pad = torch.zeros(B, fill_length, subseries_length)  # Note: removed C dimension in padding
                # 从末尾开始选择，确保正样本维度一致
                pos_sample = batch[:, anchor_dim, -(num_subseries - (anchor_index + k)):, :]
                
                pad = pad.to(device)
                pos_sample = pos_sample.to(device)
                pos_sample = torch.cat((pad, pos_sample), dim=1)  # 沿第 1 维（子序列维度）拼接
                pos_samples.append(pos_sample.unsqueeze(1))
                print('Positive sample with padding:', pos_sample.shape)

        # 堆叠锚点样本和正样本
        anchor_tensor = torch.cat(anchor_samples, dim=1)  # shape: (B, m, max_num_subseries, subseries_length)
        pos_samples_tensor = torch.cat(pos_samples, dim=1) if pos_samples else None  # shape: (B, m, max_num_subseries, subseries_length)

        # 打印形状以检查
        print("Shape of anchor_tensor:", anchor_tensor.shape)
        print("Shape of pos_samples_tensor:", pos_samples_tensor.shape)

#=====
        
# # Select negative samples
#         neg_samples = []
#         # Option 1: Randomly select tokens from the other dimensions in the current sample
#         for dim in range(C):
#             if dim not in selected_dims:
#                 neg_start = random.randint(0, subseries_length - num_subseries)
#                 neg_sample = batch[:, dim, neg_start:neg_start + num_subseries]  # Use slicing for the entire batch
#                 neg_samples.append(neg_sample)
        
#         # Option 2: Randomly select the entire time series from m unselected dimensions
#         unselected_dims = [d for d in range(C) if d not in selected_dims]
#         for dim in unselected_dims:
#             neg_samples.append(batch[:, dim])  # Append the whole time series from unselected dimensions
        
#         # Convert negative samples to tensor
#         neg_samples_tensor = torch.stack(neg_samples) if neg_samples else None  # Handle empty case

#====   锚点和正样本的表示学习
        print("anchor_tensor dtype:", anchor_tensor.dtype)
        anchor_tensor = anchor_tensor.float() 
        representation = encoder(anchor_tensor)  # Anchors representations
        print('representation', representation.shape)#representation torch.Size([3, 10, 20])
        

        pos_samples_tensor=pos_samples_tensor.float()
        positive_representation = encoder(pos_samples_tensor)  # Positive samples representations
        print('positive_representation',positive_representation.shape)# positive_representation torch.Size([3, 10, 20])
        #input()

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations

        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()
#====
     # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples,batch_size)
        )
        samples = torch.LongTensor(samples)

#=============

#====
        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # # Negative loss: -logsigmoid of minus the dot product between
            # # anchor and negative representations
            # negative_representation = encoder(
            #     torch.cat([train[samples[i, j]: samples[i, j] + 1][
            #         :, :,
            #         beginning_samples_neg[i, j]:
            #         beginning_samples_neg[i, j] + length_pos_neg
            #     ] for j in range(batch_size)])
            # )
#====       
            
            neg_samples = []  # 初始化样本 
            for j in range(batch_size):
                # 获取当前样本的起始索引
                sample_start = samples[i, j]
                # 初始化负样本列表
                selected_dims_neg = random.sample(range(C), max_value)  # Randomly select m dimensions
                
                # 随机选择
                for neg_dim in selected_dims_neg:
                    # 随机选择的起始索引
                    neg_index = random.randint(0, num_subseries - 1 - num_subseries // 3)
                    print(f'Anchor dimension: {neg_dim}, neg_index: {neg_index}')
                    
                    # 确定最大子序列数量
                    max_num_subseries = min(num_subseries // 3, num_subseries - neg_index)
                    print('max_num_subseries', max_num_subseries)
                    
                    # 从训练数据中提取该负样本片段
                    neg = train[sample_start:sample_start + 1, neg_dim, neg_index:neg_index + max_num_subseries, :]
                    neg_samples.append(neg.unsqueeze(1))  # 增加一个维度以便后续拼接
                    print('negshape:', neg.shape)

            # 在外层循环结束后，拼接所有负样本
            neg_tensor = torch.cat(neg_samples, dim=1)
            print('neg_tensor1', neg_tensor.shape)

            # 批次拼接成一个张量
            neg_tensor_batch = neg_tensor.view(batch_size, -1, neg_tensor.shape[2], neg_tensor.shape[3])  # 修改为合适的形状
            print('neg_tensor_batch', neg_tensor_batch.shape)

            # 生成的张量形状为 (batch_size, C, length_pos_neg)
            negative_representation = encoder(neg_tensor_batch)
            print('negative_representation', negative_representation)  # negative_representation 应该是形状为 [batch_size, 200]
            #input()

    #=====
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()
        print('representation', representation.shape)
        #representation torch.Size([4, 200])
        # representation torch.Size([2, 200])

        print('loss done',loss)
        #input()
        print('representation', representation)
        #input()
        return loss,representation
