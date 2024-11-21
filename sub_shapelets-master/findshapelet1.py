from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from print import print
from printt import printt
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from matplotlib.colors import ListedColormap
from matplotlib import cm

#区分维度
class MultiHeadSelfAttention2(nn.Module):
    def __init__(self, d_model, num_heads,num_channels):
        super(MultiHeadSelfAttention2, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # 线性层，假设 d_k 是每个头的维度
        self.qkv = nn.Linear(num_channels, num_channels * 3)

        # 输出层
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):  # x 的形状为 [3, 3, 19, 10]
        batch_size, num_dims, num_subsequences, seq_length = x.shape
        
        # 将 x 变形为 [3, 19, 10, 3] 以便后续计算
        x_reshaped = x.permute(0, 2, 3, 1)  # 形状变为 [3, 19, 10, 3]

        # 将 x_dim 转换为 q, k, v
        qkv = self.qkv(x_reshaped)  # 输出形状为 [3, 19, 10, 3 * d_k]
        q, k, v = qkv.chunk(3, dim=-1)  # 每个的形状为 [3, 19, 10, d_k]

        # 计算自注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力权重，形状为 [3, 19, 10, 10]
        attn_weights = attn_weights / (seq_length ** 0.5)  # 缩放
        attn_weights = F.softmax(attn_weights, dim=-1)  # softmax

        # 应用注意力权重到 v
        output = torch.matmul(attn_weights, v)  # 形状为 [3, 19, 10, d_k]

        # 将输出变形回原始形状 [3, 3, 19, d_k]
        output = output.permute(0, 3, 1, 2)  # 形状变为 [3, 3, 19, d_k]

        # 最后通过输出层
        output = self.fc_out(output)

        return output  # 返回处理后的输出



#区分维度循环
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,num_channels, n):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.n=n # 步长
        self.m=d_model # 窗口长度

        # 线性层，假设 d_k 是每个头的维度
        self.qkv = nn.Linear(d_model, d_model * 3)

        # 输出层
        self.fc_out = nn.Linear(d_model, d_model)

    def create_mask(self, num_subsequences):
            # Create a mask of shape (num_subsequences, num_subsequences)
            mask = torch.zeros((num_subsequences, num_subsequences), dtype=torch.bool)

            for i in range(num_subsequences):
                for j in range(num_subsequences):
                    # Compare different subsequences or self-subsequence
                    start_i = i * self.n
                    end_i = start_i + self.m
                    start_j = j * self.n
                    end_j = start_j + self.m

                    # If there's an overlap or if it's the same subsequence, mark the mask as True
                    if (start_i < end_j) and (start_j < end_i) or (i == j):
                        mask[i, j] = True

            return mask

    def forward(self, x):  # x 的形状为 [3, 3, 19, 10]
        batch_size, num_dims, num_subsequences, seq_length = x.shape
        
        # 初始化输出张量
        output = torch.zeros_like(x)
        attn_weights_out=torch.zeros(batch_size,num_dims,num_subsequences,num_subsequences)
        # 循环遍历每个时间序列的维度
        for dim in range(num_dims):
            # 从 x 中提取当前维度的子序列
            x_dim = x[:, dim, :, :]  # 形状为 [3, 19, 10] ([3, 17, 20])
            print('x_dim', x_dim.shape)
            # 将 x_dim 转换为 q, k, v
            qkv = self.qkv(x_dim.float() )  # 假设有一个线性层 self.qkv
            q, k, v = qkv.chunk(3, dim=-1)
            print('q', q.shape)#[3, 17, 20]
            # 计算自注意力
            attn_weights = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力权重
            attn_weights = attn_weights / (seq_length ** 0.5)  # 缩放
            attn_weights = F.softmax(attn_weights, dim=-1)  # softmax
            print('attn_weights', attn_weights.shape)#[3, 17, 17]

            mask = self.create_mask(num_subsequences)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))  # Apply the mask


            attn_weights_out[:, dim, :, :]=attn_weights

            # 应用注意力权重到 v
            output[:, dim, :, :] = torch.matmul(attn_weights, v)  # 更新输出
            print('output[:, dim, :, :]', output[:, dim, :, :].shape)#[3, 17, 20]

        #=====
        print('attn_weights_out', attn_weights_out.shape)#[3, 3, 17, 17])==

        output=output.view(batch_size, -1, seq_length)
        print('output', output.shape)#[3, 3, 17, 20]==各个维度合为一个([3, 51, 20]
        
        #====
        # 创建一个 3x51x51 的张量，初始化为 0
        output_matrix_attn = torch.zeros(batch_size, num_subsequences*num_dims, num_subsequences*num_dims)

        # 遍历 batch，依次将 17x17 的特征图放在对角线上
        for batch_idx in range(attn_weights_out.shape[0]):
            for feature_idx in range(attn_weights_out.shape[1]):
                # 放置 17x17 特征图在 51x51 矩阵的对角线上
                start = feature_idx * num_subsequences  # 对角线的开始位置
                output_matrix_attn[batch_idx, start:start+num_subsequences, start:start+num_subsequences] = \
                attn_weights_out[batch_idx, feature_idx]

        print('Output matrix shape:', output_matrix_attn.shape)#[3, 51, 51])==
        
        return output , output_matrix_attn #attn_weights_out# 返回处理后的输出

#区分维度循环，区分维度
class MultiHeadSelfAttentiondis(nn.Module):
    def __init__(self, d_model, num_heads,num_channels, n):
        super(MultiHeadSelfAttentiondis, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.n=n # 步长
        self.m=d_model # 窗口长度

        # 线性层，假设 d_k 是每个头的维度
        self.qkv = nn.Linear(d_model, d_model * 3)

        # 输出层
        self.fc_out = nn.Linear(d_model, d_model)

    def create_mask1(self, num_subsequences):
            # Create a mask of shape (num_subsequences, num_subsequences)
            mask = torch.zeros((num_subsequences, num_subsequences), dtype=torch.bool)

            for i in range(num_subsequences):
                for j in range(num_subsequences):
                    # Compare different subsequences or self-subsequence
                    start_i = i * self.n
                    end_i = start_i + self.m
                    start_j = j * self.n
                    end_j = start_j + self.m

                    # If there's an overlap or if it's the same subsequence, mark the mask as True
                    if (start_i < end_j) and (start_j < end_i) or (i == j):
                        mask[i, j] = True

            return mask
    def create_mask(self, num_subsequences):
        # Create a mask of shape (num_subsequences, num_subsequences)
        mask = torch.zeros((num_subsequences, num_subsequences), dtype=torch.bool)
        mask.fill_(False)
        for i in range(num_subsequences):
            for j in range(num_subsequences):
                # Calculate start and end indices for subsequences
                start_i = i * self.n
                end_i = start_i + self.m
                start_j = j * self.n
                end_j = start_j + self.m

                # Calculate the overlapping range
                overlap_start = max(start_i, start_j)
                overlap_end = min(end_i, end_j)

                # Calculate the length of the overlapping region
                overlap_length = max(0, overlap_end - overlap_start)

                # Calculate the length of each subsequence
                length_i = end_i - start_i
                length_j = end_j - start_j

                # Calculate overlap ratio
                overlap_ratio = overlap_length / min(length_i, length_j)
                
                # If overlap ratio exceeds 80% or if it's the same subsequence, mark the mask as True
                if overlap_ratio >= 0.5 and overlap_ratio < 1 or i == j:#0.5
                    mask[i, j] = True
                elif overlap_ratio >= 1:
                    mask.fill_diagonal_(False)
        #mask.fill_(True) #全计算
        return mask






    def forward(self, x):  # x 的形状为 [3, 3, 19, 10]
        batch_size, num_dims, num_subsequences, seq_length = x.shape
        
        # 初始化输出张量
        output = torch.zeros_like(x)
        attn_weights_out=torch.zeros(batch_size,num_dims,num_subsequences,num_subsequences)
        # 循环遍历每个时间序列的维度
        for dim in range(num_dims):
            # 从 x 中提取当前维度的子序列
            x_dim = x[:, dim, :, :]  # 形状为 ([3, 17, 20])
            print('x_dim', x_dim.shape)
            # 将 x_dim 转换为 q, k, v
            qkv = self.qkv(x_dim.float() )  # 假设有一个线性层 self.qkv
            q, k, v = qkv.chunk(3, dim=-1)
            print('q', q.shape)#[3, 17, 20]
            # 计算自注意力
            attn_weights = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力权重
            attn_weights = attn_weights / (seq_length ** 0.5)  # 缩放
            attn_weights = F.softmax(attn_weights, dim=-1)  # softmax
            print('attn_weights', attn_weights.shape)#[3, 17, 17]

            mask = self.create_mask(num_subsequences)
            print('mask', mask.shape)
            attn_weights = attn_weights.masked_fill(mask, 0)  # Apply the mask -1e9
            print('attn_weights', attn_weights)

            attn_weights_out[:, dim, :, :]=attn_weights

            # 应用注意力权重到 v
            output[:, dim, :, :] = torch.matmul(attn_weights, v)  # 更新输出
            print('output[:, dim, :, :]', output[:, dim, :, :].shape)#[3, 17, 20]

        #=====
        print('attn_weights_out', attn_weights_out[0])#[3, 3, 17, 17])==

        output=output.view(batch_size, -1, seq_length)
        print('output', output[0])#[3, 3, 17, 20]==各个维度合为一个([3, 51, 20]
       # input()
        #====
        # 创建一个 3x51x51 的张量，初始化为 0
        output_matrix_attn = torch.zeros(batch_size, num_subsequences*num_dims, num_subsequences*num_dims)

        # 遍历 batch，依次将 17x17 的特征图放在对角线上
        for batch_idx in range(attn_weights_out.shape[0]):
            for feature_idx in range(attn_weights_out.shape[1]):
                # 放置 17x17 特征图在 51x51 矩阵的对角线上
                start = feature_idx * num_subsequences  # 对角线的开始位置
                output_matrix_attn[batch_idx, start:start+num_subsequences, start:start+num_subsequences] = \
                attn_weights_out[batch_idx, feature_idx]

        print('Output matrix shape:', output_matrix_attn.shape)#[3, 51, 51])==

        a=2
        if a==1:
    ##############
            #plot_attention_weights(output_matrix_attn, num_subsequences*num_dims)

            output_matrix_attn = output_matrix_attn[0].detach().numpy() 
            output_matrix_attn[output_matrix_attn != 0] = 1 #0
            plt.figure(figsize=(10, 8))  # 设置图形大小

            #purple_cmap = ListedColormap(['orange'])
    #==========
            # 自定义颜色映射，最低颜色为浅色，最高颜色为紫色
            cmap = sns.color_palette("Oranges", as_cmap=True).copy()
            #cmap.set_over((0.4980392156862745, 0.15294117647058825, 0.01568627450980392, 1.0))  # 设置最高值颜色

            cmap.set_over('purple')
    #===========
            
            # color_to_set = (0.4980392156862745, 0.15294117647058825, 0.01568627450980392, 1.0)

            # # 创建颜色映射
            # # 创建一个仅包含特定颜色的颜色映射
            # cmap = plt.cm.colors.ListedColormap([color_to_set])
    #=============
            sns.heatmap(output_matrix_attn, 
                annot=False,  # 关闭数值显示
                cmap=cmap,  # 颜色映射 #purple_cmap ， custom_cmap
                cbar=False,  # 关闭颜色条
                square=True,  # 使单元格为正方形
                xticklabels=False,  # 关闭 x 轴标签
                yticklabels=False,  # 关闭 y 轴标签
                linewidths=1.4,  # 设置方块轮廓线宽度
                linecolor='black'
                ,vmax=1.0)  # 设置方块轮廓颜色

            # 设置标题
            #plt.title()
            plt.savefig('selfattn.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
            # 显示图形
            plt.show()

            # # 获取值为1的区域对应的颜色代码
            # norm = plt.Normalize(vmin=0, vmax=1)  # 归一化
            # cmap = cm.get_cmap("Oranges", 256)  # 获取颜色映射

            # # 打印颜色代码
            # for i in range(output_matrix_attn.shape[0]):
            #     for j in range(output_matrix_attn.shape[1]):
            #         if output_matrix_attn[i, j] == 1:
            #             color_code = cmap(norm(output_matrix_attn[i, j]))
            #             printt(f'Value at ({i}, {j}) is 1, corresponding color code (RGBA): {color_code}')
        return output , output_matrix_attn #attn_weights_out# 返回处理后的输出

def plot_attention_weights(attention_weights, num_subsequences):
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.xticks(ticks=np.arange(num_subsequences), labels=np.arange(num_subsequences))
        plt.yticks(ticks=np.arange(num_subsequences), labels=np.arange(num_subsequences))
        plt.show()

    # 可视化第一个子序列的注意力权重
    

#不区分维度
class MultiHeadSelfAttention1(nn.Module):
    def __init__(self, dd_model, num_heads):
        super(MultiHeadSelfAttention1, self).__init__()
        assert dd_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = dd_model
        self.num_heads = num_heads
        self.head_dim =1 #dd_model // num_heads
        # self.n=n # 步长
        # self.m=m# 窗口长度

        # Linear transformations for queries, keys, and values
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)  # Project input to Q, K, V
        self.o_proj = nn.Linear(self.d_model, self.d_model)        # Output projection
        
        self.scale = self.head_dim ** -0.5               # Scaling factor for attention scores
    
    def create_mask(self, num_subsequences):
        # Create a mask of shape (num_subsequences, num_subsequences)
        mask = torch.zeros((num_subsequences, num_subsequences), dtype=torch.bool)

        for i in range(num_subsequences):
            for j in range(num_subsequences):
                # Compare different subsequences or self-subsequence
                start_i = i * self.n
                end_i = start_i + self.m
                start_j = j * self.n
                end_j = start_j + self.m

                # If there's an overlap or if it's the same subsequence, mark the mask as True
                if (start_i < end_j) and (start_j < end_i) or (i == j):
                    mask[i, j] = True

        return mask


    def forward(self, x):
        #x#[3, 3, 19, 10])
        x=x.view(x.shape[0],-1,x.shape[3])
        print('多头x', x.shape)#[3, 57, 10])
        print('dmodel',self.d_model)#10
        batch_size, num_subsequences, seq_length = x.shape
        
        # Step 1: Linear projection
        qkv = self.qkv_proj(x.float())  # Shape: (batch_size, num_subsequences, seq_length, 3 * d_model)
        print('qkv', qkv.shape)

        # Step 2: Split qkv into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Shape: (batch_size, num_subsequences, seq_length, d_model)
        print('q', q.shape) #[3, 57, 10])
        # Step 3: Reshape for multi-head attention
        q = q.view(batch_size, num_subsequences, seq_length, self.num_heads)#, self.head_dim)
        k = k.view(batch_size, num_subsequences, seq_length, self.num_heads)#, self.head_dim)
        v = v.view(batch_size, num_subsequences, seq_length, self.num_heads)#, self.head_dim)

        # Step 4: Permute to get shape (batch_size, num_heads, num_subsequences, seq_length)
        q = q.permute(0, 3, 1, 2)  # Shape: (batch_size, num_heads, num_subsequences, seq_length)
        k = k.permute(0, 3, 1, 2)  # Shape: (batch_size, num_heads, num_subsequences, seq_length)
        v = v.permute(0, 3, 1, 2)  # Shape: (batch_size, num_heads, num_subsequences, seq_length)

        # Step 5: Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, num_heads, num_subsequences, seq_length)

        mask = self.create_mask(num_subsequences)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))  # Apply the mask


        attn_scores = attn_scores / (self.head_dim ** 0.5)  # Scale the scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # Shape: (batch_size, num_heads, num_subsequences, seq_length)

        # Step 6: Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # Shape: (batch_size, num_heads, num_subsequences, head_dim)
        
        # Step 7: Reshape back to (batch_size, num_subsequences, seq_length, d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, num_subsequences, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, num_subsequences, self.d_model)  # Shape: (batch_size, num_subsequences, d_model)
        print('attn_output', attn_output.shape)
        # Step 8: Final linear layer
        output = self.o_proj(attn_output)  # Shape: (batch_size, num_subsequences, d_model)

        return output, attn_weights  # 返回最终输出和注意力权重

#无对比学习
class TimeSeriesAttentionModel1(nn.Module):
    def __init__(self, d_model, num_heads, num_channels, num_subsequences, sub_length):
        super(TimeSeriesAttentionModel, self).__init__()
        self.n = n  # Step size
        self.m = m  # Window length
        self.d_model = d_model
        self.sub_length=sub_length
        # 位置编码，针对子序列的位置
        self.position_encoding = self.create_position_encoding(num_subsequences, sub_length)#num_channels, num_subsequences,

        self.self_attention1 = MultiHeadSelfAttention1(self.sub_length, num_heads)
        self.self_attention = MultiHeadSelfAttention(self.sub_length, num_heads, num_channels)
    
    def create_position_encoding(self, num_subsequences, sub_length):
            # 创建位置编码矩阵
            position_encoding = np.zeros((1, 1, num_subsequences, sub_length))

            # 生成位置编码
            for pos in range(num_subsequences):
                for i in range(sub_length):
                    # 根据公式生成正弦和余弦位置编码
                    if i % 2 == 0:
                        position_encoding[0, 0, pos, i] = np.sin(pos / (10000 ** (i / sub_length)))
                    else:
                        position_encoding[0, 0, pos, i] = np.cos(pos / (10000 ** (i / sub_length)))

            return position_encoding


    def forward(self, x):
        print('子序列', x.shape)#([3, 3, 19, 10])
        batch_size, num_channels, num_subsequences, sub_length = x.shape

        # 1. Extract subsequences
        subsequences =x# self.extract_subsequences(x)  # Shape: (batch_size, num_channels, num_subsequences, sub_length)
        print('subsequences', subsequences.shape)  # e.g. [3, 3, 19, 10]
        num_subsequences = subsequences.shape[2]
   

        # 2. Add position encoding (Ensure broadcasting is correct)
        subsequences = subsequences # Add extra dimension for d_model
        print('subsequences0', subsequences.shape)#[3, 3, 19, 10])
        print('self.position_encoding', self.position_encoding.shape)#(1, 1, 19, 10)
        subsequences = subsequences + self.position_encoding  # Shape: (batch_size, num_channels, num_subsequences, m, d_model)
        print('subsequences相加', subsequences.shape)#[3, 3, 19, 10])

        # 3. Apply self-attention
         # Reshape to (batch_size, num_subsequences, sub, d_model)
        #subsequences = subsequences.view(batch_size, -1, self.sub_length) 
        print('subsequences2', subsequences.shape)#([3, 19, 10, 3]) / [3, 57, 10])
        
        output,attn = self.self_attention(subsequences)  # Shape: (batch_size, num_subsequences, sub, d_model)

        return output,attn
    

#对比学习,不区分维度
class TimeSeriesAttentionModel2(nn.Module):
    def __init__(self, d_model, num_heads, num_channels, num_subsequences, sub_length,num_classes,n,top_k):
        super(TimeSeriesAttentionModel, self).__init__()
        self.n = n  # Step size
        # self.m = m  # Window length
        self.sub_length=sub_length

        self.top_k=top_k

        # 位置编码，针对子序列的位置
        self.position_encoding = self.create_position_encoding(num_subsequences, sub_length)#num_channels, num_subsequences,

        self.self_attention1 = MultiHeadSelfAttention1(self.sub_length, num_heads)
        self.self_attention = MultiHeadSelfAttention(self.sub_length, num_heads, num_channels,self.n)

        # 分类头
        self.classifier = nn.Linear(num_subsequences*num_channels, num_classes)
        # Triplet Loss 的距离度量函数
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def create_position_encoding(self, num_subsequences, sub_length):
            # 创建位置编码矩阵
            position_encoding = np.zeros((1, 1, num_subsequences, sub_length))

            # 生成位置编码
            for pos in range(num_subsequences):
                for i in range(sub_length):
                    # 根据公式生成正弦和余弦位置编码
                    if i % 2 == 0:
                        position_encoding[0, 0, pos, i] = np.sin(pos / (10000 ** (i / sub_length)))
                    else:
                        position_encoding[0, 0, pos, i] = np.cos(pos / (10000 ** (i / sub_length)))

            return position_encoding


    def forward(self, x):
        print('子序列', x.shape)#([3, 3, 19, 10])
        batch_size, num_channels, num_subsequences, sub_length = x.shape

        # 1. Extract subsequences
        subsequences =x # self.extract_subsequences(x)  # 
        print('subsequences', subsequences.shape)  # e.g. [3, 3, 19, 10]
        num_subsequences = subsequences.shape[2]
   

        # 2. Add position encoding (Ensure broadcasting is correct)
        subsequences = subsequences = subsequences.to(torch.float32) # 
        print('subsequences0', subsequences.shape)#[3, 3, 19, 10])

        print('self.position_encoding', self.position_encoding.shape)#(1, 1, 19, 10)
        subsequences = subsequences + self.position_encoding  # 
        print('subsequences相加', subsequences.shape)#[3, 3, 19, 10])

        # 3. Apply self-attention
        print('subsequences2', subsequences.shape)#
        
        output,attn = self.self_attention(subsequences)  
        print('OUTPUT',output.shape)#([3, 3, 17, 20]) 、 [3, 51, 20]) 
        #attn 3，51，51

        #=======
        #正负样本构造
        triplets=self.attention_weight_based_triplet(batch_size,num_subsequences,output,attn)
        

        #====对比学习
       # 使用 Triplet Loss 的自监督任务
        triplet_loss = self.compute_triplet_loss(triplets)

        #====选择最重要的
        # 计算平均注意力权重
        mean_attention_weights = output.mean(dim=2)
        print('mean_attention_weights',mean_attention_weights.shape)#[3, 57])
        # 挑选出具有最高平均注意力权重的子序列
        print('topk', self.top_k)

        important_indices = torch.topk(mean_attention_weights, self.top_k, dim=-1).indices
        print('important_indices', important_indices.shape)#[15, 58]

        # 输出具有区别能力的子序列
        expanded_indices = important_indices.unsqueeze(-1)

        # 使用 torch.gather 来选择重要子序列
        subsequences=subsequences.view(batch_size,-1,sub_length)
        important_subsequences = subsequences.gather(1, expanded_indices.expand(-1, -1, subsequences.size(-1)))

        #important_subsequences = subsequences.reshape(batch_size,-1,sub_length)[important_indices.numpy()]
        print('important_subsequences', important_subsequences.shape)#(

        #====
        # 分类输出
        # print(num_subsequences*num_channels)
        # print('output.mean(dim=2)',output.mean(dim=2).shape)
        # print(output.mean(dim=2).dtype)  # 输出 input 的数据类型
        # print(self.classifier.weight.dtype)  # 输出 classifier 的权重数据类型
        
        #classification_output = self.classifier(output.mean(dim=2).float())  # 取子序列的平均作为分类输入

        return important_subsequences, triplet_loss
    def compute_triplet_loss(self,triplets):
        loss = 0.0
        for anchor, positive, negative in triplets:
            loss += self.triplet_loss_fn(anchor, positive, negative)
        
        return loss / len(triplets) 

    def attention_weight_based_triplet(self,batch_size,num_subsequences, attention_output, attention_weights):
        triplets = []
        # batch_size, num_channels, num_subsequences, sub_length = batch_subsequences.shape

        # 1. 计算自注意力特征和权重
        # attention_output, attention_weights = attention_output, attention_weights

        #attention_output, attention_weight
        #[3, 51, 20]) #attn 3，51，51
        
        for i in range(batch_size):
            anchor_idx = np.random.randint(0, num_subsequences)
            anchor = attention_output[i, anchor_idx,:]
            anchor_weights = attention_weights[i, anchor_idx,:]

            # 2. 计算正负样本
            #positive_idx = np.argsort(anchor_weights)[-2]  # 选择权重最高的样本
            positive_idx = np.argsort(anchor_weights.detach().numpy())[-2]

            positive = attention_output[i,positive_idx, :]

            negative_idx = np.argsort(anchor_weights.detach().numpy())[0]  # 选择权重最低的样本
            negative = attention_output[i,positive_idx, :]

            triplets.append((anchor, positive, negative))

        return triplets


#对比学习,区分维度
class TimeSeriesAttentionModel(nn.Module):
    def __init__(self, d_model, num_heads, num_channels, num_subsequences, sub_length,num_classes,n,top_k):#
        super(TimeSeriesAttentionModel, self).__init__()
        self.n = n  # Step size
        # self.m = m  # Window length
        self.sub_length=sub_length

        self.top_k=top_k

        # 位置编码，针对子序列的位置
        self.position_encoding = self.create_position_encoding(num_subsequences, sub_length)#num_channels, num_subsequences,

        self.self_attention = MultiHeadSelfAttentiondis(self.sub_length, num_heads, num_channels,self.n)

        # 分类头
        self.classifier = nn.Linear(num_subsequences*num_channels, num_classes)
        # Triplet Loss 的距离度量函数
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def create_position_encoding(self, num_subsequences, sub_length):
            # 创建位置编码矩阵
            position_encoding = np.zeros((1, 1, num_subsequences, sub_length))

            # 生成位置编码
            for pos in range(num_subsequences):
                for i in range(sub_length):
                    # 根据公式生成正弦和余弦位置编码
                    if i % 2 == 0:
                        position_encoding[0, 0, pos, i] = np.sin(pos / (10000 ** (i / sub_length)))
                    else:
                        position_encoding[0, 0, pos, i] = np.cos(pos / (10000 ** (i / sub_length)))

            return position_encoding


    def forward(self, x):
        print('子序列', x.shape)#([3, 3, 19, 10])
        batch_size, num_channels, num_subsequences, sub_length = x.shape

        # 1. Extract subsequences
        subsequences =x # self.extract_subsequences(x)  # 
        print('subsequences', subsequences.shape)  # e.g. [3, 3, 19, 10]
        num_subsequences = subsequences.shape[2]
   

        # 2. Add position encoding (Ensure broadcasting is correct)
        subsequences =  subsequences.to(torch.float32) # 
        print('subsequences0', subsequences.shape)#[3, 3, 19, 10])

        print('self.position_encoding', self.position_encoding.shape)#(1, 1, 19, 10)
        subsequences = subsequences + self.position_encoding  # 
        print('subsequences相加', subsequences.shape)#[3, 3, 19, 10])

        # 3. Apply self-attention
        print('subsequences2', subsequences.shape)#
        
        output,attn = self.self_attention(subsequences)  
        printt('OUTPUT attn',output.shape, attn.shape)
        # e([15, 232, 64]) torch.Size([15, 232, 232])#ttn torch.Size([3, 18, 30]) torch.Size([3, 18, 18]
    

#         #====选择最重要的
#         # 计算平均注意力权重
#         outputt=output.view(batch_size, num_channels, num_subsequences, sub_length)#[15, 2, 116, 64])
#         mean_attention_weights = outputt.mean(dim=-1)#[15, 2, 116
#         print('mean_attention_weights',mean_attention_weights.shape)
#         # 挑选出具有最高平均注意力权重的子序列
#         print('topk', self.top_k)

#         important_indices = torch.topk(mean_attention_weights, self.top_k, dim=-1).indices
#         print('important_indices', important_indices.shape)#([15, 2, 58])
# #====
#         # 确保 important_indices 是 long 型
#         important_indices = important_indices.long()

#         # 扩展 important_indices 以适应 subsequences 的最后一个维度
#         # 需要在最后一维加上一个维度
#         expanded_indices = important_indices.unsqueeze(-1)  # 形状变为 [15, 2, 58, 1]

#         # 使用 torch.gather 在时间序列维度（第三维）上选择子序列
#         # expanded_indices 要扩展到 [15, 2, 58, 64]，以匹配 subsequences
#         important_subsequences = subsequences.gather(2, expanded_indices.expand(-1, -1, -1, subsequences.size(-1)))

#         print('important_subsequences', important_subsequences.shape)#[15, 2, 38, 64])

        #=======
        #正负样本构造
        #triplets=self.attention_weight_based_triplet(batch_size,num_subsequences,output,attn)
        #printt('triplets', len(triplets))#15
        #input()

        #====对比学习
       # 使用 Triplet Loss 的自监督任务
        #triplet_loss = self.compute_triplet_loss(triplets)

        #====
        # 分类输出
        # print(num_subsequences*num_channels)
        # print('output.mean(dim=2)',output.mean(dim=2).shape)
        # print(output.mean(dim=2).dtype)  # 输出 input 的数据类型
        # print(self.classifier.weight.dtype)  # 输出 classifier 的权重数据类型
        
        #classification_output = self.classifier(output.mean(dim=2).float())  # 取子序列的平均作为分类输入

        return output,attn
    def compute_triplet_loss(self,triplets):
        loss = 0.0
        for anchor, positive, negative in triplets:
            loss += self.triplet_loss_fn(anchor, positive, negative)
        
        return loss / len(triplets) 

    def attention_weight_based_triplet(self, batch_size, num_subsequences, attention_output, attention_weights):
        print('attention_weights', attention_weights[0])
        triplets = []

        #print('OUTPUT attn',output.shape, attn.shape)
        # e([15, 232, 64]) torch.Size([15, 232, 232])

        for i in range(batch_size):
            # 随机选择 anchor
            anchor_idx = np.random.randint(0, num_subsequences)
            anchor = attention_output[i, anchor_idx, :]
            anchor_weights = attention_weights[i, anchor_idx, :]

            # 选择正样本 (权重第二高的)
            positive_idx = torch.argsort(anchor_weights)[-2].item()
            while positive_idx == anchor_idx:
                positive_idx = torch.argsort(anchor_weights)[-3].item()

            positive = attention_output[i, positive_idx, :]

            # 选择负样本 (权重最低的)
            negative_idx = torch.argsort(anchor_weights)[0].item()
            while negative_idx == anchor_idx:
                negative_idx = torch.argsort(anchor_weights)[1].item()

            negative = attention_output[i, negative_idx, :]

            # 保存三元组
            triplets.append((anchor, positive, negative))

        return triplets

    def attention_weight_based_triplet1(self,batch_size,num_subsequences, attention_output, attention_weights):
        triplets = []
        #print('OUTPUT attn',output.shape, attn.shape)
        # e([15, 232, 64]) torch.Size([15, 232, 232])
        for i in range(batch_size):
            anchor_idx = np.random.randint(0, num_subsequences)
            anchor = attention_output[i, anchor_idx,:]
            anchor_weights = attention_weights[i, anchor_idx,:]

            # 2. 计算正负样本
            #positive_idx = np.argsort(anchor_weights)[-2]  # 选择权重最高的样本
            positive_idx = np.argsort(anchor_weights.detach().numpy())[-2]

            positive = attention_output[i,positive_idx, :]

            negative_idx = np.argsort(anchor_weights.detach().numpy())[0]  # 选择权重最低的样本
            negative = attention_output[i,negative_idx, :]

            triplets.append((anchor, positive, negative))

        return triplets



def extract_subsequences( x,m,n):
        
        batch_size, num_channels, seq_length = x.shape
        subsequences = []
        
        for i in range(0, seq_length - m + 1, n):
            subsequence = x[:, :, i:i+m]
            subsequences.append(subsequence)

        subsequences = torch.stack(subsequences, dim=2)
        # subsequences = np.stack(subsequences, axis=2)  # Shape: (batch_size, num_channels, num_subsequences, m)
        return subsequences
a=1
if a==1:
    # Parameters
    batch_size, num_channels, seq_length = 3, 3, 100
    # tsd_model = 192 # Model dimension
    num_heads = 1
    n = 13 # Step size
    m = 30  # Window length
    num_classes=2

    # Example input
    x = torch.randn(batch_size, num_channels, seq_length)

    topk=5

    subsequences = extract_subsequences(x,m,n) 
    printt('subsequences',subsequences.shape)#[3, 3, 6, 30]

    batch_size, num_channels, num_subsequences, sub_length= \
    subsequences.shape[0],subsequences.shape[1],subsequences.shape[2],subsequences.shape[3]
    tsd_model=sub_length

    # Model
    model = TimeSeriesAttentionModel(tsd_model, num_heads, num_channels, num_subsequences, sub_length,num_classes,n,topk)
    output , attn= model(subsequences)

    printt('output', output.shape)  # [3, 18, 30])




#===============================
a=2
if a==1:
        # 计算平均注意力权重
        mean_attention_weights = output.mean(dim=2)
        print('mean_attention_weights',mean_attention_weights.shape)#[3, 57])
        # 挑选出具有最高平均注意力权重的子序列
        top_k = 3  # 选择前3个子序列
        important_indices = torch.topk(mean_attention_weights[0], top_k, dim=0).indices.flatten()
        print('important_indices', important_indices)#[12, 19, 35])

        # 输出具有区别能力的子序列
        important_subsequences = subsequences.reshape(batch_size,-1,sub_length)[0][important_indices.numpy()]
        print("Most Important Subsequences:")
        print('important_subsequences', important_subsequences.shape)#([3, 10])

        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # 可视化注意力权重
        attn=attn.squeeze()
        print('attn', attn.shape)#([3, 1, 57, 57])
        def plot_attention_weights1(attention_weights, num_subsequences):
            plt.figure(figsize=(10, 8))
            # Use 'viridis_r' to reverse the color map
            plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis_r', aspect='auto')
            plt.colorbar()
            plt.title('Attention Weights')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            plt.xticks(ticks=np.arange(num_subsequences), labels=np.arange(num_subsequences))
            plt.yticks(ticks=np.arange(num_subsequences), labels=np.arange(num_subsequences))
            plt.show()

        def plot_attention_weights(attention_weights, num_subsequences):
            plt.figure(figsize=(10, 8))
            plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Attention Weights')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            plt.xticks(ticks=np.arange(num_subsequences), labels=np.arange(num_subsequences))
            plt.yticks(ticks=np.arange(num_subsequences), labels=np.arange(num_subsequences))
            plt.show()

        # 可视化第一个子序列的注意力权重
        plot_attention_weights(attn, num_subsequences*num_channels)