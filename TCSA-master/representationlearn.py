import random
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

from losses.triplet_loss import TripletLoss_ts
import utils
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse

from readuea import load_UEA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            mask = self.create_mask(num_subsequences).to(device)
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

    

#对比学习,区分维度
class TimeSeriesAttentionModel(nn.Module):
    def __init__(self, num_heads, num_channels, num_subsequences, sub_length,num_classes,n,top_k):#
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
        self.lin1 = nn.Linear(num_subsequences*num_channels*sub_length, 200)
        
        self.norm1 = nn.LayerNorm(sub_length)
        self.ffn = nn.Sequential(
            nn.Linear(sub_length, sub_length * 4),
            nn.ReLU(),
            nn.Linear(sub_length * 4, sub_length)
        )
        self.norm2 = nn.LayerNorm(sub_length)
    def create_position_encoding_np(self, num_subsequences, sub_length):
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

    import torch

    def create_position_encoding(self, num_subsequences, sub_length):
        # 创建位置编码矩阵（直接使用 PyTorch 张量）
        position_encoding = torch.zeros(1, 1, num_subsequences, sub_length, dtype=torch.float32)

        # 生成位置编码
        for pos in range(num_subsequences):
            for i in range(sub_length):
                # 根据公式生成正弦和余弦位置编码
                if i % 2 == 0:
                    position_encoding[0, 0, pos, i] = torch.sin(torch.tensor(pos) / (10000 ** (i / sub_length)))
                else:
                    position_encoding[0, 0, pos, i] = torch.cos(torch.tensor(pos) / (10000 ** (i / sub_length)))

        return position_encoding


    def forward(self, x):
        print('自注意力机制子序列', x.shape)#([3, 3, 19, 10])
        batch_size, num_channels, num_subsequences, sub_length = x.shape

        # 1. Extract subsequences
        subsequences =x # self.extract_subsequences(x)  # 
        print('subsequences', subsequences.shape)  # e.g. [3, 3, 19, 10]
        num_subsequences = subsequences.shape[2]
   

        # 2. Add position encoding (Ensure broadcasting is correct)
        subsequences =  subsequences.to(torch.float32) # 
        print('subsequences0', subsequences.shape)#[3, 3, 19, 10])

        print('self.position_encoding', self.position_encoding.shape)#(1, 1, 19, 10)
        subsequences = subsequences + self.position_encoding.to(device)  # 
        print('subsequences相加', subsequences.shape)#[3, 3, 19, 10])

        # 3. Apply self-attention
        print('subsequences2', subsequences.shape)#
        
        attn_output,attn = self.self_attention(subsequences)  
        print('selfOUTPUT attn', attn_output.shape, attn.shape)#selfOUTPUT attn torch.Size([4, 10, 50]) torch.Size([4, 10, 10])

        attn_output = attn_output.to(torch.float32)  # 或者根据需要使用 bfloat16
        subsequences = subsequences.to(torch.float32)
        output = self.norm1(subsequences.view(batch_size, -1, sub_length) + attn_output) 
        print('output1', output.shape)

        # 3. Feed-forward network
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)  # Residual connection and Layer Normalization
        print('output2', output.shape)

        output=output.view(batch_size, -1)
        output=output.float()
        
        output=self.lin1(output)
        print(output.shape)
        #input()
        return output#,attn
    



def extract_subsequences( x,m,n):
        
        batch_size, num_channels, seq_length = x.shape
        subsequences = []
        
        for i in range(0, seq_length - m + 1, n):
            subsequence = x[:, :, i:i+m]
            subsequences.append(torch.tensor(subsequence))

        subsequences = torch.stack(subsequences, dim=2)
        #subsequences = np.stack(subsequences, axis=2)  # Shape: (batch_size, num_channels, num_subsequences, m)
        return subsequences

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def train_model(X, y, dataset_name, len_train, len_test):
    # 创建模型和优化器
    c,l=X.shape[1], X.shape[2]
    #model = ConvPoolModel(num_layers=2, c=c,l=l)
    lr=0.0001
    optimizer = torch.optim.Adam(model_ts.parameters(), lr=lr)

    # 模拟对比学习过程
    num_epochs = 50
    batch_size = 4
    best_loss = float('inf')  
    
    # 保存模型的路径
   # best_model_path = os.path.join('model', f'best_conv_pool_model_{dataset_name}.pth')
    

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32).to(device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long).to(device)
    printt('标签y', y.shape)
    # 创建 TensorDataset 和 DataLoader
    # traindataset = TensorDataset(X)
    # dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=False)

    train_torch_dataset = utils.Dataset(X)
    dataloader = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=batch_size, shuffle=False
        )
    printt('数据X', X.shape)#torch.Size([30, 2, 30, 50])
    best_accuracy = 0  # 初始化最佳验证精度
    for epoch in range(num_epochs):
        val_rep=[]
        for batch in dataloader:
            batch = batch.to(device)
            #printt('batch', batch.shape)
            loss, rep = loss_fn( batch, model_ts, X.to(device), save_memory=save_memory)
            print('rep',rep.shape)#([4, 200])
            loss.backward()
            optimizer.step()
            val_rep.append(rep)
        if (epoch + 1) % 10 == 0:
            printt(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 使用标签 y 进行验证，记录表示和验证精度
        model_ts.eval()
        # 禁用梯度计算
        with torch.no_grad():

            val_rep = torch.cat(val_rep, dim=0)  # 拼接成完整 tensor
            val_rep = val_rep.cpu().numpy()  # 将特征表示从Tensor转换为Numpy数组，以便适配SVM
            y_val = y.cpu().numpy()  # 将标签从Tensor转换为Numpy数组

            # # 训练SVM分类器
            # svm_classifier = SVC(kernel='linear')  # 可根据需求选择其他核函数
            # svm_classifier.fit(val_rep, y_val)

            # # 预测验证集
            # val_preds = svm_classifier.predict(val_rep)
            # val_accuracy = accuracy_score(y_val, val_preds)
            # printt('val_accuracy', val_accuracy)


            # 1. 分割训练集和验证集
            train_rep = val_rep[:len_train]  # 训练集表示
            train_labels = y[:len_train]      # 训练集标签

            val_rep_split = val_rep[len_train:len_train + len_test]  # 验证集表示
            val_labels = y[len_train:len_train + len_test]           # 验证集标签

            # 2. 训练 SVM 分类器
            svm_classifier = SVC(kernel='linear')  # 可根据需求选择其他核函数
            
            train_rep = train_rep
            train_labels = train_labels.cpu().numpy()  # 移动到 CPU 并转换为 numpy 数组
            svm_classifier.fit(train_rep, train_labels)

            # 3. 使用 SVM 对验证集进行预测
            val_preds = svm_classifier.predict(val_rep_split)

            val_labels = val_labels.cpu().numpy()  # 如果 val_labels 是 GPU 上的张量
            val_preds = val_preds
            # 4. 计算验证集的准确率
            val_accuracy = accuracy_score(val_labels, val_preds)
            printt('验证集准确率:', val_accuracy)

            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

                best_val_rep = val_rep  # 更新最佳表示
                os.makedirs(os.path.dirname(best_rep_path), exist_ok=True)
                torch.save(best_val_rep, best_rep_path)  # 保存最佳表示到文件
                
               # print(f'最佳表示已保存，验证精度: {best_accuracy:.4f}')
                #os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                #torch.save(model_ts.state_dict(), best_model_path)

                #print(f'最佳模型已保存，验证精度: {best_accuracy:.4f}')
        # # 清空 val_rep，准备下一轮 epoch 的表示
        # val_rep.clear()
    return best_val_rep  #model_ts, best_accuracy
#from soft_dtw import SoftDTW #no cuda
from soft_dtw_cuda_ori import SoftDTW #cuda
def get_dtw1(X):
    # criterion = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()
    # ...
    # loss = criterion(out, target)

    soft_dtw = SoftDTW(gamma=1.0, normalize=False)
    X = X.copy(order='C').astype(np.float64)
    X[np.isnan(X)] = 0
    distances = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    for i in range(len(X)):
        for j in range(len(X)):
            data = X[i]
           # print('data',data.shape)
            query = X[j]
            # 定义优化器（比如 Adam）
            optimizer = optim.Adam([data], lr=0.01)

            # 定义训练 epoch 数量
            num_epochs = 100

            # 训练循环
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # 清除上一次迭代的梯度

                # 计算 SoftDTW 距离作为损失
                distance = soft_dtw(data, query)

                # 假设 SoftDTW 距离是我们要最小化的损失
                loss = distance.sum()

                # 打印每个 epoch 的损失
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

                # 反向传播计算梯度
                loss.backward()

                # 优化更新 x 的值
                optimizer.step()

            # 打印优化后的 x
            print("优化后的 distance:", distance)
        print(f"第: {i} 个")
    print('done')
    return distances

import torch
import numpy as np
#from soft_dtw import SoftDTW 

def get_dtw(X):
    # 初始化 SoftDTW 距离函数
    soft_dtw = SoftDTW(True, gamma=1.0, normalize=True)

    # 将数据转换为 PyTorch 张量并移到 GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to('cuda')  # 将整个数据矩阵移动到 GPU 上

    # 初始化相似矩阵
    num_samples = X_tensor.shape[0]
    distances = torch.zeros((num_samples, num_samples), dtype=torch.float32, device='cuda')  # 在 GPU 上初始化

    # 遍历所有样本对
    for i in range(num_samples):
        for j in range(i, num_samples):
            # 获取当前两个样本的张量
            data = X_tensor[i].unsqueeze(0)  # 增加 batch 维度
            query = X_tensor[j].unsqueeze(0)

            # 计算 SoftDTW 距离
            distance = soft_dtw(data, query)

            # 更新距离矩阵
            distances[i, j] = distance
            distances[j, i] = distance  # 对称赋值

        print(f"Processed sample {i+1}/{num_samples}")

    print('Distance matrix computation completed.')
    return distances.cpu().numpy()  # 计算完成后将结果从 GPU 移回 CPU，并转换为 NumPy 数组




#===================
#dataset_dir = './datasets/UCRArchive_2018'
root=r'/root'
output_dir =os.path.join(root,'HGRL-master_root','TCSA-master','dtw') #r'H:\HGRL-master\TCSA-master\dtw'
#I:\D\桌面\多元时间序列数据集\Multivariate_arff
def argsparser():
    parser = argparse.ArgumentParser("tw creator")
    parser.add_argument('--dataset', help='Dataset name', default='AtrialFibrillation')#HAR
   # a=os.path.join()#r'I:/D/桌面/多元时间序列数据集/Multivariate_arff'
    b=os.path.join(root,'HGRL-master_root','sub_shapelets-master','demo')#r'H:/HGRL-master/sub_shapelets-master/demo'
    #a=os.path.join(I:/D/桌面/多元时间序列数据集/Multivariate_arff)
    parser.add_argument('--data_path', type=str, default=b)  # ./data

    parser.add_argument('--cache_path', type=str, default='H:\HGRL-master\TCSA-master\cache_path')
    return parser

if __name__ == "__main__":
    
    print(f"Using device: {device}")
#=====
    # Parameters
    #batch_size, num_channels, seq_length = 5, 8, 100 #5, 3, 100
    # tsd_model = 192 # Model dimension
    num_heads = 1
    n = 20 # Step size
    m = 50  # Window length
    num_classes=2

    # Example input
    #x = torch.randn(batch_size, num_channels, seq_length) #模拟数据

# Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    result_dir = os.path.join(output_dir, 'datasets_dtw')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    #1
    #X = HAR()##(10299, 9, 128)
    #visualize_time_series(X, sample_idx=0, feature_idx=0)
    #visualize_multiple_dimensions(X, sample_idx=0, feature_indices=[0, 1, 2,3,4,5,6,7,8])
    # visualize_heatmap(X, sample_idx=0)
    # interactive_plot(X, sample_idx=0, feature_idx=0)
    
    #2
    #X = np.random.rand(30, 3, 50)

    #3
    X, y, len_train, len_test=load_UEA(args.dataset, args)
    X=X.reshape(X.shape[0], X.shape[2], X.shape[1])
    printt('初始数据', X.shape)

#+===========
    topk=5

    subsequences = extract_subsequences(X,m,n) 
    printt('生成子序列subsequences',subsequences.shape)#[5, 3, 6, 30]

    batch_size, num_channels, num_subsequences, sub_length= \
    subsequences.shape[0],subsequences.shape[1],subsequences.shape[2],subsequences.shape[3]
    tsd_model=sub_length

    #======
    # Randomly choose m dimensions from C dimensions
    C=num_channels
    max_value = C // 3  # 计算最大值
    # m = random.randint(1, max_value)  # 在 1 和 C//3 之间选择随机整数
    # selected_dims = random.sample(range(C), m)  # Randomly select m dimensions
    selected_dims=max(1,max_value)
    #======
    # Model
    #model = TimeSeriesAttentionModel(tsd_model, num_heads, num_channels, num_subsequences, sub_length,num_classes,n,topk)
    model_ts = TimeSeriesAttentionModel(num_heads, selected_dims, num_subsequences // 3, sub_length,num_classes,n,topk)
    model_ts=model_ts.to(device)
    # output , attn= model(subsequences)

    # printt('output', output.shape)  # [3, 18, 30])

    compared_length='Infinity'
    nb_random_samples=5
    negative_penalty=1
    loss_fn = TripletLoss_ts(
            compared_length, nb_random_samples, negative_penalty
        )
     
    train_subseq=subsequences
    save_memory=False
    
    max_score = 0
    i = 0  # Number of performed optimization steps
    epochs = 0  # Number of performed epochs
    count = 0  # Count of number of epochs without improvement
    # Will be true if, by enabling epoch_selection, a model was selected
    # using cross-validation
    found_best = False
    nb_steps=10
    verbose=False
    
    varying=False

#======

    #dataset='HAR'
    # 模型保存路径
    best_rep_path = os.path.join('representation', f'best_rep_{args.dataset}.pt')


    #X_tensor = torch.tensor(train_subseq, dtype=torch.float32)
    X_tensor = train_subseq.clone().detach().float()
    X_tensor=X_tensor#.to(device)
    y=y#.to(device)
    
    if os.path.exists(best_rep_path):
        X_rep=torch.load(best_rep_path)
        printt("加载已有表示", X_rep.shape)#(30, 200)
    else:
        X_rep = train_model(X_tensor,y, args.dataset, len_train, len_test)  # 训练模型并保存

    if isinstance(X_rep, np.ndarray):
        X_rep = torch.from_numpy(X_rep)
    #X_rep=X_rep.unsqueeze(dim=2)
    #X=X.reshape(X.shape[0], X.shape[2], X.shape[1])
    X_rep=X_rep.detach().numpy()
    printt('表示数据维度', X_rep.shape)#
    #input()
    #==== 归一化
    #gpu
    X_rep = torch.from_numpy(X_rep).float()
    X_rep = X_rep.unsqueeze(-1)
    dtw_arr=get_dtw(X_rep)
    print('dtw_arr', dtw_arr)
    
    #cpu
    #dtw_arr = get_dtw(X_rep)
    
    #====
    min_distance = dtw_arr.min()
    # printt('min_distance', min_distance)
    max_distance = dtw_arr.max()


    dtw_arr = (dtw_arr - min_distance) / (max_distance - min_distance)
    #=========

    # 将 DTW 距离转换为相似度值
    alpha = 0.3  # 调整 alpha 参数
    similarity_matrix = np.zeros_like(dtw_arr)

    for i in range(dtw_arr.shape[0]):
        for j in range(dtw_arr.shape[0]):
            similarity_matrix[i, j] = 1 / np.exp(alpha * dtw_arr[i, j]) #+1 #np.exp

    #====
    # 归一化
    # min_distance = similarity_matrix.min()
    # max_distance = similarity_matrix.max()


    # similarity_matrix = (similarity_matrix - min_distance) / (max_distance - min_distance)
 
    #==== 标准化，会出现负值
    # # 计算均值和标准差
    # mean_distance = dtw_arr.mean()
    # std_distance = dtw_arr.std()
    # standardized_dtw_distance_matrix = (dtw_arr - mean_distance) / std_distance
 
    #===
    printt('dtw_attr similarity_matrix',similarity_matrix)
    printt('dtw_attr similarity_matrix',similarity_matrix.shape)
    np.save(os.path.join(result_dir, args.dataset), similarity_matrix)
  

