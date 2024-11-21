import os
from print import print
from printt import printt
import numpy as np
def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

import torch
import numpy as np
import os
def convolve_shapelets_with_torch(X, shapelets_data, epsilon):
    # 获取数据的形状
    num_samples, seq_length = X.shape
    num_shapelets,  shapelet_length = shapelets_data.shape

    # 用于存储每个样本与每个形状提取物的相似性
    similarity_indices = [[] for _ in range(num_samples)]

    # 将数据转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)  # (num_samples, num_channels, seq_length)
    shapelets_tensor = torch.tensor(shapelets_data, dtype=torch.float32)  # (num_shapelets, num_channels, shapelet_length)

     # 遍历每个样本
    for sample_idx in range(num_samples):
        # 获取当前样本的时间序列
        time_series = X_tensor[sample_idx].unsqueeze(0).unsqueeze(0)  # 形状为 (num_channels, seq_length)
        print('时间序列', time_series.shape)
        # 对每个形状提取物进行卷积
        for shapelet_idx in range(num_shapelets):
            shapelet = shapelets_tensor[shapelet_idx].unsqueeze(0).unsqueeze(0) # 形状为 (1, shapelet_length)
            
            print('shapelet',shapelet.shape)

            # 使用 conv1d 进行卷积操作，注意这里的输入维度

            C = torch.nn.functional.conv1d(time_series, shapelet, stride=1)
            
            # 获取卷积结果并检查阈值
            C_values = C.squeeze().detach()# 转换为 NumPy 数组并去掉多余的维度
            print('C_values',C_values.shape) # (1217,)
            #input()

           # 找到 C_values 中的最大值及其索引
            # 找到 C_values 中的最大值及其索引
            max_value, max_index = torch.max(C_values, dim=0)   

            # 验证最大值是否小于阈值
            #每个样本对应的shapelet的相似度
            if max_value < epsilon:
                similarity_indices[sample_idx].append((shapelet_idx))
            else:
                similarity_indices[sample_idx].append(0)

    return similarity_indices

def convolve_shapelets_with_torch1(X, shapelets_data, epsilon):
    # 获取数据的形状
    num_samples, num_channels, seq_length = X.shape
    num_shapelets,  shapelet_length = shapelets_data.shape

    # 用于存储每个样本与每个形状提取物的相似性
    similarity_indices = [[] for _ in range(num_samples)]

    # 将数据转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)  # (num_samples, num_channels, seq_length)
    shapelets_tensor = torch.tensor(shapelets_data, dtype=torch.float32)  # (num_shapelets, num_channels, shapelet_length)

     # 遍历每个样本
    for sample_idx in range(num_samples):
        # 获取当前样本的时间序列
        time_series = X_tensor[sample_idx].unsqueeze(0)   # 形状为 (num_channels, seq_length)
        print('时间序列', time_series.shape)
        # 对每个形状提取物进行卷积
        for shapelet_idx in range(num_shapelets):
            shapelet = shapelets_tensor[shapelet_idx].unsqueeze(0)  # 形状为 (1, shapelet_length)
            
            shapelet = shapelet.repeat(num_channels,  1).unsqueeze(0) 
            print('shapelet',shapelet.shape)

            # 使用 conv1d 进行卷积操作，注意这里的输入维度
            C = torch.nn.functional.conv1d(time_series, shapelet, stride=1)  # (1, num_channels, output_length)
           # C = torch.nn.functional.conv1d(time_series, shapelet.unsqueeze(0), stride=1)
            
            # 获取卷积结果并检查阈值
            C_values = C.squeeze().detach().numpy()  # 转换为 NumPy 数组并去掉多余的维度
            print('C_values',C_values.shape)
            input()
            # 检查每个位置的卷积结果
            for i in range(len(C_values)):
                if C_values[i] < epsilon:
                    similarity_indices[sample_idx].append((shapelet_idx, i))

    return similarity_indices

shapelets=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\shapelets'
dataset='AtrialFibrillation'
data_dir=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo'
X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))
X=X.transpose(0, 2, 1)
print('数据X',X.shape)# (30, 2, 640)
X=X.reshape(X.shape[0], -1)
#====
shapelets_path = os.path.join(shapelets, dataset+'_shapelets.npy')
# 加载 .npy 文件
shapelets_data = np.load(shapelets_path)

# 输出数据的维度
print("Shape of shapelets data:", shapelets_data.shape)# (6, 2, 64)
shapelets_data=shapelets_data.reshape(-1,shapelets_data.shape[-1])
#====
# 设置阈值
epsilon = 5  # 这是一个示例值，根据需要调整

# 进行卷积操作
similarity_indices = convolve_shapelets_with_torch(X, shapelets_data, epsilon)

printt('similarity_indices', np.array(similarity_indices).shape)#(30,12)
printt('similarity_indices',similarity_indices)

save=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\time_shape'
shapelets_path = os.path.join(save, dataset+'_timeshape.npy')
similarity_indices_array = np.array(similarity_indices)

similarity_indices_array[similarity_indices_array != 0] = 1

# 保存数组到文件
np.save(shapelets_path, similarity_indices_array)

print(f'Saved similarity indices to {shapelets_path}')