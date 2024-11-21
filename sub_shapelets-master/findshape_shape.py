

import os

import numpy as np
import torch
from soft_dtw import SoftDTW #no cuda
#from soft_dtw_cuda import SoftDTW
def normalize_dtw_matrix(dtw_matrix, method='normalize'):
    """
    对 DTW 相似矩阵进行归一化或标准化。
    
    参数：
    dtw_matrix: 输入的 DTW 相似矩阵
    method: 'normalize' 进行归一化，'standardize' 进行标准化
    
    返回：
    处理后的矩阵
    """
    if method == 'normalize':
        # 归一化到 [0, 1]
        min_val = np.min(dtw_matrix)
        max_val = np.max(dtw_matrix)
        normalized_matrix = (dtw_matrix - min_val) / (max_val - min_val)
        return normalized_matrix
    
    elif method == 'standardize':
        # 标准化到均值为 0，标准差为 1
        mean = np.mean(dtw_matrix)
        std = np.std(dtw_matrix)
        standardized_matrix = (dtw_matrix - mean) / std
        return standardized_matrix
    
    else:
        raise ValueError("Method must be 'normalize' or 'standardize'.")
data=r'H:\HGRL-master\sub_shapelets-master\shapelets'
dataset='AtrialFibrillation'

shapelets = os.path.join(data, dataset+'_shapelets.npy')
# 加载 .npy 文件
shapelets = np.load(shapelets)
print('shapelets', shapelets.shape)#(30, 12)


shapelets=shapelets.reshape(-1, shapelets.shape[2])
shapelets= torch.tensor(shapelets, dtype=torch.float32)
print('shapelets', shapelets.shape)

# 计算序列数量
num_shapelets = shapelets.shape[0]

# 创建一个对称矩阵，初始化为零
distance_matrix = torch.zeros((num_shapelets, num_shapelets))
dtw_distance_matrix = torch.zeros((num_shapelets, num_shapelets))
soft_dtw = SoftDTW(gamma=1.0, normalize=False)
# 计算欧氏距离，只计算上三角部分
for i in range(num_shapelets):
    for j in range(i + 1, num_shapelets):
        # # 计算欧氏距离
        # distance = np.linalg.norm(shapelets[i] - shapelets[j])
        # distance_matrix[i, j] = distance
        # distance_matrix[j, i] = distance  # 对称填充

        # 计算 DTW 距离
       
        shapelet_i = shapelets[i].unsqueeze(0)  # (1, 64)
        shapelet_j = shapelets[j].unsqueeze(0)  # (1, 64)
        #print('shapelet_i',shapelet_i.shape)
        
        dtw_distance = soft_dtw(shapelet_i, shapelet_j)
        dtw_distance_matrix[i, j] = dtw_distance
        dtw_distance_matrix[j, i] = dtw_distance  # 对称填充
print('dtw_distance_matrix',dtw_distance_matrix.shape)
#dtw_distance_matrix=normalize_dtw_matrix(dtw_distance_matrix, method='normalize')

#=====
# 归一化
min_distance = dtw_distance_matrix.min()
max_distance = dtw_distance_matrix.max()

if max_distance - min_distance > 0:  # 避免除以零
    normalized_dtw_distance_matrix = (dtw_distance_matrix - min_distance) / (max_distance - min_distance)
else:
    normalized_dtw_distance_matrix = dtw_distance_matrix  # 如果所有值相同，则不进行标准化

# 将 DTW 距离转换为相似度值
alpha = 0.3  # 调整 alpha 参数
similarity_matrix = np.zeros_like(normalized_dtw_distance_matrix)

for i in range(num_shapelets):
    for j in range(num_shapelets):
        similarity_matrix[i, j] = 1 / np.exp(alpha * normalized_dtw_distance_matrix[i, j])

# 归一化处理
# similarity_matrix /= similarity_matrix.sum(axis=1, keepdims=True)


#===保存
save=r'H:\HGRL-master\sub_shapelets-master\shape_shape'

subshape_path = os.path.join(save, dataset+'_shapeshape.npy')

np.save(subshape_path, similarity_matrix)

