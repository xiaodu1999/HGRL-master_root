import os

import numpy as np


data=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\time_shape'
dataset='AtrialFibrillation'

shapetime = os.path.join(data, dataset+'_timeshape.npy')
# 加载 .npy 文件
shapetime = np.load(shapetime)
print('shapetime', shapetime.shape)#(30, 12)
#====
label_path = r"I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\mtivariate_sublabel"
label_path = os.path.join(label_path, dataset+'.txt')
labels_sub = np.loadtxt(label_path, delimiter=',')  # Load the labels as a NumPy array
print('labels_sub', labels_sub.shape)#(30,)
#===

# 获取个体类型
sub_types = np.unique(labels_sub)  # 获取所有独特的个体类型
num_sub = len(sub_types)  # 个体类型的数量

# 初始化对应关系矩阵
correspondence_matrix = np.zeros((num_sub, 12), dtype=int)

# 填充对应关系矩阵
for i in range(shapetime.shape[0]):  # 遍历每个时间序列
    sub_idx = np.where(sub_types == labels_sub[i])[0][0]  # 找到个体类型的索引
    correspondence_matrix[sub_idx] += shapetime[i]  # 将对应的 shapelet 加入到该个体类型的行中

# 将非零位置的值改为 1
correspondence_matrix[correspondence_matrix > 0] = 1

# 打印结果
print('Correspondence matrix:\n', correspondence_matrix)

save=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\sub_shape'

subshape_path = os.path.join(save, dataset+'_subshape.npy')

np.save(subshape_path, correspondence_matrix)