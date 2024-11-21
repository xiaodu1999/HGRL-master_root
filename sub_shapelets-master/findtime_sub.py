import os

import numpy as np
def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

dataset='AtrialFibrillation'

data_dir=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo'
X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))
len_sam=len(y)#数据集样本数量，30
print('样本数量',len_sam)
#====
label_path = r"I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\mtivariate_sublabel"
label_path = os.path.join(label_path, dataset+'.txt')
labels_sub = np.loadtxt(label_path, delimiter=',')  # Load the labels as a NumPy array
print('labels_sub', labels_sub.shape)#(30,)

# 获取唯一的标签集合
unique_labels, label_indices = np.unique(labels_sub, return_inverse=True)
print('unique_labels',unique_labels)
num_labels = len(unique_labels)  # 计算个体类型数量

print('计算个体类型数量',num_labels)
#====
# 初始化 30 x n 的对应关系矩阵
correspondence_matrix = np.zeros((len_sam, num_labels), dtype=int)

# 填充矩阵
# 填充矩阵
for i, label_index in enumerate(label_indices):
    correspondence_matrix[i, label_index] = 1

# 打印结果
print('Correspondence matrix :')
print(correspondence_matrix.shape)


#=====
save=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\time_sub'
sub_path = os.path.join(save, dataset+'_timesub.npy')
similarity_indices_array = np.array(correspondence_matrix)


# 保存数组到文件
np.save(sub_path, similarity_indices_array)

print(f'Saved similarity indices to {sub_path}')