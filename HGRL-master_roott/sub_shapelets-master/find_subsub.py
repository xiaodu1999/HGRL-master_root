

import os
import numpy as np

dataset='AtrialFibrillation'
def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

label_path=r'H:\HGRL-master\sub_shapelets-master\demo\mtivariate_sublabel'
label_path = os.path.join(label_path, dataset+'.txt')
labels_sub = np.loadtxt(label_path, delimiter=',')  # Load the labels as a NumPy array
print('labels_sub', labels_sub.shape)#(30,)

y=np.unique(labels_sub)
len_y=len(y)

# 创建一个全为零的方阵
zero_matrix = np.zeros((len_y, len_y))
np.fill_diagonal(zero_matrix, 1)

# 打印结果
print("Zero matrix:")
print(zero_matrix.shape)

#=====
save=r'H:\HGRL-master\sub_shapelets-master\sub_sub'
sub_path = os.path.join(save, dataset+'_subsub.npy')
similarity_indices_array = np.array(zero_matrix)


# 保存数组到文件
np.save(sub_path, zero_matrix)

print(f'Saved similarity indices to {sub_path}')