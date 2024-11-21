
import os
import numpy as np
def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

#"AtrialFibrillation", "FingerMovements", "PenDigits", 
    #"HandMovementDirection", "Heartbeat", "Libras",
   # "MotorImagery", "NATOPS", "SelfRegulationSCP2", "StandWalkJump"
dataset = "HandMovementDirection"
data_dir=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo'
X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))

X_train = X[train_idx].transpose(0, 2, 1)
y_train = y[train_idx]
X_test = X[test_idx].transpose(0, 2, 1)
y_test = y[test_idx]

X_train=X_train[0]

print('X_train',X_train.shape)

import matplotlib.pyplot as plt

# 假设 X_train 的形状为 (n, length)
n, length = X_train.shape



import matplotlib.pyplot as plt

# 定义颜色列表
colors = plt.cm.viridis(np.linspace(0, 1, n))  # 使用 Viridis 颜色图

# 可视化每个维度的时间序列
for i in range(n):
    plt.figure(figsize=(17, 3))  # 设置图形大小
    plt.plot(X_train[i], label=f'Dimension {i+1}', color=colors[i], linewidth=3)  # 绘制每个维度的时间序列，设置线条颜色和粗细
    
    # 取消所有图例、标签、网格线等
    plt.xticks([])  # 取消 x 轴刻度
    plt.yticks([])  # 取消 y 轴刻度
    plt.grid(False)  # 不显示网格线
    plt.box(False)  # 取消边框
    
    # 保存每一张图片
    plt.savefig(f'Dimension_{i+1}.png', bbox_inches='tight', transparent=True)  # 保存图像并紧凑布局
    plt.close()  # 关闭当前图形，释放内存
    input()
