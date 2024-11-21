import os
from src.learning_shapelets import LearningShapelets
import math
import random
import numpy
from matplotlib import pyplot
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
import torch
from torch import nn, optim

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(numpy.product(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(numpy.product(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if scaler is None:
        X, scaler = normalize_standard(X)
    else:
        X, scaler = normalize_standard(X, scaler)
    
    return X, scaler

import numpy as np
def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

dataset = "AtrialFibrillation"
# Load training data
# X_train = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_train.npy'), 'rb'))
# print(f"Shape X_train: {X_train.shape}")
# # load trainng data labels
# y_train = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_train_labels.npy'), 'rb'))

data_dir=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo'
X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))

X_train = X[train_idx].transpose(0, 2, 1)
y_train = y[train_idx]
X_test = X[test_idx].transpose(0, 2, 1)
y_test = y[test_idx]

# normalize training data
X_train, scaler = normalize_data(X_train)
print('X_train, y_train', X_train.shape, y_train.shape)#


pyplot.title("Sample from the data")
pyplot.plot(X_train[0, 0], color='black')

def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = numpy.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments

def get_weights_via_kmeanso(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    print('segments',segments.shape)

    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters

def get_weights_via_kmeans(X, shapelets_size, num_shapelets,segments, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = segments#sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    print('segments',segments.shape)
    
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters

n_ts, n_channels, len_ts = X_train.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train))
# learn 2 shapelets of length 130
#shapelets_size_and_len = {int(0.2 * len_ts): num_classes*1} #{130: 2}
shapelets_size_and_len = {int(i): num_classes*1 for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.25 * len_ts), 4, dtype=int)}
print('shapelets_size_and_len', shapelets_size_and_len)
first_key, first_value = next(iter(shapelets_size_and_len.items()))
shapelets_size_and_len= {first_key: first_value}
dist_measure = "euclidean"
lr = 1e-3
wd = 1e-3
epsilon = 1e-7

# X_train, y_train (15, 2, 640) (15,)
# shapelets_size_and_len {64: 3, 96: 3, 128: 3, 160: 3}

learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=True,
                                       verbose=1,
                                       dist_measure=dist_measure)

from findshapelet1 import TimeSeriesAttentionModel, extract_subsequences
num_heads = 1
n=5

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):#
    print('shapelet训练')
    

    #===
    m = shapelets_size
    
    subsequences = extract_subsequences(X_train,m,n) 
    print('subsequences',subsequences.shape)#[15, 2, 116, 64])
    topk = int(subsequences.shape[1]*subsequences.shape[2]) // 6
    print('topk',topk)
    batch_size, num_channels, num_subsequences, sub_length= \
    subsequences.shape[0],subsequences.shape[1],subsequences.shape[2],subsequences.shape[3]
    tsd_model=sub_length
    #===
    # 模型定义
    model = TimeSeriesAttentionModel(tsd_model, num_heads, num_channels, num_subsequences, sub_length,num_classes,n,topk)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练过程
    epochs = 50
    for epoch in range(epochs):
        model.train()  # 设定为训练模式

        # 前向传播
        important_subsequences, triplet_loss = model(subsequences)  # 得到重要子序列和三元组损失
        important_subsequences = important_subsequences.view(-1, sub_length, num_channels)  # 调整形状

        # 反向传播
        optimizer.zero_grad()  # 清除梯度
        triplet_loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {triplet_loss.item()}")

    print("训练完成！")

    
    

   # input()

    #返回聚类的质心，即每个聚类的中心点（或代表时间序列）
    weights_block = get_weights_via_kmeans(X_train, shapelets_size,num_shapelets,important_subsequences)
    print('weights_block', weights_block.shape)
    #input()#(3, 2, 64)



    ###====学习shapelet
    learning_shapelets.set_shapelet_weights_of_block(i, weights_block)

    optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
    learning_shapelets.set_optimizer(optimizer)

    losses = learning_shapelets.fit(X_train, y_train, epochs=1000, batch_size=256, shuffle=False, drop_last=False)#2000

    pyplot.plot(losses, color='black')
    pyplot.title("Loss over training steps")
    pyplot.show()
    #input()

def eval_accuracy(model, X, Y):
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")

# # Load data set
# X_test = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_test.npy'), 'rb'))
# print(f"Shape X_train: {X_test.shape}")
# y_test = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_test_labels.npy'), 'rb'))

# normalize data
X_train, scaler = normalize_data(X_train, scaler)

def torch_dist_ts_shapelet(ts, shapelet, cuda=True):
    """
    Calculate euclidean distance of shapelet to a time series via PyTorch and returns the distance along with the position in the time series.
    """
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if cuda:
        ts = ts#.cuda()
        shapelet = shapelet#.cuda()
    shapelet = torch.unsqueeze(shapelet, 0)
    # unfold time series to emulate sliding window
    ts = ts.unfold(1, shapelet.shape[2], 1)
    # calculate euclidean distance
    dists = torch.cdist(ts, shapelet, p=2)
    dists = torch.sum(dists, dim=0)
    # otherwise gradient will be None
    # hard min compared to soft-min from the paper
    d_min, d_argmin = torch.min(dists, 0)

    return (d_min[0].item(), d_argmin[0].item())  # 选择第一个元素
    #return (d_min.item(), d_argmin.item())

def lead_pad_shapelet(shapelet, pos):
    """
    Adding leading NaN values to shapelet to plot it on a time series at the best matching position.
    """
    pad = numpy.empty(pos)
    pad[:] = numpy.NaN
    padded_shapelet = numpy.concatenate([pad, shapelet])
    return padded_shapelet

import matplotlib.pyplot as plt
shapelets = learning_shapelets.get_shapelets()
print('shapelets', shapelets.shape)#shapelets (2, 1, 130) (3, 2, 64)

# 目标文件夹和文件名
folder_path = "I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\shapelets"
file_name = dataset+"_shapelets.npy"
file_path = os.path.join(folder_path, file_name)

# 创建文件夹（如果不存在）
os.makedirs(folder_path, exist_ok=True)

# 将 shapelets 转换为 NumPy 数组（如果需要）
shapelets_array = np.array(shapelets)

# 保存为 .npy 文件
np.save(file_path, shapelets_array)

print(f"Shapelets saved to {file_path}")

input()
for i in range(shapelets.shape[0]):
    for j in range(shapelets.shape[1]):
        # 提取第三维度的第一个 shapelet
        first_shapelet = shapelets[i,j,  :]  # 选择第一个 shapelet

        # 可视化
        plt.figure(figsize=(10, 6))
        plt.plot(first_shapelet)
        plt.title('Visualization of the First Shapelet')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()


shapelet_transform = learning_shapelets.transform(X_test)
dist_s1 = shapelet_transform[:, 0]
dist_s2 = shapelet_transform[:, 1]
print('dist_s1, dist_s2',dist_s1.shape, dist_s2.shape)
weights, biases = learning_shapelets.get_weights_linear_layer()

#(3, 2, 64)
fig = pyplot.figure(facecolor='white')
fig.set_size_inches(20, 8)
gs = fig.add_gridspec(12, 8)
fig_ax1 = fig.add_subplot(gs[0:3, :4])
fig_ax1.set_title("First learned shapelet plotted (in red) on top of its 10 best matching time series.")
for i in numpy.argsort(dist_s1)[:1]:#10
    fig_ax1.plot(X_test[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[0])
    fig_ax1.plot(lead_pad_shapelet(shapelets[0, 0], pos), color='#F03613', alpha=0.5)


fig_ax2 = fig.add_subplot(gs[0:3, 4:])
fig_ax2.set_title("Second learned shapelet plotted (in red) on top of its 10 best matching time series.")
for i in numpy.argsort(dist_s2)[:1]:#1
    fig_ax2.plot(X_test[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[1])
    fig_ax2.plot(lead_pad_shapelet(shapelets[1, 0], pos), color='#F03613', alpha=0.5)

#===
# fig_ax3 = fig.add_subplot(gs[4:, :])
# fig_ax3.set_title("The decision boundaries learned by the model to separate the four classes.")
# color = {0: '#F03613', 1: '#7BD4CC', 2: '#00281F', 3: '#BEA42E'}
# fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_test])


# viridis = cm.get_cmap('viridis', 4)
# # Create a meshgrid of the decision boundaries
# xmin = numpy.min(shapelet_transform[:, 0]) - 0.1
# xmax = numpy.max(shapelet_transform[:, 0]) + 0.1
# ymin = numpy.min(shapelet_transform[:, 1]) - 0.1
# ymax = numpy.max(shapelet_transform[:, 1]) + 0.1
# xx, yy = numpy.meshgrid(numpy.arange(xmin, xmax, (xmax - xmin)/200),
#                         numpy.arange(ymin, ymax, (ymax - ymin)/200))
# Z = []
# for x, y in numpy.c_[xx.ravel(), yy.ravel()]:
#     Z.append(numpy.argmax([biases[i] + weights[i][0]*x + weights[i][1]*y
#                            for i in range(3)]))#4
# Z = numpy.array(Z).reshape(xx.shape)
# fig_ax3.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)
# fig_ax3.set_xlabel("$dist(x, s_1)$", fontsize=14)
# fig_ax3.set_ylabel("$dist(x, s_2)$", fontsize=14)

#===
caption = """Shapelets learned for the FaceFour dataset of the UCR archive plotted on top of the best matching time series (top two pictures).
        And the corresponding learned decision boundaries of the linear classifier on top of the shapelet transformed test data (bottom picture)."""
pyplot.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=14)
pyplot.savefig('learning_shapelets.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
pyplot.show()







