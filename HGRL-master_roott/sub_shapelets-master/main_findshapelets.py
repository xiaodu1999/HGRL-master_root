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

root=r'/root'
data_dir=os.path.join(root,'HGRL-master_root','sub_shapelets-master','demo') #r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo'
X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))
label_path =os.path.join(data_dir, 'mtivariate_sublabel') #r"I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\mtivariate_sublabel"
label_path = os.path.join(label_path, dataset+'.txt')

#label_path =os.path.join() #r"I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\mtivariate_sublabel"
#label_path = os.path.join(label_path, dataset+'.txt')
labels_sub = np.loadtxt(label_path, delimiter=',')  # Load the labels as a NumPy array
print('labels_sub', labels_sub.shape)#(30,)

X_train = X[train_idx].transpose(0, 2, 1)
y_train = y[train_idx]
X_test = X[test_idx].transpose(0, 2, 1)
y_test = y[test_idx]


# normalize training data
X_train, scaler = normalize_data(X_train)
print('X_train, y_train', X_train.shape, y_train.shape)#
#=====
# Read the labels from the file


# Output the shape (dimensions) of the labels
print("Labels shape:", labels_sub.shape) #(30,)

# Split the labels into training and testing sets
y_train_sub = labels_sub[:len(y_train)]      # First n samples for training
y_test_sub = labels_sub[len(y_train):len(labels_sub)]  # Next m samples for testing
num_classes_sub = len(set(y_train_sub))
print('个体数量sub', num_classes_sub)

#======

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

def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=1000):#10000
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments)
    print('segments',segments.shape)
    segments=segments.transpose(0, 2, 1)
    print('segments',segments.shape)
    #input()
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters

n_ts, n_channels, len_ts = X_train.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train))
# learn 2 shapelets of length 130
#shapelets_size_and_len = {int(0.2 * len_ts): num_classes*1} #{130: 2}
shapelets_size_and_len = {int(i): num_classes*num_classes_sub for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.25 * len_ts), 4, dtype=int)}
print('shapelets_size_and_len', shapelets_size_and_len)#{64: 3, 96: 3, 128: 3, 160: 3}
first_key, first_value = next(iter(shapelets_size_and_len.items()))
shapelets_size_and_len= {first_key: first_value}
dist_measure = "euclidean"
lr = 1e-3
wd = 1e-3
epsilon = 1e-7

#input()
learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=True,#True
                                       verbose=1,
                                       dist_measure=dist_measure
                                       ,num_classes_sub=num_classes_sub
                                       )

for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):#
    print('shapelet训练')
    weights_block = get_weights_via_kmeans(X_train, shapelets_size, num_shapelets)
    learning_shapelets.set_shapelet_weights_of_block(i, weights_block)

    optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
    learning_shapelets.set_optimizer(optimizer)

    losses = learning_shapelets.fit(X_train, y_train, epochs=1000, batch_size=256, 
                                    shuffle=False, drop_last=False,Y_sub=y_train_sub)#2000

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
        ts = ts.cuda()
        shapelet = shapelet.cuda()
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
folder_path =os.path.join(root,'HGRL-master_root','sub_shapelets-master','shapelets') #"I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\shapelets"
file_name = dataset+"_shapelets.npy"
file_path = os.path.join(folder_path, file_name)

# 创建文件夹（如果不存在）
os.makedirs(folder_path, exist_ok=True)

# 将 shapelets 转换为 NumPy 数组（如果需要）
shapelets_array = np.array(shapelets)

# 保存为 .npy 文件
np.save(file_path, shapelets_array)

print(f"Shapelets saved to {file_path}")








