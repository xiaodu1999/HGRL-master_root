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


dataset = "FaceFour"
# Load training data
X_train = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_train.npy'), 'rb'))
print(f"Shape X_train: {X_train.shape}")
# load trainng data labels
y_train = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_train_labels.npy'), 'rb'))
# normalize training data
X_train, scaler = normalize_data(X_train)
print('X_train, y_train', X_train.shape, y_train.shape)#

input()
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

def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    print('segments',segments.shape)
    input()
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters

n_ts, n_channels, len_ts = X_train.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train))
# learn 2 shapelets of length 130
shapelets_size_and_len = {130: 2}
dist_measure = "euclidean"
lr = 1e-2
wd = 1e-3
epsilon = 1e-7


learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=True,
                                       verbose=1,
                                       dist_measure=dist_measure)

for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):#
    print('shapelet训练')
    weights_block = get_weights_via_kmeans(X_train, shapelets_size, num_shapelets)
    learning_shapelets.set_shapelet_weights_of_block(i, weights_block)

    optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
    learning_shapelets.set_optimizer(optimizer)

    losses = learning_shapelets.fit(X_train, y_train, epochs=50, batch_size=256, shuffle=False, drop_last=False)#2000

    pyplot.plot(losses, color='black')
    pyplot.title("Loss over training steps")
    pyplot.show()
    input()

def eval_accuracy(model, X, Y):
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")

# Load data set
X_test = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_test.npy'), 'rb'))
print(f"Shape X_train: {X_test.shape}")
y_test = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_test_labels.npy'), 'rb'))
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
    return (d_min.item(), d_argmin.item())

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
print('shapelets', shapelets.shape)#shapelets (2, 1, 130)

# 提取第三维度的第一个 shapelet
first_shapelet = shapelets[0, 0, :]  # 选择第一个 shapelet

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

weights, biases = learning_shapelets.get_weights_linear_layer()

fig = pyplot.figure(facecolor='white')
fig.set_size_inches(20, 8)
gs = fig.add_gridspec(12, 8)
fig_ax1 = fig.add_subplot(gs[0:3, :4])
fig_ax1.set_title("First learned shapelet plotted (in red) on top of its 10 best matching time series.")
for i in numpy.argsort(dist_s1)[:10]:
    fig_ax1.plot(X_test[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[0])
    fig_ax1.plot(lead_pad_shapelet(shapelets[0, 0], pos), color='#F03613', alpha=0.5)

fig_ax2 = fig.add_subplot(gs[0:3, 4:])
fig_ax2.set_title("Second learned shapelet plotted (in red) on top of its 10 best matching time series.")
for i in numpy.argsort(dist_s2)[:10]:
    fig_ax2.plot(X_test[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[1])
    fig_ax2.plot(lead_pad_shapelet(shapelets[1, 0], pos), color='#F03613', alpha=0.5)


fig_ax3 = fig.add_subplot(gs[4:, :])
fig_ax3.set_title("The decision boundaries learned by the model to separate the four classes.")
color = {0: '#F03613', 1: '#7BD4CC', 2: '#00281F', 3: '#BEA42E'}
fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_test])


viridis = cm.get_cmap('viridis', 4)
# Create a meshgrid of the decision boundaries
xmin = numpy.min(shapelet_transform[:, 0]) - 0.1
xmax = numpy.max(shapelet_transform[:, 0]) + 0.1
ymin = numpy.min(shapelet_transform[:, 1]) - 0.1
ymax = numpy.max(shapelet_transform[:, 1]) + 0.1
xx, yy = numpy.meshgrid(numpy.arange(xmin, xmax, (xmax - xmin)/200),
                        numpy.arange(ymin, ymax, (ymax - ymin)/200))
Z = []
for x, y in numpy.c_[xx.ravel(), yy.ravel()]:
    Z.append(numpy.argmax([biases[i] + weights[i][0]*x + weights[i][1]*y
                           for i in range(4)]))
Z = numpy.array(Z).reshape(xx.shape)
fig_ax3.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)
fig_ax3.set_xlabel("$dist(x, s_1)$", fontsize=14)
fig_ax3.set_ylabel("$dist(x, s_2)$", fontsize=14)

caption = """Shapelets learned for the FaceFour dataset of the UCR archive plotted on top of the best matching time series (top two pictures).
        And the corresponding learned decision boundaries of the linear classifier on top of the shapelet transformed test data (bottom picture)."""
pyplot.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=14)
pyplot.savefig('learning_shapelets.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
pyplot.show()







