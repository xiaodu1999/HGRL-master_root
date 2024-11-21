import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']
    #return data['X'], data['y'], data['train_idx'], data['test_idx']


def read_dataset(ucr_root_dir, dataset_name, shot):
    """ Read univariate dataset from UCR
    """
    dataset_dir = os.path.join(ucr_root_dir, dataset_name)
    df_train = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TRAIN.tsv'), sep='\t', header=None)
    df_test = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TEST.tsv'), sep='\t', header=None)

    y_train = df_train.values[:, 0].astype(np.int64)
    y_test = df_test.values[:, 0].astype(np.int64)
    y = np.concatenate((y_train, y_test))
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    X_train = df_train.drop(columns=[0]).astype(np.float32)
    X_test = df_test.drop(columns=[0]).astype(np.float32)

    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])

    X_train = X_train.values
    X_test = X_test.values
    X = np.concatenate((X_train, X_test))
    idx = np.array([i for i in range(len(X))])

    np.random.shuffle(idx)
    train_idx, test_idx = idx[:int(len(idx)*0.8)], idx[int(len(idx)*0.8):]

    tmp = [[] for _ in range(len(np.unique(y)))]
    for i in train_idx:
        tmp[y[i]].append(i)
    train_idx = []
    for _tmp in tmp:
        train_idx.extend(_tmp[:shot])

    # znorm
    X[np.isnan(X)] = 0
    std_ = X.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    X = (X - X.mean(axis=1, keepdims=True)) / std_

    # add a dimension to make it multivariate with one dimension 
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y, train_idx, test_idx

from readuea import load_UEA, load_UEAsuper
def read_multivariate_dataseto(root_dir, dataset_name, shot):
    """ Read multivariate dataset
    """
    # X = np.load(os.path.join(root_dir, dataset_name+".npy"), allow_pickle=True)
    # y = np.loadtxt(os.path.join(root_dir, dataset_name+'_label.txt'))
    # y = y.astype(np.int64)
    # print('x', X.shape)
    # print('y', y.shape)

    X,y=load_UEA(dataset_name)
    print('x', X.shape)
    print('y', y.shape)

    #input()
    dim = X[0].shape[0]
    max_length = 0
    for _X in X:
        if _X.shape[1] > max_length:
            max_length = _X.shape[1]

    X_list = []
    for i in range(len(X)):
        _X = np.zeros((dim, max_length))
        _X[:, :X[i].shape[1]] = X[i]
        X_list.append(_X)
    X = np.array(X_list, dtype=np.float32)

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    print('y', len(y))#2858
    idx = np.array([i for i in range(len(X))])

    np.random.shuffle(idx)
    print('len idx',len(idx))#2858
    train_idx, test_idx = idx[:int(len(idx)*0.8)], idx[int(len(idx)*0.8):]
    print('train_idx, test_idx',len(train_idx), len(test_idx))# 2286 572
    #input()

    #根据每个类别选择一定数量的训练样本，并确保不会超出每个类别的样本数量。
    # 这样可以避免潜在的索引问题，并确保训练集的构建是合理的
    tmp = [[] for _ in range(len(np.unique(y)))]
    print('tmp1', len(tmp))#20
    for i in train_idx:
        tmp[y[i]].append(i)
    print('tmp2', len(tmp))#20
    train_idx = []
    for _tmp in tmp:
        train_idx.extend(_tmp[:shot])#shot=1, 每类多少个标记样本

    
    # znorm
    # std_ = X.std(axis=2, keepdims=True)
    # std_[std_ == 0] = 1.0
    # X = (X - X.mean(axis=2, keepdims=True)) / std_
    print('train_idx, test_idx',len(train_idx), len(test_idx))#  20 572
   # input()

    return X, y, train_idx, test_idx
def read_multivariate_dataset_flexi(root_dir, dataset_name, shot):
    """ Read multivariate dataset
    """
    # 加载数据
    X, y = load_UEA(dataset_name)
    print('x', X.shape)
    print('y', y.shape)

    # 打印 y 标签的分布
    unique_labels, counts = np.unique(y, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label}: {count} 个样本")
    #input()

    # 获取维度和最大长度
    dim = X[0].shape[0]
    max_length = 0
    for _X in X:
        if _X.shape[1] > max_length:
            max_length = _X.shape[1]

    # 统一每个样本的长度
    X_list = []
    for i in range(len(X)):
        _X = np.zeros((dim, max_length))
        _X[:, :X[i].shape[1]] = X[i]
        X_list.append(_X)
    X = np.array(X_list, dtype=np.float32)

    # 标签编码
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    print('y', len(y))  # 2858

    # 通过类别划分索引并按比例分配训练集和测试集
    
    category_idx = defaultdict(list)
    for i, label in enumerate(y):
        category_idx[label].append(i)

    train_idx = []
    test_idx = []

    for label, idx_list in category_idx.items():
        np.random.shuffle(idx_list)  # 随机打乱每个类别的样本索引
        train_size = int(len(idx_list) * 0.8)  # 80% 用于训练
        train_idx.extend(idx_list[:train_size])
        test_idx.extend(idx_list[train_size:])

    # 打乱训练集和测试集的索引顺序
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    print('train_idx, test_idx:', len(train_idx), len(test_idx))  # 2286 572
    #====

    # 打印训练集和测试集的类别分布
    train_labels = y[train_idx]
    test_labels = y[test_idx]

    unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
    unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)

    print("训练集类别分布:")
    for label, count in zip(unique_train_labels, train_counts):
        print(f"标签 {label}: {count} 个样本")

    print("测试集类别分布:")
    for label, count in zip(unique_test_labels, test_counts):
        print(f"标签 {label}: {count} 个样本")
    #=====
    #input()

    # 根据每个类别选择一定数量的训练样本，确保不会超出每个类别的样本数量
    tmp = [[] for _ in range(len(np.unique(y)))]
    for i in train_idx:
        tmp[y[i]].append(i)

    train_idx = []
    for _tmp in tmp:
        train_idx.extend(_tmp[:shot])  # 每类选择 `shot` 个样本

    # Z标准化
    # std_ = X.std(axis=2, keepdims=True)
    # std_[std_ == 0] = 1.0
    # X = (X - X.mean(axis=2, keepdims=True)) / std_

    print('train_idx, test_idx:', len(train_idx), len(test_idx))  # 20 572

    return X, y, train_idx, test_idx
def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']
from collections import defaultdict
def read_multivariate_dataset(root_dir, dataset_name, shot):
    """ Read multivariate dataset
    """
    # 加载数据
    #===========#原始数据集
    # X, y = load_UEA(dataset_name)

    #==============#原始数据转的npy文件
    dataset="AtrialFibrillation"
    # Load training data
    X_all, y_all, train_idx, test_idxx = read_dataset_from_npy(os.path.join(root_dir, 'multivariate_datasets_' + 'supervise', dataset_name+'.npy'))
    
    #===训练和验证从整个数据集里挑选
    #X = X.transpose(0, 2, 1)

    #===训练和验证从训练集里挑选
    X_train =X_all[train_idx].transpose(0, 2, 1)
    X=X_train
    y_train = y_all[train_idx]
    y=y_train

    X_test = X_all[test_idxx].transpose(0, 2, 1)
    y_test = y_all[test_idxx]
    #=============

    # 打印 y 标签的分布
    unique_labels, counts = np.unique(y, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label}: {count} 个样本")
    #input()

    # 获取维度和最大长度
    dim = X[0].shape[0]
    max_length = 0
    for _X in X:
        if _X.shape[1] > max_length:
            max_length = _X.shape[1]

    # 统一每个样本的长度
    X_list = []
    for i in range(len(X)):
        _X = np.zeros((dim, max_length))
        _X[:, :X[i].shape[1]] = X[i]
        X_list.append(_X)
    X = np.array(X_list, dtype=np.float32)

    # 标签编码
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    print('y', len(y))  # 2858

    # 通过类别划分索引并按比例分配训练集和测试集
    
    category_idx = defaultdict(list)
    for i, label in enumerate(y):
        category_idx[label].append(i)

    train_idx = []
    test_idx = []

    for label, idx_list in category_idx.items():
        np.random.shuffle(idx_list)  # 随机打乱每个类别的样本索引
        train_size = int(len(idx_list) * 0.8)  # 80% 用于训练
        train_idx.extend(idx_list[:train_size])
        test_idx.extend(idx_list[train_size:])

    # 打乱训练集和测试集的索引顺序
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    print('train_idx, vali_idx:', len(train_idx), len(test_idx))  # 2286 572
    #====

    # 打印训练集和测试集的类别分布
    train_labels = y[train_idx]
    test_labels = y[test_idx]

    unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
    unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)

    print("训练集类别分布:")
    for label, count in zip(unique_train_labels, train_counts):
        print(f"标签 {label}: {count} 个样本")

    print("测试集类别分布:")
    for label, count in zip(unique_test_labels, test_counts):
        print(f"标签 {label}: {count} 个样本")
    #=====
    #input()

    # 根据每个类别选择一定数量的训练样本，确保不会超出每个类别的样本数量
    tmp = [[] for _ in range(len(np.unique(y)))]
    for i in train_idx:
        tmp[y[i]].append(i)

    train_idx_final = []
    for _tmp in tmp:
        train_idx_final.extend(_tmp[:shot])  # 每类选择 `shot` 个样本

    # Z标准化
    # std_ = X.std(axis=2, keepdims=True)
    # std_[std_ == 0] = 1.0
    # X = (X - X.mean(axis=2, keepdims=True)) / std_

    print('train_idx,训练集里带标签的, test_idx数量:', len(train_idx), len(train_idx_final), len(test_idx))  # 20 572

    train_idx_final=train_idx_final#有标签训练

    train_idx = set(train_idx)        # 转换为集合以便于求差集
    train_idx_final = set(train_idx_final)  # 转换为集合

    # 计算 train_idx 与 train_idx_final 的差集
    diff_set = train_idx - train_idx_final  # 得到属于 train_idx 但不在 train_idx_final 中的索引

    # 将差集并入 test_idx
    vali_idx = list(set(test_idx).union(diff_set))#验证

    test_idxx=test_idxx #测试
    
    return X, y_all, list(train_idx_final), vali_idx,X_test, test_idxx

def read_multivariate_datasetsuper(root_dir, dataset_name, shot):
    """ Read multivariate dataset
    """
    # 加载数据
    train_x,train_y,test_x, test_y,num_class= load_UEAsuper(dataset_name)
    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    print('x', x.shape)
    print('y', y.shape)

    
    # 生成训练和测试索引
    train_idx = np.arange(len(train_y))
    test_idx = np.arange(len(train_y), len(train_y) + len(test_y))

    print('x', x.shape)
    print('y', y.shape)
    # print('train_idx:', train_idx)
    # print('test_idx:', test_idx)

    return x, y, train_idx, test_idx

def read_X(ucr_root_dir, dataset_name):
    """ Read the raw time-series
    """
    dataset_dir = os.path.join(ucr_root_dir, dataset_name)
    df_train = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TRAIN.tsv'), sep='\t', header=None)
    df_test = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TEST.tsv'), sep='\t', header=None)

    X_train = df_train.drop(columns=[0]).astype(np.float32)
    X_test = df_test.drop(columns=[0]).astype(np.float32)

    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])

    X_train = X_train.values
    X_test = X_test.values
    X = np.concatenate((X_train, X_test), axis=0)

    return X

class Logger:
    def __init__(self, f):
        self.f = f

    def log(self, content):
        print(content)
        self.f.write(content + '\n')
        self.f.flush()


