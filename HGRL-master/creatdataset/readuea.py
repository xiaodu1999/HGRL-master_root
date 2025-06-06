import math
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.io.arff import loadarff
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    # swapaxes的用法就是交换轴的位置，前后两个的位置没有关系。


def load_UEA(archive_name):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    data_path='/root/MF-Net-1202/data/Multivariate_arff'
    a=2
    # load from cache
    #cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
    if a==1:#os.path.exists(cache_path) is True:
        print('load form cache....')
        #train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)

    
    # load from arff
    else:
        print('reading dataset')
        train_data = \
            loadarff(open(f'{data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)  ##y为标签
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        # scaler = StandardScaler()
        # scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        # train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        # test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)  ##标签
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        #torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)
    # # return TrainDataset,TestDataset,len(labels)
    # # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # # batchszie：批大小，决定一个epoch有多少个Iteration；
    # train_loader = DataLoader(dataset=TrainDataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=True)
    # test_loader = DataLoader(dataset=TestDataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    # print("Train labels:", train_y)
    # print("Test labels:", test_y)
    #input()
    return x , y
    #return train_loader, test_loader, num_class
def load_UEAsuper(archive_name):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    data_path='/root/MF-Net-1202/data/Multivariate_arff'
    a=2
    # load from cache
    #cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
    if a==1:#os.path.exists(cache_path) is True:
        print('load form cache....')
        #train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)

    
    # load from arff
    else:
        print('reading dataset')
        train_data = \
            loadarff(open(f'{data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)  ##y为标签
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        # scaler = StandardScaler()
        # scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        # train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        # test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)  ##标签
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        #torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)
    # # return TrainDataset,TestDataset,len(labels)
    # # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # # batchszie：批大小，决定一个epoch有多少个Iteration；
    # train_loader = DataLoader(dataset=TrainDataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=True)
    # test_loader = DataLoader(dataset=TestDataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    print("Train labels:", train_y)
    print("Test labels:", test_y)
    #input()
    return train_x,train_y,test_x, test_y,num_class
    #return train_loader, test_loader, num_class
class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, y):
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]
        # self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))