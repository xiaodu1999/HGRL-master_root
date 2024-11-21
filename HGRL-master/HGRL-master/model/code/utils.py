import numpy as np
import scipy.sparse as sp
from random import shuffle
import torch
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from printt import printt
from print import print
from sklearn.preprocessing import StandardScaler

def read_dataset_from_npyy(path):
        """ Read dataset from .npy file
        """
        data = np.load(path, allow_pickle=True)
        #print('data',data.shape)#(2858,)
        return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

def read_dataset_from_npy(path):
        """ Read dataset from .npy file
        """
        data = np.load(path, allow_pickle=True)
        #print('data',data.shape)#(2858,)
        return data
def read_patient_data_from_txt(file_path):
    patients = []
    with open(file_path, 'r') as file:
        for line in file:

            patients.append(line.strip())  
    return patients

def load_data_time(path=r'H:\nodetext\HGAT-master\multivariate_datasets', dataset="AtrialFibrillation"):
    print('Loading {} dataset...'.format(dataset))

    root=r'/root'
    path_data=os.path.join(root,'HGRL-master_root','sub_shapelets-master','demo')#r'\root\HGRL-master_root\sub_shapelets-master\demo'
    X, y, train_idx, test_idx = read_dataset_from_npyy(os.path.join(path_data, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))
    
    X=torch.load(os.path.join(path, 'representation','best_rep_'+dataset+'.pt'))
    printt('表示特征X',X.shape)


    # scaler = StandardScaler()

    # # 将 X 转换成二维矩阵以计算标准化参数
    # scaler.fit(X.reshape(-1, X.shape[-1]))

    # X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    #====
    # Load training data
    #X, y, train_idx, test_idx = read_dataset_from_npyy(os.path.join(path, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))
    
    #====
    # print('y', y)
    # X_train = X[train_idx]#.transpose(0, 2, 1)
    # y_train = y[train_idx]
    # X_test = X[test_idx]#.transpose(0, 2, 1)
    # y_test = y[test_idx]
    # print('X_train, y_train', X_train.shape, y_train.shape)#

    # 初始化标准化器
    # scaler = StandardScaler()

    # # 将 train_x 转换成二维矩阵以计算标准化参数
    # scaler.fit(train_x.reshape(-1, train_x.shape[-1]))

    # # 对训练集进行标准化
    # train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)

    # # 对测试集进行相同的标准化
    # test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

    # X=X.transpose(0, 2, 1)
    # X=X.reshape(X.shape[0],-1)
    # print('X', X.shape)

    #===== shapelet特征
    # matrix=read_dataset_from_npy(os.path.join(path, 'multivariate_datasets_' + 'dtw', dataset+'.npy'))
    # print('matrix', matrix.shape)
    shapelets=read_dataset_from_npy(os.path.join(path, 'shapelets' , dataset+'_shapelets.npy'))
    shapelets=shapelets.reshape(-1, shapelets.shape[2])

    # scaler1 = StandardScaler()
    # scaler1.fit(shapelets.reshape(-1, shapelets.shape[-1]))

    # shapelets= scaler1.transform(shapelets.reshape(-1, shapelets.shape[-1])).reshape(shapelets.shape)
    print('shapelets', shapelets.shape)
    
    
#====== 个体特征
    subject_feature=read_dataset_from_npy(os.path.join(path, 'subject_feature', dataset+'_subfeature.npy'))

    # scaler2 = StandardScaler()
    # scaler2.fit(subject_feature.reshape(-1, subject_feature.shape[-1]))

    # subject_feature= scaler2.transform(subject_feature.reshape(-1, 
    #                                                           subject_feature.shape[-1])).reshape(subject_feature.shape)
    print('subject_feature', subject_feature.shape)
    
#======================三种特征合并
   # input()
    All=[X, np.array(subject_feature), shapelets]

#=============

    features_block = True  # 是否连接特征空间的标志 False
    MULTI_LABEL = 'multi' in dataset  # 判断数据集是否为多标记类型
    
    type_list = ['time', 'patient', 'shapelet']  # 定义节点的类型
    type_have_label = 'time'  # 指定特征节点类型

    features_list = []  # 存储特征矩阵的列表
    idx_map_list = []  # 存储索引映射的列表
    idx2type = {t: set() for t in type_list}  # 创建类型到索引的映射

    for i, (type_name, all) in enumerate(zip(type_list, All)):
    #for type_name in type_list:

        print('Loading {} content...'.format(type_name))

        indexes, features, labels = [], [], []  # 初始化索引、特征和标签
        # 读取特定类型的内容文件
        #with open("{}{}.content.{}".format(path, dataset, type_name)) as f:
        #for line in tqdm(X):  # 遍历文件中的每一行
            #cache = line.strip().split('\t')  # 使用制表符分隔每一行
        indexes=list(range(All[i].shape[0]))  # 存储节点索引
        features=All[i]  # 存储特征
        features=features.reshape(features.shape[0], -1)
        if type_name == type_have_label:
            labels=y
            labels = labels.reshape(len(labels),-1)  # 存储标签
            print('labels',np.array(labels).shape)#(30,)
        else:
            labels=[1] * All[i].shape[0]
        features = np.stack(features)  # 将特征列表堆叠为矩阵
        #features = normalize(features)  # 归一化特征矩阵
        print('features1', features.shape)#(30, 640, 2) (30, 1280) (30, 1)

        # 根据特征连接标志转换特征为稀疏格式
        if not features_block:
            features = torch.FloatTensor(np.array(features))
            features = dense_tensor_to_sparse(features)
            print('features 稀疏', features)
        features_list.append(features)  # 将特征矩阵添加到列表

#==========
       # input()
        # 处理如果该类型具有标签
        if type_name == type_have_label:
            labels = np.stack(labels)  # 堆叠标签
            if not MULTI_LABEL:
                labels = encode_onehot(labels)  # 单标签编码
                #printt('标签：',labels)
                #input()
            Labels = torch.LongTensor(labels)  # 将标签转换为 Long Tensor
            print("label matrix shape: {}".format(Labels.shape))

        idx = np.stack(indexes)  # 堆叠索引
        for i in idx:
            idx2type[type_name].add(i)  # 更新类型到索引的映射
        idx_map = {j: i for i, j in enumerate(idx)}  # 创建索引映射
        idx_map_list.append(idx_map)  # 保存至列表 
        
        print('done.')


    # 计算各类型索引的长度
    len_list = [len(idx2type[t]) for t in type_list]
    type2len = {t: len(idx2type[t]) for t in type_list}  # 构建类型到长度的映射
    len_all = sum(len_list)  # 所有类型的总节点数
    printt('len_all', len_all) #63

#=================
    print('Building graph...')
    adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
    
    # 构建图的邻接矩阵
    #edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    
    adj_all=read_dataset_from_npy(os.path.join(path, 'finalmatrix' , dataset+'.npy'))
    #print(adj_all)
    #adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)


#===========
    

#======
   # 对称化邻接矩阵并添加自环
    print('adj_all1',adj_all.shape)#(39, 39)
    print("Num of edges1: {}".format(len(adj_all.nonzero()[0])))
    if isinstance(adj_all, np.ndarray):  # 如果是 NumPy 数组，转换为稀疏矩阵
                adj_all = sp.csr_matrix(adj_all)
    print('adj_all2',adj_all.shape)

    #adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    #adj_all = normalize_adj(adj_all )#+ sp.eye(adj_all.shape[0])

    # adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    # adj_all = adj_all + adj_all.T * (adj_all.T > adj_all) - adj_all * (adj_all.T > adj_all)
    # adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))
#====
    
    # 将稀疏邻接矩阵转换为PyTorch格式
    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            # 确保 submatrix 是一个稀疏矩阵
            
            adj_list[i1][i2] = sparse_mx_to_torch_sparse_tensor(
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                        sum(len_list[:i2]): sum(len_list[:i2 + 1])]
            )
    #print('adj_listfinal',adj_list)
    print("Num of edges: {}".format(len(adj_all.nonzero()[0])))  # 输出边的数量
   # input()
    # 加载训练、验证和测试集的索引
    
    idx_train, idx_val, idx_test = load_divide_idx(path,dataset, idx_map_list[0])

# 返回邻接列表、特征列表、标签和索引
    print('adj_list',adj_list)
    print('features_list',features_list)
    print('Labels',Labels)
    #input()


    for i in range(len(features_list)):
        features_list[i] = torch.from_numpy(features_list[i]).float()

    return adj_list, features_list, Labels, idx_train, idx_val, idx_test, idx_map_list[0]



def multi_label(labels):
    def myfunction(x):
        return list(map(int, x[0].split()))
    return np.apply_along_axis(myfunction, axis=1, arr=labels)


def encode_onehot(labels):
    classes = set(labels.T[0])
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels.T[0])),
                             dtype=np.int32)
    return labels_onehot

def load_divide_idx(path, dataset, idx_map):
    idx_train = []
    idx_val = []
    idx_test = []
    path=os.path.join(path,'map',dataset)
    with open(os.path.join(path, 'train.map'), 'r') as f:
        for line in f:
            idx_train.append( idx_map.get(int(line.strip('\n'))) )
    with open(os.path.join(path, 'val.map'), 'r') as f:
        for line in f:
            idx_val.append( idx_map.get(int(line.strip('\n'))) )
    with open(os.path.join(path, 'test.map'), 'r') as f:
        for line in f:
            idx_test.append( idx_map.get(int(line.strip('\n'))) )

    print("train, vali, test: ", len(idx_train), len(idx_val), len(idx_test))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test


def resample(train, val, test : torch.LongTensor, path, idx_map, rewrite=True):
    if os.path.exists(path+'train_inductive.map'):
        rewrite = False
        filenames = ['train', 'unlabeled', 'vali', 'test']
        ans = []
        for file in filenames:
            with open(path+file+'_inductive.map', 'r') as f:
                cache = []
                for line in f:
                    cache.append(idx_map.get(int(line)))
            ans.append(torch.LongTensor(cache))
        return ans

    idx_train = train
    idx_test = val
    cache = list(test.numpy())
    shuffle(cache)
    idx_val = cache[: idx_train.shape[0]]
    idx_unlabeled = cache[idx_train.shape[0]: ]
    idx_val = torch.LongTensor(idx_val)
    idx_unlabeled = torch.LongTensor(idx_unlabeled)

    print("\n\ttrain: ", idx_train.shape[0],
          "\n\tunlabeled: ", idx_unlabeled.shape[0],
          "\n\tvali: ", idx_val.shape[0],
          "\n\ttest: ", idx_test.shape[0])
    if rewrite:
        idx_map_reverse = dict(map(lambda t: (t[1], t[0]), idx_map.items()))
        filenames = ['train', 'unlabeled', 'vali', 'test']
        ans = [idx_train, idx_unlabeled, idx_val, idx_test]
        for i in range(4):
            with open(path+filenames[i]+'_inductive.map', 'w') as f:
                f.write("\n".join(map(str, map(idx_map_reverse.get, ans[i].numpy()))))

    return idx_train, idx_unlabeled, idx_val, idx_test


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if len(sparse_mx.nonzero()[0]) == 0:
    #if sparse_mx.nnz == 0:
        # 空矩阵
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_tensor_to_sparse(dense_mx):
    return sparse_mx_to_torch_sparse_tensor( sp.coo.coo_matrix(dense_mx) )


def makedirs(dirs: list):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    return