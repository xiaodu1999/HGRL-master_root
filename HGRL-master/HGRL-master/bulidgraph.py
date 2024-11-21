



import os
import numpy as np
from dtaidistance import dtw

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
            # 去掉行末的换行符并添加到列表
            patients.append(line.strip())  # 使用 strip() 去除换行符
    return patients

path_dataset=r'H:\HGRL-master\sub_shapelets-master\demo'
path_other=r'H:\HGRL-master\HGRL-master\multivariate_datasets'
dataset="AtrialFibrillation"
# Load training data
X, y, train_idx, test_idx = read_dataset_from_npyy(os.path.join(path_dataset, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))
X_train = X[train_idx].transpose(0, 2, 1)
y_train = y[train_idx]
X_test = X[test_idx].transpose(0, 2, 1)
y_test = y[test_idx]
print('X_train, y_train', X_train.shape, y_train.shape)#X_train, y_train (15, 2, 640) (15,)

def load_data_time(path=r'H:\nodetext\HGAT-master\multivariate_datasets', dataset="AtrialFibrillation"):
    print('读取各个小图')


    #======================timetotime
    matrixtt=read_dataset_from_npy(os.path.join(path, 'datasets_dtw', dataset+'.npy'))
    print('tt', matrixtt.shape)# (30, 30)

    
    #===========timetipatient
    matrix=read_dataset_from_npy(os.path.join(path, 'time_sub', dataset+'_timesub.npy'))
    matrixtp=matrix
    print('tsub')
    print(matrixtp.shape)#(30, 3)

    #===================timetoshapelet
    matrix=read_dataset_from_npy(os.path.join(path, 'time_shape', dataset+'_timeshape.npy'))

    matrixts=matrix
    # 输出结果
    print("ts")
    print(matrixts.shape)#30, 6)

    #============patienttoshapelet

    matrix=read_dataset_from_npy(os.path.join(path, 'sub_shape', dataset+'_subshape.npy'))

    matrixps=matrix
    # 输出结果
    print("subs")
    print(matrixps.shape)

    #==pp
    matrix=read_dataset_from_npy(os.path.join(path, 'sub_sub', dataset+'_subsub.npy'))
    matrixpp=matrix
    print('matrixsubsub', matrixpp.shape)
    #=================shapelettoshapelet
    matrix=read_dataset_from_npy(os.path.join(path, 'shape_shape', dataset+'_shapeshape.npy'))

    matrixss=matrix
    print('ss')
    print(matrixss.shape)#(6, 6)


    return  matrixtt,matrixtp,matrixts,matrixps,matrixpp,matrixss


a=1
if a==1:
    matrixtt,matrixtp,matrixts,matrixps,matrixpp,matrixss=load_data_time(path=path_other, dataset=dataset)

    matrixpt=matrixtp.T
    matrixst=matrixts.T
    matrixsp=matrixps.T
    print("matrixtt:", matrixtt.shape)
    print("matrixtp:", matrixtp.shape)
    print("matrixts:", matrixts.shape)

    print("matrixpt:", matrixpt.shape)
    print("matrixpp:", matrixpp.shape)
    print("matrixps:", matrixps.shape)

    print("matrixst:", matrixst.shape)
    #print("matrixst:", matrixst)
    print("matrixsp:", matrixsp.shape)
    #print("matrixsp:", matrixsp)
    print("matrixss:", matrixss.shape)
    #print("matrixss:", matrixss)
    print('异构图生成')
    #input()
    big_matrix = np.block([[matrixtt,matrixtp,matrixts],
                        [matrixpt,matrixpp,matrixps],
                        [matrixst,matrixsp,matrixss]])

    print(big_matrix.shape)#(39, 39)

    # 创建完整路径
    full_path = os.path.join(path_other,'finalmatrix', f"{dataset}.npy")

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # 保存矩阵为 .npy 文件
    np.save(full_path, big_matrix)

    # 输出保存的文件路径
    print(f"Matrix saved to: {full_path}")


len_train=len(y_train)
len_test=len(y_test)
def getmap1(path, dataset):

    len=len_train+len_test
    len_train
    len_test
    outpath=path + 'map' + dataset

     # save mappings
    with open(outpath + 'train.map', 'w') as f:
        f.write('\n'.join())

    with open(outpath + 'val.map', 'w') as f:
        f.write('\n'.join())

    with open(outpath + 'test.map', 'w') as f:
        f.write('\n'.join())


def getmap(path, dataset, len_train, len_val, len_test):
    # 总长度
    #total_len = len_train + len_val + len_test
    total_len = len_train + len_test

    # 生成索引
    train_indices = list(range(len_train))
    # val_indices = list(range(len_train, len_train + len_val))
    # test_indices = list(range(len_train + len_val, total_len))
    test_indices = list(range(len_train , total_len))
    val_indices=test_indices

    # 输出路径
    outpath = os.path.join(path, 'map', dataset)

    # 确保输出目录存在
    os.makedirs(outpath, exist_ok=True)

    # 保存训练集映射
    with open(os.path.join(outpath, 'train.map'), 'w') as f:
        f.write('\n'.join(map(str, train_indices)))

    # 保存验证集映射
    with open(os.path.join(outpath, 'val.map'), 'w') as f:
        f.write('\n'.join(map(str, val_indices)))

    # 保存测试集映射
    with open(os.path.join(outpath, 'test.map'), 'w') as f:
        f.write('\n'.join(map(str, test_indices)))

    print(f"Mappings saved to {outpath}")

b=2
if b==1:
    len_val=len_test
    getmap(path_other, dataset, len_train, len_val, len_test)
