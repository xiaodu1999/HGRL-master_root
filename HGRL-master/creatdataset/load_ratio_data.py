import math
import os
import argparse
import numpy as np
from utils import read_dataset, read_multivariate_dataset

def getmap(path, dataset, train_indices, val_indices, test_indices):
    

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

path_dataset=r'H:\HGRL-master\sub_shapelets-master\demo'
path=r'H:\HGRL-master\HGRL-master\multivariate_datasets'
# output_dir = r'/root/SimTSC-main/SimTSC-main/tmp'

multivariate_datasets = [
    "AtrialFibrillation"]#, "FingerMovements", "PenDigits", 
   # "HandMovementDirection", "Heartbeat", "Libras",
  #  "MotorImagery", "NATOPS", "SelfRegulationSCP2", "StandWalkJump"]
#multivariate_datasets = ["Libras"]
def argsparser():
    parser = argparse.ArgumentParser("creat")
    parser.add_argument('--shot', help='How many labeled time-series per class', type=int, default=1)
    return parser
from  config_dataset import config_dataset
if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()
    args.bili=0.4#0.05

    # Seeding
    np.random.seed(0)

    # Create output directory

    #版本1 ，固定数量
    #output_dir = os.path.join(output_dir, 'multivariate_datasets_' + str(args.shot) + '_shot')#
    
    #版本2，固定比例
    output_dir = os.path.join(path_dataset, 'multivariate_datasets_' + 'flexible_' +str(args.bili)+ '_shot')#
    os.makedirs(output_dir, exist_ok=True)

    

    # Loop through all datasets
    for dataset_name in multivariate_datasets:

        train_len, test_len, _, _, nclass=config_dataset(dataset_name)
        num=train_len+test_len
        train_lenn = int(num * 0.8)  # 训练集数量
        test_lenn = int(num * 0.2)    # 测试集数量

        # 计算训练集中标记样本的数量
        
        labeled_count = int(train_lenn * args.bili)  # 标记样本数量

        # 计算每个类别的标记样本数量，上
        labeled_per_class = labeled_count / nclass
        labeled_per_class=math.ceil(labeled_per_class)

        # 向下取整
        #labeled_per_class_floor = math.floor(labeled_per_class)

        # 四舍五入
        #labeled_per_class_rounded = round(labeled_per_class)

        print(f"Processing dataset: {dataset_name}")

        # Read data
        X_train, y, train_idx, vali_idx, X_test, test_idx = read_multivariate_dataset(path_dataset, dataset_name, labeled_per_class)
        X = np.concatenate((X_train, X_test), axis=0)
        print('X, y, train_idx, vali_idx:', X.shape, y.shape, len(train_idx), len(vali_idx))
        
        # data = {
        #     'X': X,
        #     'y': y,
        #     'train_idx': train_idx,
        #     'vali_idx': vali_idx,
        #     'test_idx': test_idx
        # }
        
        print('  train_idx, vali_idx, test_idx',  train_idx, vali_idx,  test_idx)
        print('=============')
        getmap(path,dataset_name,train_idx, vali_idx,  test_idx)


    # X_train, y_train = data['X'][data['train_idx']], data['y'][data['train_idx']]
    # X_vali, y_vali = data['X'][data['vali_idx']], data['y'][data['vali_idx']]
    # X_test, y_test = data['X'][data['test_idx']], data['y'][data['test_idx']]

    # # 打印检查形状
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_train shape:", y_train)
    # print("X_vali shape:", X_vali.shape)
    # print("y_vali shape:", y_vali.shape)
    # print("y_vali shape:", y_vali)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)
    # print("y_test shape:", y_test)

    
        #=============== Save as .npy file
        # np.save(os.path.join(output_dir, f"{dataset_name}.npy"), data)
        # print(f"Data saved to {os.path.join(output_dir, f'{dataset_name}.npy')}")