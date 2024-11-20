import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# 假设数据集为二维数组，每行是一个样本
data = np.random.rand(50, 50)  # 示例数据集 (1000 个样本，每个样本长度为 50)

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from joblib import Parallel, delayed

def calculate_dtw_pair(idx_i, idx_j, data):
    # 计算两个样本之间的DTW距离
    distance, _ = fastdtw(data[idx_i].reshape(-1, 1), data[idx_j].reshape(-1, 1), dist=euclidean)
    similarity = 1 / (1 + distance)
    return idx_i, idx_j, similarity

def calculate_dtw_similarity_batch(data, batch_size):
    n = len(data)
    similarity_matrix = np.zeros((n, n))
    num_batches = (n + batch_size - 1) // batch_size
    batches = [range(i * batch_size, min((i + 1) * batch_size, n)) for i in range(num_batches)]

    results = []
    # 批次内计算DTW相似度
    for i, batch_i in enumerate(batches):
        for j, batch_j in enumerate(batches):
            if j < i:
                continue
            indices_i = list(batch_i)
            indices_j = list(batch_j)
            results.extend(Parallel(n_jobs=-1)(delayed(calculate_dtw_pair)(idx_i, idx_j, data) for idx_i in indices_i for idx_j in indices_j))

    # 填充相似度矩阵
    for idx_i, idx_j, similarity in results:
        similarity_matrix[idx_i, idx_j] = similarity
        if idx_i != idx_j:  # 利用对称性
            similarity_matrix[idx_j, idx_i] = similarity

    return similarity_matrix

# 参数设置
batch_size = 10  # 每个批次的大小

# 计算相似度矩阵
similarity_matrix = calculate_dtw_similarity_batch(data, batch_size)

# 打印结果
print("DTW 相似度矩阵计算完成。", similarity_matrix.shape)
