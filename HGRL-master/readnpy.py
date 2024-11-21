import numpy as np

# 读取npy文件
data = np.load(r'H:\nodetext\HGAT-master\multivariate_datasets\finalmatrix\AtrialFibrillation.npy')

# 输出矩阵的维度
print("矩阵的维度:", data.shape)  # (39, 39)

# 定义一个函数，处理大于10的元素
def process_value(value):
    if value > 10:
        return round(value / 100, 1)
    else:
        return round(value, 1)

# 使用向量化的方式处理数组中的每个元素
formatted_data = np.vectorize(process_value)(data)

# 输出处理后的数据（如果需要）
print("处理后的数据:\n", formatted_data)

# 保存处理后的数据为新的npy文件
np.save(r'H:\nodetext\HGAT-master\multivariate_datasets\finalmatrix\AtrialFibrillation_formatted.npy', formatted_data)

print("数据已保存为新的npy文件。")
