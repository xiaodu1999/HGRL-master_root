import numpy as np
import pandas as pd

# 读取 .npy 文件
data = np.load(r'H:\HGRL-master\HGRL-master\multivariate_datasets\finalmatrix\AtrialFibrillation.npy')

# 如果数据是多维数组，你可能需要将其转换为适当的形状，例如：
# data = data.reshape(-1, data.shape[-1])  # 将数据展平为二维数组

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv('output_file.csv', index=False)
