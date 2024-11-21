import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = r'H:\icde异构图论文提交\data\消融等\respana.xlsx'  # 替换为您的文件路径
df = pd.read_excel(file_path)

# 去除最后两行
# df = df[:-2]  # 保留除了最后两行之外的所有行

# 假设数据格式如下：
# | 数据集 | 方法1准确率 | 方法2准确率 | 方法3准确率 |
# | ------ | ------------ | ------------ | ------------ |
# | 数据集1| 0.90        | 0.85        | 0.80        |
# | 数据集2| 0.92        | 0.88        | 0.82        |
# | 数据集3| 0.89        | 0.87        | 0.84        |

# 设置数据集名称为索引
df.set_index(df.columns[0], inplace=True)

# 绘制线状图
plt.figure(figsize=(10, 6))

# 创建图形
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column, linewidth=3)  # 设置线条宽度

# 添加图例和标签
#plt.title('Accuracy by Datasets', fontsize=20)  # 设置标题字体大小
plt.xlabel('Datasets', fontsize=20)  # 设置 x 轴标签字体大小
plt.ylabel('Accuracy', fontsize=20)  # 设置 y 轴标签字体大小
plt.xticks(rotation=45, fontsize=20)  # 设置 x 轴刻度字体大小
plt.yticks(fontsize=20)  # 设置 y 轴刻度字体大小
plt.legend(title='', fontsize=20)  # 设置图例字体大小
plt.grid()


# # 去掉背景方框线和网格线
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# # 去掉网格线
#plt.grid(False)


# 保存图形为图片文件
plt.savefig('representation.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 文件，分辨率为 300 DPI

# 显示图形
plt.tight_layout()
plt.show()