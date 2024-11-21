

import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from demo.sub_label_text.sublabel_text import dataset_info
from transformers import T5Tokenizer, T5Model
import torch

dataset='AtrialFibrillation'
label_path = r"I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\mtivariate_sublabel"
label_path = os.path.join(label_path, dataset+'.txt')
labels_sub = np.loadtxt(label_path, delimiter=',')  # Load the labels as a NumPy array
print('labels_sub', labels_sub.shape)#(30,)


# 将 ID 列表转换为 NumPy 数组，并重塑形状
#labels_array = np.array(labels_sub).reshape(-1, 1)

a=2
#====onhot
if a==1:
    labels_array = np.array(labels_sub).reshape(-1, 1)
    # 创建 OneHotEncoder 实例
    encoder = OneHotEncoder(sparse_output=False)

    # 用整个标签种类进行 fit
    encoder.fit(labels_array)

    # 获取每种标签对应的独热编码
    unique_labels = np.unique(labels_sub).reshape(-1, 1)
    one_hot_encoded = encoder.transform(unique_labels)

    print('个体标签特征', one_hot_encoded.shape)
    print(one_hot_encoded)

#==========T5
else:
    labels_sub = np.array(labels_sub)#.reshape(-1, 1)
    # 获取唯一的标签
    unique_labels = np.unique(labels_sub)

    # 打印唯一标签
    print("Unique labels:", unique_labels)
    # 加载 T5 模型和 tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small")

    # 将标签转换为文本并生成嵌入
    # 存储嵌入
    embeddings = []

    # 遍历每个标签
    for label in unique_labels:
        # 获取对应的 sublabel
        sublabel = dataset_info['AtrialFibrillation']['sublabel']
        
        # 检查标签是否在 sublabel 中
        if label in sublabel:
            # 生成文本描述
            text = f"{dataset_info['AtrialFibrillation']['prefix']} {sublabel[label]} {dataset_info['AtrialFibrillation']['suffix']}"
            print(f"Generated text: {text}")

            # 使用 tokenizer 对文本进行编码
            inputs = tokenizer(text, return_tensors="pt")

            
            # 生成解码器输入，假设需要一个开始标记
            decoder_input_ids = tokenizer.encode("summarize: ", return_tensors='pt')  # 这里的输入可以根据任务修改

            # 使用模型生成嵌入
            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)
            
            # 获取最后一层的隐藏状态并求平均，作为嵌入向量
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
        else:
            print(f"Label {label} not found in sublabel.")

    # 转换为 NumPy 数组
    embeddings = np.array(embeddings)
    print("Embeddings shape:", embeddings.shape)#Embeddings shape: (2, 512)

#===保存
save=r'H:\HGRL-master\sub_shapelets-master\subject_feature'

subshape_path = os.path.join(save, dataset+'_subfeature.npy')

# np.save(subshape_path, one_hot_encoded)

np.save(subshape_path, embeddings)
