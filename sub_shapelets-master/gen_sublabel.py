#I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\mtivariate_sublabel\AtrialFibrillation.txt

import os

import numpy as np
def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
   # print('data',data.shape)#(2858,)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

dataset = "AtrialFibrillation"
# Load training data
# X_train = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_train.npy'), 'rb'))
# print(f"Shape X_train: {X_train.shape}")
# # load trainng data labels
# y_train = numpy.load(open(os.path.join('I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\data', f'{dataset}_train_labels.npy'), 'rb'))

data_dir=r'I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo'
X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'supervise', dataset+'.npy'))

# X_train = X[train_idx].transpose(0, 2, 1)
# y_train = y[train_idx]
# X_test = X[test_idx].transpose(0, 2, 1)
# y_test = y[test_idx]

label_path = r"I:\D\桌面\shapelet新\代码\Learning-Shapelets-main\demo\mtivariate_sublabel\AtrialFibrillation.txt"


# Generate a list of 1s for each sample
labels = [1] * len(y)

# Write the labels to a new file, separated by commas
with open(label_path, 'w') as file:
    file.write(",".join(map(str, labels)))

print(f"Labels file created with {len(y)} entries at {label_path}")


#====

# Read the labels from the file
labels = np.loadtxt(label_path, delimiter=',')  # Load the labels as a NumPy array

# Output the shape (dimensions) of the labels
print("Labels shape:", labels.shape) #(30,)