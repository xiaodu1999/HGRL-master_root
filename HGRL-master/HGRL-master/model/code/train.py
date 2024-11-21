from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pickle as pkl
from copy import deepcopy
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from utils import  accuracy, dense_tensor_to_sparse, resample, makedirs, load_data_time
from utils_inductive import transform_dataset_by_idx
from models import HGRL
import os, gc, sys
from print_log import Logger
from printt import printt
from print import print
logdir = "log/"
savedir = 'model/'
embdir = 'embeddings/'
makedirs([logdir, savedir, embdir])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

write_embeddings = True
HOP = 2

dataset = "AtrialFibrillation"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,#300
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,#LR
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,#WD
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,##DP
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--inductive', type=bool, default=False,
                    help='Whether use the transductive mode or inductive mode. ')
parser.add_argument('--dataset', type=str, default=dataset,
                    help='Dataset')
parser.add_argument('--repeat', type=int, default=1,
                    help='Number of repeated trials')
parser.add_argument('--node', action='store_false', default=True,
                    help='Use node-level attention or not. ')
parser.add_argument('--type', action='store_false', default=True,
                    help='Use type-level attention or not. ')

args = parser.parse_args()
dataset = args.dataset

args.cuda = not args.no_cuda and torch.cuda.is_available()
sys.stdout = Logger(logdir + "{}.log".format(dataset))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loss_list = dict()

def margin_loss(preds, y, weighted_sample=False):
    nclass = y.shape[1]
    preds = preds[:, :nclass]
    y = y.float()
    lam = 0.25
    m = nn.Threshold(0., 0.)
    loss = y * m(0.9 - preds) ** 2 + \
        lam * (1.0 - y) * (m(preds - 0.1) ** 2)

    if weighted_sample:
        n, N = y.sum(dim=0, keepdim=True), y.shape[0]
        weight = torch.where(y == 1, n, torch.zeros_like(loss))
        weight = torch.where(y != 1, N-n, weight)
        weight = N / weight / 2
        loss = torch.mul(loss, weight)

    loss = torch.mean(torch.sum(loss, dim=1))
    return loss

def nll_loss(preds, y):
    y = y.max(1)[1].type_as(labels)
    return F.nll_loss(preds, y)

def evaluate(preds_list, y_list):
    nclass = y_list.shape[1]
    preds_list = preds_list[:, :nclass]
    if not preds_list.device == 'cpu':
        preds_list, y_list = preds_list.cpu(), y_list.cpu()

    threshold = 0.5
    

    y_list = y_list.numpy()
    preds_probs = preds_list.detach().numpy()
    preds = deepcopy(preds_probs)
    preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
    preds[np.where(preds < 1)] = 0.0
    [precision, recall, F1, support] = \
        precision_recall_fscore_support(y_list, preds, average='macro',zero_division=0)
    ER = accuracy_score(y_list, preds) * 100
    printt(' Ac: %6.2f' % ER,
            'P: %5.1f' % (precision*100),
            'R: %5.1f' % (recall*100),
            'F1: %5.1f' % (F1*100),
            end="")
    return ER, F1

LOSS =  nll_loss


def train(epoch,
          input_adj_train, input_features_train, idx_out_train, idx_train,
          input_adj_val, input_features_val, idx_out_val, idx_val):

    printt('Epoch: {:04d}'.format(epoch+1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(input_features_train, input_adj_train)

    print('idx_out_train, idx_train',idx_out_train, idx_train)
    #input()
    if isinstance(output, list):
        O, L = output[0][idx_out_train], labels[idx_train]
    else:
        O, L = output[idx_out_train], labels[idx_train]
    
    print(' O, L',  O, L)
  
    loss_train = LOSS(O, L)
    printt(' | loss: {:.4f}'.format(loss_train.item()), end='')
    acc_train, f1_train = evaluate(O, L)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(input_features_val, input_adj_val)

    print('标签数量', labels.shape[1])#3
    print('output', output[0].shape)#torch.Size([30, 5])
    print('output', output[1].shape)#output torch.Size([2, 5])
    print('output', output[2].shape)#output torch.Size([12, 5])
    if isinstance(output, list): ###
        print('list') 
        #input()
        loss_val = LOSS(output[0][idx_out_val], labels[idx_val])
        printt(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[0][idx_out_val], labels[idx_val])
    else:
        loss_val = LOSS(output[idx_out_val], labels[idx_val])
        printt(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[idx_out_val], labels[idx_val])
    printt(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] = [loss_train.item()]

     # 更新学习率调度器
    scheduler.step(loss_val)  # 传入验证损失
    #input()
    acc_val, f1_val = results
    #input()
    return float(acc_val), float(f1_val)#.item() .item()


def test(epoch, input_adj_test, input_features_test, idx_out_test, idx_test):
    printt('测试========================')
    #printt(' '*90 if 'multi' in dataset else ' '*65, end='')

    t = time.time()
    model.eval()
    output = model(input_features_test, input_adj_test)

    if isinstance(output, list):
        loss_test = LOSS(output[0][idx_out_test], labels[idx_test])
        printt(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[0][idx_out_test], labels[idx_test])
    else:
        loss_test = LOSS(output[idx_out_test], labels[idx_test])
        printt(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[idx_out_test], labels[idx_test])
    printt(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] += [loss_test.item()]

    acc_test, f1_test = results
    return float(acc_test), float(f1_test)#.item()




path=root=r'/root'
path=os.path.join(path, 'HGRL-master_root','HGRL-master', 'multivariate_datasets')
#r'\root\HGRL-master_root\HGRL-master\HGRL-master\multivariate_datasets'

adj, features, labels, idx_train_ori, idx_val_ori, idx_test_ori, idx_map = load_data_time(path = path, dataset = dataset)

print('labels',labels.shape)

#input()

print(f"\nLength of idx_map: {len(idx_map)}")#Length of idx_map: 3200

printt("Transfer to be transductive.")
input_adj_train, input_features_train, idx_out_train = adj, features, idx_train_ori
input_adj_val, input_features_val, idx_out_val = adj, features, idx_val_ori
input_adj_test, input_features_test, idx_out_test = adj, features, idx_test_ori
idx_train, idx_val, idx_test = idx_train_ori, idx_val_ori, idx_test_ori


if args.cuda:
    N = len(features)
    for i in range(N):
        if input_features_train[i] is not None:
            input_features_train[i] = input_features_train[i].cuda()
        if input_features_val[i] is not None:
            input_features_val[i] = input_features_val[i].cuda()
        if input_features_test[i] is not None:
            input_features_test[i] = input_features_test[i].cuda()
    for i in range(N):
        for j in range(N):
            if input_adj_train[i][j] is not None:
                input_adj_train[i][j] = input_adj_train[i][j].cuda()
            if input_adj_val[i][j] is not None:
                input_adj_val[i][j] = input_adj_val[i][j].cuda()
            if input_adj_test[i][j] is not None:
                input_adj_test[i][j] = input_adj_test[i][j].cuda()
    labels = labels.cuda()
    idx_train, idx_out_train = idx_train.cuda(), idx_out_train.cuda()
    idx_val, idx_out_val = idx_val.cuda(), idx_out_val.cuda()
    idx_test, idx_out_test = idx_test.cuda(), idx_out_test.cuda()

printt('args.dropout',args.dropout)
#input()

FINAL_RESULT = []
for i in range(args.repeat):
# Model and optimizer
    printt("\n\nNo. {} test.\n".format(i+1))
    print()
    print('features', features[0])#3
    model = HGRL(nfeat_list=[i.shape[1] for i in features],
                 type_attention=args.type,
                 node_attention=args.node,
                    nhid=args.hidden,
                    nclass=labels.shape[1],
                    dropout=args.dropout,
                    gamma=0.1,
                    orphan=False,# True
                 )

    from torch.optim.lr_scheduler import ReduceLROnPlateau

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-4)
    if args.cuda:
        model.cuda()

    # Train model
    t_total = time.time()
    vali_max = [0, [0, 0], -1]


    for epoch in range(args.epochs):
        vali_acc, vali_f1 = train(epoch,
                         input_adj_train, input_features_train, idx_out_train, idx_train,
                         input_adj_val, input_features_val, idx_out_val, idx_val)
        test_acc, test_f1 = test(epoch,
                        input_adj_test, input_features_test, idx_out_test, idx_test)
        if vali_acc > vali_max[0]:
            vali_max = [vali_acc, (test_acc, test_f1), epoch+1]
            with open(savedir + "{}.pkl".format(dataset), 'wb') as f:
                pkl.dump(model, f)

            if write_embeddings:
                makedirs([embdir])
                with open(embdir + "{}.emb".format(dataset), 'w') as f:
                    for i in model.emb.tolist():
                        f.write("{}\n".format(i))
                with open(embdir + "{}.emb2".format(dataset), 'w') as f:
                    for i in model.emb2.tolist():
                        f.write("{}\n".format(i))

    printt("Optimization Finished!")
    printt("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    printt("The best result is: ACC: {0:.4f} F1: {1:.4f}, where epoch is {2}\n\n".format(
        vali_max[1][0],
        vali_max[1][1],
        vali_max[2]))
FINAL_RESULT.append(list(vali_max))

printt("\n")
for i in range(len(FINAL_RESULT)):
   
    printt("{}:\tvali:  {:.5f}\ttest:  ACC: {:.4f} F1: {:.4f}, epoch={}".format(
                    i,
        FINAL_RESULT[i][0],
        FINAL_RESULT[i][1][0],
        FINAL_RESULT[i][1][1],
        FINAL_RESULT[i][2]))