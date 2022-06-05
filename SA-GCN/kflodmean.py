from sklearn.model_selection import KFold, StratifiedKFold
import scipy.io
import torch
from torch_geometric.data import Data
import scipy.io
import numpy as np
from torch_geometric.data import DataLoader
import argparse
from networks import Net
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=64,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.1,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=10000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
args.num_classes = 2   #类别数
args.num_features =2   #特征数
args.num_nodes = 116
#原始数据116*170(一次相关)+一次相关116*116(两次相关)
#加载数据
Data1 = scipy.io.loadmat(r'data/fMRI.mat')  # 读取mat文件
data1 = Data1['fMRImciNcT']
lab=Data1['lab']
lab[lab == -1] = 0
dataall= np.linspace(0, 91, num=92, dtype=int)
floder = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)# # 存放5折的测试集集划分
train_files = []  # 存放5折的训练集划分
test_files = []

train = []
test = []
trainall=[]
mm=0
acc = 0
#数据预处理
M = 170
W = 130
S = 5
K = np.floor((M - W) / S).astype(int)  # 滑动窗
for i in range(1):
    for j, (Trindex, Tsindex) in enumerate(floder.split(np.squeeze(data1,axis=0),np.squeeze(lab,axis=0))):
        train_list = []
        test_list = []
        Tr_data = []
        TrLab=[]
        for k1 in Trindex:
            train_data=data1[0][k1]
            Tr_data.append(train_data)
            train_lab=lab[0][k1]
            TrLab.append(train_lab)
        for step,series in enumerate(Tr_data):
            ss=Tr_data[step]
            corr_mean1 = np.mean(ss, axis=1, dtype=np.float64, out=None, keepdims=True)
            corr_var1 = np.var(ss, axis=1, dtype=np.float64, out=None, ddof=0, keepdims=True)
            d1 = np.hstack((corr_mean1, corr_var1))
            for l1 in range(K):
                id1 = []
                id2 = []
                idx1 = l1 * S  # 窗口起始位置
                idx2 = l1 * S + W  # 窗口结尾位置
                Bigdata = series[:, idx1:idx2]
                lcorr = np.corrcoef(Bigdata)
                hlcorr = np.corrcoef(lcorr)
                corr_mean2 = np.mean(lcorr, axis=1, dtype=np.float64, out=None, keepdims=True)
                corr_var2 = np.var(lcorr, axis=1, dtype=np.float64, out=None, ddof=0, keepdims=True)
                d2 = np.hstack((corr_mean2, corr_var2))
                for p1 in range(116):
                    for q1 in range(116):
                        if -0.6 < lcorr[p1][q1] < 0.6:
                            lcorr[p1][q1] = 0
                        else:
                            lcorr[p1][q1] = 1
                            id1.append([p1, q1])
                for p2 in range(116):
                    for q2 in range(116):

                        if -0.6 < hlcorr[p2][q2] < 0.6:
                            hlcorr[p2][q2] = 0
                        else:
                            hlcorr[p2][q2] = 1
                            id2.append([p2, q2])
                edge1 = torch.tensor(id1, dtype=torch.long)
                edge2 = torch.tensor(id2, dtype=torch.long)
                y = torch.tensor(TrLab[step], dtype=torch.long)

                train_garph_data = Data(x=torch.tensor(d1, dtype=torch.float32),edge1_index=edge1.t().contiguous(), y=y,s=torch.tensor(np.corrcoef(series), dtype=torch.float32),edge2_index=edge2.t().contiguous(),)
                train_list.append(train_garph_data)




#----------------------------------------------------------------------------------------------------------------------------------------------------------
        Ts_data = []
        TsLab = []
        for k2 in Tsindex:
            test_data = data1[0][k2]
            Ts_data.append(test_data)
            ts_lab = lab[0][k2]
            TsLab.append(ts_lab)
        for step, series in enumerate(Ts_data):
            uu=Ts_data[step]
            corr_mean3 = np.mean(uu, axis=1, dtype=np.float64, out=None, keepdims=True)
            corr_var3 = np.var(uu, axis=1, dtype=np.float64, out=None, ddof=0, keepdims=True)
            d3 = np.hstack((corr_mean3, corr_var3))
            for l2 in range(K):
                id11 = []
                id22 = []
                idx1 = l2 * S  # 窗口起始位置
                idx2 = l2 * S + W  # 窗口结尾位置
                Bigdata = series[:, idx1:idx2]
                lcorr = np.corrcoef(Bigdata)
                hlcorr = np.corrcoef(lcorr)
                corr_mean4 = np.mean(lcorr, axis=1, dtype=np.float64, out=None, keepdims=True)
                corr_var4 = np.var(lcorr, axis=1, dtype=np.float64, out=None, ddof=0, keepdims=True)
                d4 = np.hstack((corr_mean4, corr_var4))
                for p3 in range(116):
                    for q3 in range(116):
                        if -0.6 < lcorr[p3][q3] < 0.6:
                            lcorr[p3][q3] = 0
                        else:
                            lcorr[p3][q3] = 1
                            id11.append([p3, q3])
                for p4 in range(116):
                    for q4 in range(116):

                        if -0.6 < hlcorr[p4][q4] < 0.6:
                            hlcorr[p4][q4] = 0
                        else:
                            hlcorr[p4][q4] = 1
                            id22.append([p4, q4])
                edge11 = torch.tensor(id11, dtype=torch.long)
                edge22 = torch.tensor(id22, dtype=torch.long)
                y = torch.tensor(TsLab[step], dtype=torch.long)

                test_garph_data = Data(x=torch.tensor(d3, dtype=torch.float32),
                              edge1_index=edge11.t().contiguous(), y=y,
                              s=torch.tensor(np.corrcoef(series), dtype=torch.float32),
                              edge2_index=edge22.t().contiguous(), )
                test_list.append(test_garph_data)

        print("第" + str(i) + "次外循环")
        model = Net(args).to(args.device)
        model.load_state_dict(torch.load("./chushi.pth"))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_loader = DataLoader(
            dataset=train_list,
            batch_size=args.batch_size,
            shuffle=True
        )
        print("第" + str(mm) + "次训练")
        test_loader = DataLoader(
            dataset=test_list,
            batch_size=1
        )
        print("第" + str(mm) + "次测试")
        mm = mm + 1

        AUC_0 = []
        AUC_1 = []

        def test(model, loader):
            model.eval()
            correct = 0.
            loss = 0.
            for data1 in loader:
                data1 = data1.to(args.device)
                out = model(data1)
                pre = torch.nn.functional.softmax(out)
                # pre=out.cpu().detach().numpy()
                # pre = np.argmax(pre)
                # print(pre)#预测的概率
                # print(data.y)#真实标签

                pred = out.max(dim=1)[1]

                correct += pred.eq(data1.y).sum().item()
                loss += F.nll_loss(out, data1.y, reduction='sum').item()

            return correct / len(loader.dataset), loss / len(loader.dataset)

        min_loss = 0
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            for iiii, data2 in enumerate(train_loader):
                data2 = data2.to(args.device)
                out = model(data2)
                loss = F.nll_loss(out, data2.y)
                # print("Training loss:{}".format(loss.item()))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            val_acc, val_loss = test(model, test_loader)
            print("Validation loss:{}".format(val_loss))
            if val_acc > min_loss:
                torch.save(model.state_dict(), 'latest.pth')
                print("Model saved at epoch{}".format(epoch))
                min_loss = val_acc
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break

        model = Net(args).to(args.device)
        model.load_state_dict(torch.load('latest.pth'))
        test_acc, test_loss = test(model, test_loader)
        print(test_acc)
        acc = acc + test_acc

print(acc/5)



