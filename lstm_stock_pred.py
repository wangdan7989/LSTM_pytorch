#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset 
import tqdm
import csv
from torch.autograd import Variable

#torch.random.manual_seed(0)
#np.random.seed(0)

class LSTM_model(nn.Module):
    def __init__(self, input_dim, input_size, rnn_unit, run_layer=1):
        super(LSTM_model, self).__init__()
        self.dim = input_dim
        self.rnn_unit = rnn_unit
        #输入输出层为全联接层
        self.emb_layer = nn.Linear(input_dim, input_size)#input_dim:全连接层的输入层的特征为度；rnn_unit:输出的特征维度;input_size:lstm的输入维度
        self.out_layer = nn.Linear(rnn_unit, input_dim)
        self.rnn_layer = run_layer# 单层lstm
        #input_size：lstm的输入
        #input_size：x的特征维度；hidden_size：隐藏层的特征维度；num_layers：lstm隐层的层数，默认为1；num_layers：lstm隐层的层数，默认为1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=rnn_unit, num_layers=self.rnn_layer, batch_first=True)
    
    def init_hidden(self, x):
        batch_size = x.shape[0]
        rtn = (torch.zeros(self.rnn_layer, batch_size, self.rnn_unit, device=x.device).requires_grad_(),
                torch.zeros(self.rnn_layer, batch_size, self.rnn_unit, device=x.device).requires_grad_())
        return rtn

    def forward(self, input_data, h0=None):
        # batch x time x dim
        h0 = h0 if h0 else self.init_hidden(input_data)
        x = self.emb_layer(input_data)#输入层
        
        output, hidden = self.lstm(x, h0)#lstm层：输入：input, (h0, c0)
                                                #输出：output, (hn,cn)

        out = self.out_layer(output[:,-1,:].squeeze()).squeeze()#输出层
        return out, hidden


class StockDataset(Dataset):
    def __init__(self, data_all, T=10, train_flag=True, data_train_ratio=0.8):
        # read data
        self.train_flag = train_flag
        self.data_train_ratio = data_train_ratio
        self.T = T # use 10 data to pred
        data_all = (data_all - np.mean(data_all)) / np.std(data_all)#normalization  data_all： numpy 1 dim
        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_all))
            self.data = data_all[:self.data_len]
        else:
            self.data_len = int((1-self.data_train_ratio) * len(data_all))
            self.data = data_all[-self.data_len:]
        print("data len:{}".format(self.data_len))


    def __len__(self):
        return self.data_len-self.T

    def __getitem__(self, idx):
        return self.data[idx:idx+self.T], self.data[idx+self.T]


def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, size_average=True)
    return loss

def train_once(model, dataloader, optimizer):
    model.train()
    
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        # data: batch, time
        data = data.unsqueeze(2)
        data, label = Variable(data.float()), Variable(label.float())
        output, _ = model(data)
        optimizer.zero_grad()
        loss = l2_loss(output, label)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.detach().item()

    loss_epoch /= len(loader)
    return loss_epoch
    print("epoch:{:5d}, loss:{:6.3f}".format(epoch, loss_epoch))


def eval_once(model, dataloader):
    model.eval()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        # data: batch, time x 1
        data = data.unsqueeze(2)
        data, label = data.float(), label.float()
        output, _ = model(data)
        #print(output)
        loss = l2_loss(output, label)
        loss_epoch += loss.detach().item()
    loss_epoch /= len(loader)
    return loss_epoch

def eval_plot(model, dataloader, rnn_unit=0, plot_flag=True, write_flag = False, writeresult_path = 'result.csv'):
    dataloader.shuffle = False
    preds = []
    labels = []
    model.eval()
    loader = tqdm.tqdm(dataloader)

    #temp = set()
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        # data: batch, time x 1
        data, label = data.float().unsqueeze(2), label.float()
        output, _ = model(data)
        loss = l2_loss(output, label)
        loss_epoch += loss.detach().item()

        #preds += [output.item()]#test batch_size = 1
        preds += (label.detach().tolist())
        labels+=(label.detach().tolist())
    loss_epoch /= len(loader)
    print("\n preds:", preds,"\n labels:",labels)
    print("test loss:", loss_epoch)

    #write to file
    if(write_flag):
        with open(writeresult_path, "w", newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(labels)
            writer.writerows(preds)

    #plot
    if(plot_flag):
        #fig, ax = plt.subplots()
        data_x = list(range(len(preds)))
        plt.plot(data_x, preds, **{"label": "Predicted values", "color":"blue", "linestyle":"-.",  "marker":","})
        plt.plot(data_x, labels, **{"label": "Actual values", "color":"red", "linestyle":":", "marker":","})
        plt.title(str(rnn_unit))
        plt.legend()
        #plt.savefig('./dataset/result_jpg/'+str(rnn_unit) + '.pdf')
        plt.show()

    return preds, labels

def Save_model_param(model_path = "", write_path = ""):
    model = torch.load(model_path)
    with open(write_path,'w+') as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                #print(param.detach().tolist())
                f.write('%s\n' % name)
                f.write('%s\n\n' % param.detach().tolist())
        print()

def main(data_all, train_flag = True, test_flag = False, model_path = "model.pkl", batch_size=64, input_dim=1, input_size=8, rnn_unit=8, total_epoch=101, epoch_times=10):

    #data_all为数据预处理后的数据集
    data_all = data_all
    dataset_train = StockDataset(data_all)
    dataset_val = StockDataset(data_all, train_flag=False)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = LSTM_model(input_dim=input_dim, input_size=input_size, rnn_unit=rnn_unit)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    total_epoch = total_epoch
    if(train_flag):
        for epoch_idx in range(total_epoch):
            train_loss = train_once(model, train_loader, optimizer)
            print("stage: train, epoch:{:5d}, loss:{:6.3f}".format(epoch_idx, train_loss))
            if epoch_idx%epoch_times==0:
                eval_loss = eval_once(model, val_loader)
                print("stage: test, epoch:{:5d}, loss:{:6.3f}".format(epoch_idx, eval_loss))

        torch.save(model, model_path)
    if(test_flag):
        model = torch.load(model_path)
        preds, labels = eval_plot(model, val_loader, rnn_unit=rnn_unit)
        return preds, labels

if __name__ == "__main__":
#使用该文件时，将自己预处理好的数据，一维的numpy array格式，将参数传入main函数即可，返回预测的结果和实际的标签
    #数据预处理部分
    with open("./dataset/dataset_1.csv", "r", encoding="GB2312") as fp:
        data_pd = pd.read_csv(fp)
    data_all = np.array(data_pd['最高价'])[::-1]

    main(data_all, train_flag = False, test_flag = True)







