import csv
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

def dataset_1_train_test_split(data_all_path, data_train_path, data_test_path, T = 10, data_train_ratio = 0.8):
    with open(data_all_path, "r", encoding="GB2312") as fp:
        data_all = fp.readlines()
    data_all = [data.strip() for data in data_all]
    data_all = data_all[1:]
    data_train_test = [data.split(',')[1] for data in data_all]
    print(data_train_test)
    data_np = np.array(data_train_test, dtype = 'float_')[::-1]
    data_np = (data_np-np.mean(data_np))/np.std(data_np)

    train_len = int(data_np.shape[0] * data_train_ratio)
    data_train = data_np[:train_len]
    data_test = data_np[train_len:]
    data_train_label_data = [np.hstack([data_train[idx+T], data_train[idx:idx+T]]) for idx in range(len(data_train)-T-1)]
    data_test_label_data = [np.hstack([data_test[idx+T], data_test[idx:idx+T]]) for idx in range(len(data_test)-T-1)]

    print(len(data_test_label_data[0]))

    with open(data_train_path, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(data_train_label_data)
    with open(data_test_path, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(data_test_label_data)

def dataHKEX_train_test_split(data_all, data_train_path, data_test_path, T = 10, data_train_ratio = 0.9):

    data_np = data_all
    data_np = (data_np-np.mean(data_np))/np.std(data_np)

    train_len = int(data_np.shape[0] * data_train_ratio)
    data_train = data_np[:train_len]
    data_test = data_np[train_len:]
    data_train_label_data = [np.hstack([data_train[idx+T], data_train[idx:idx+T]]) for idx in range(len(data_train)-T-1)]
    data_test_label_data = [np.hstack([data_test[idx+T], data_test[idx:idx+T]]) for idx in range(len(data_test)-T-1)]

    with open(data_train_path, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(data_train_label_data)
    with open(data_test_path, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(data_test_label_data)
    print(data_all.shape)
    print(len(data_train_label_data))
    print(len(data_test_label_data))

if __name__ == "__main__":
    #print(123)
    #data_all_path = "./dataset/HKEX_TRAIN_TEST/dataset_1.csv"


    file_path = './dataset/HKEX_HSIF_2013-2018_all4rows.csv'
    pd_data = pd.read_csv(file_path)
    # afterhour_daily_high  train and test
    # data_train_path = "./dataset/HKEX_TRAIN_TEST/afterhour_dh_train.csv"
    # data_test_path = "./dataset/HKEX_TRAIN_TEST/afterhour_dh_test.csv"
    # data_ahdh = np.array(pd_data['afterhour_dailyhigh-0'])

    # afterhour_daily_high  train and test
    data_train_path = "./dataset/HKEX_TRAIN_TEST/day_dh_train.csv"
    data_test_path = "./dataset/HKEX_TRAIN_TEST/dat_dh_test.csv"
    data_ahdh = np.array(pd_data['day_dailyhigh-0'])

    data_all = data_ahdh[data_ahdh > 0]



    dataHKEX_train_test_split(data_all, data_train_path, data_test_path)