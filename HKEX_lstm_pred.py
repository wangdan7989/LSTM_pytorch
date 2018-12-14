import csv
import numpy as np
import pandas as pd
import lstm_stock_pred

def Get_data_all(file_path = './dataset/HKEX_HSIF_2013-2018_all4rows.csv'):
    pd_data = pd.read_csv(file_path)
    data_ahdh = np.array(pd_data['afterhour_dailyhigh-0'])
    # data_ahdh = np.array(pd_data['afterhour_closeprice-0'])
    data_all = data_ahdh[data_ahdh > 0]
    return data_all
def Write_result(file_path = "./dataset/HKEX_all4rows_result.csv", restore_data = True,labels=[], preds =[]):
    #预测结果和原始label还原
    if(restore_data):
        data_all = Get_data_all()
        all_mean = np.mean(data_all)
        all_std = np.std(data_all)

        preds = np.dot(preds, all_std) + all_mean
        labels = np.dot(labels, all_std) + all_mean

    with open(file_path, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(labels)
        writer.writerow(preds)

def main():
    data_all = Get_data_all()
    preds, labels = lstm_stock_pred.main(data_all, train_flag = False, test_flag = True)
    Write_result(labels=labels, preds =preds)

if __name__ == "__main__":

    main()
