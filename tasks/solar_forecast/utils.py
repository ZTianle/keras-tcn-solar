import numpy as np
import pandas as pd
from sklearn import preprocessing





def data_generator(seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        n: # of data in the set
    """
    csv_data = pd.read_csv('station1_new.csv')
    csv_data_array = np.array(csv_data.tail(5479))
    x_all = np.array(csv_data_array[:, 2: 9])
    y_all = np.array(csv_data_array[:, 9])
    x = np.zeros([len(y_all)-seq_length, seq_length, 7])
    y = np.zeros([len(y_all)-seq_length, 1])
    print(x_all.shape, len(y_all))

    min_max_scaler = preprocessing.MinMaxScaler()
    x_all = min_max_scaler.fit_transform(x_all)

    for i in range(len(y_all)-seq_length):
        x[i, :, :] = x_all[i:i+seq_length, :]
        y[i] = y_all[i+seq_length]
   # print(x[-1,:,:], y[-1])

    print(x.shape, y.shape)
    return x, y

def data_generator_today(seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        n: # of data in the set
    """
    csv_data = pd.read_csv('station1_new.csv')
    csv_data_array = np.array(csv_data.tail(5479))
    x_all = np.array(csv_data_array[:, 2: 9])
    y_all = np.array(csv_data_array[:, 9])
    x = np.zeros([len(y_all)-seq_length, seq_length, 7])
    y = np.zeros([len(y_all)-seq_length, 1])
    print(x_all.shape, len(y_all))

    min_max_scaler = preprocessing.MinMaxScaler()
    x_all = min_max_scaler.fit_transform(x_all)

    for i in range(len(y_all)-seq_length):
        x[i, :, :] = x_all[i:i+seq_length, :]
        y[i] = y_all[i+seq_length-1]
   # print(x[-1,:,:], y[-1])

    print(x.shape, y.shape)
    return x, y

if __name__ == '__main__':
    csv_data = pd.read_csv('station1_new.csv')
    csv_data_array = np.array(csv_data.tail(5479))
    print(csv_data.shape)
    print(csv_data_array.shape)
    print(data_generator(seq_length=10))
