import os
import pandas as pd
import numpy as np

def process_data(dataset_num=2, gender = 0):

    '''
    Processes the EMG data for a specified gender from the raw csv files 
    and returns a list of inputs and a list of the corresponding labels

    Arguments: dataset number (either 1 or 2), gender (0 for female, 1 for male)
    Returns: data_list, label_list; two numpy arrays containing the features and the labels respectively
    '''

    data_path = 'data/dataset' + str(dataset_num)

    label_dict = {'spher': 0, 'tip':1, 'palm':2, 'lat':3, 'cyl':4, 'hook':5}
    data_list = []
    label_list = []
    
    if gender or dataset_num==2:
        ignore_str = 'female'
    else:
        ignore_str = 'male'

    if dataset_num == 1:
        num_rows = 30
        num_col = 3000
    elif dataset_num ==2:
        num_rows = 100
        num_col = 2500
    else:
        raise NotImplementedError

    for file in os.listdir(data_path):
        if 'ch2' in file:
            continue
        if '_' + ignore_str + '_' in file:
            continue
        base_name = [i for i in file.split('_') if 'ch' not in i]
        label = label_dict[base_name[0]]
        print('processing data for: ' + '_'.join(base_name))
        channel_1 = pd.read_csv(data_path + '/' + base_name[0] + '_ch1_' + '_'.join(base_name[1:]), header=None).to_numpy()
        channel_2 = pd.read_csv(data_path + '/' + base_name[0] + '_ch2_' + '_'.join(base_name[1:]), header=None).to_numpy()
        for i in range(num_rows):
            data_list.append(np.array([channel_1[i, :], channel_2[i, :]]))
            label_list.append(label)
    data_list = np.array(data_list)
    label_list = np.array(label_list)
    return data_list, label_list