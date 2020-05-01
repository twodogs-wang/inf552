import numpy as np
import pandas as pd
import os
import copy

def __split_features(data):
    """
    split numerical labels and classify labels
    :param data:
    :return:
    """
    _classify_labels=['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage','regionCode']
    _num_list = []
    for i in data.columns:
        if i in _classify_labels:
            continue
        _num_list.append(i)
    return _classify_labels, _num_list


def __normalize(data,column_list):
    """
    normolize numerical columns
    :param data:
    :param column_list:
    :return:
    """
    _max = 0
    _min = 0
    if len(column_list) == 0:
        return data
    for feature in column_list:
        try:
            data[feature] = (data[feature] - np.min(data[feature])) / (np.max(data[feature] - np.min(data[feature])))
        except:
            print(feature)
            quit()
    return data


def  __prepare_data_for_linear_regression(data):
    _data = copy.deepcopy(data)
    if os.path.exists("train_data_for_LR.csv"):
        os.remove("train_data_for_LR.csv")
    print("generating dataset for linear regression and neural network........")
    _classify_labels, _numerical_labels = __split_features(_data)
    _data = __normalize(_data, _numerical_labels)
    _classify_labels.remove("notRepairedDamage")
    _data = _data.dropna().replace("-", 0).reset_index(drop=True)
    _data["regionCode"] = _data["regionCode"].apply(lambda x : int(str(x)[0]))
    _data = pd.get_dummies(_data, columns=_classify_labels)
    _data.to_csv("train_data_for_LR.csv",index=0)
    return


def __prepare_test_data_for_LR(data):
    _data = copy.deepcopy(data)
    _classify_labels, _numerical_labels = __split_features(_data)
    _data = __normalize(_data, _numerical_labels)
    _classify_labels.remove("notRepairedDamage")
    _data = _data.dropna().replace("-", 0).reset_index(drop=True)
    _data["regionCode"] = _data["regionCode"].apply(lambda x : int(str(x)[0]))
    _data = pd.get_dummies(_data, columns=_classify_labels)
    return _data
