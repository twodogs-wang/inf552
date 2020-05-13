import os
import copy
import pandas as pd


def box_plot_filter(data,column_name, scale):
    """
    remove starnge values base on box-plot
    :param column:
    :param scale:
    :return: indices that is out of range
    """
    column = data[column_name]
    IQR = scale * (column.quantile(0.75) - column.quantile(0.25))
    low = column.quantile(0.25) - IQR
    up = column.quantile(0.75) + IQR
    index = data[(data[column_name]>up)|(data[column_name]<low)].index
    return index


def __replace_nan_with_mean(data, column):
    pass



def data_filter(data, column_name,scale):
    
    """
    use box-plot find strange values and replace them with mean value
    :param data:
    :param column_name:
    :param scale:
    :return:
    """
    index = (box_plot_filter(data,column_name,scale))
    _new_data = copy.deepcopy(data)
    _new_data.drop(index = index)
    mean = _new_data[column_name].mean()
    mean = round(mean,3)
    for i in index:
        data.loc[i,column_name] = mean
    return data


def remove_null(data,column_name):
    """
    remove rows with null values
    :param data:
    :param column_name:
    :return:
    """
    data[column_name] = data[column_name].fillna('-1.14')
    _index = data[(data[column_name]=='-1.14')].index.tolist()
    data = data.drop(_index)
    return data


def __drop_unneeded_cols(data):
    """
    drop some cols that no longer needed
    """
    data = data.drop(['SaleID','name','regDate','creatDate','offerType','seller'], axis=1)
    return data


