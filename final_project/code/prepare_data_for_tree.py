import pandas as pd
import os
from data_pre_processing import __drop_unneeded_cols, box_plot_filter, remove_null, data_filter
import numpy as np


def __add_potential_features(data):
    """
    add some new features to current data set for better accuracy


    """
    sub_data = data.groupby("brand")
    _temp0 = {}
    for brand, brand_data in data.groupby("brand"):
        temp1 = {}
        price_col = brand_data['price']
        temp1['price_median_of_this_brand'] = np.median(price_col)
        temp1['price_average_of_this_brand'] = np.mean(price_col)
        temp1['price_max'] = np.max(price_col)
        temp1['price_min'] = np.min(price_col)
        _temp0[brand] = temp1
    _temp0 = pd.DataFrame(_temp0).T.reset_index().rename(columns={"index":"brand"})
    data = data.merge(_temp0, how = "left", on = "brand")
    #print(data)
    return data, _temp0


def __prepare_train_data_for_tree(_data):
    """
    prepare train dataset for decision tree and random forest

    return: new cols added to dataset, will be added to test dataset later
    """
    if os.path.exists("train_data_for_tree.csv"):
        os.remove("train_data_for_tree.csv")
    print("generating dataset for decision tree and random forest.....")
    _data['notRepairedDamage'].replace('-', np.nan, inplace=True)
    _train_data = data_filter(_data, 'power', 3)  # filter strange values
    _train_data['age'] = round(((pd.to_datetime(_train_data['creatDate'], format='%Y%m%d',\
                                                errors='coerce') - pd.to_datetime(_train_data['regDate'],\
                                                            format='%Y%m%d',errors='coerce')).dt.days) / 365, 3)
    _train_data = remove_null(_train_data, 'age')
    _train_data = __drop_unneeded_cols(_train_data)
    _train_data , added_features= __add_potential_features(_train_data)
    _train_data.to_csv('train_data_for_tree.csv', index=0)
    return added_features


def __prepare_test_data_for_tree(_data, new_cols):
    _data['age'] = round(((pd.to_datetime(_data['creatDate'], format='%Y%m%d', \
                                                errors='coerce') - pd.to_datetime(_data['regDate'], \
                                                        format='%Y%m%d',errors='coerce')).dt.days) / 365, 3)
    _data = __drop_unneeded_cols(_data)
    _data = _data.dropna().replace("-", 0).reset_index(drop=True)
    _data = _data.merge(new_cols, how = "left", on = "brand")
    return _data


