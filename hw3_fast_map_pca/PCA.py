import numpy as np
from typing import List
import copy
from pca_plot import __plot_pca

def __read_file(file_name: str) -> List[List[float]]:
    #read file
    try:
        _file = open(file_name,'r')
    except:
        print("file", file_name, "does not exist\n")
        quit()
    _content = []
    while True:
        _line = _file.readline()
        if not _line:
            break
        _temp = _line.split()
        try:
            _temp = [float(a) for a in _temp]
            _content.append(_temp)
        except:
            print("data is not pure digits\n")
            quit()
    _file.close()
    return _content


def __get_mean(_data: List[List[float]]) -> List[float]:
    # get the mean vector
    if len(_data) == 0 or len(_data[0]) == 0:
        print("data matrix is empty\n")
        return []

    _mean_list = []
    for i in range(len(_data[0])):
        _mean = (sum([x[i] for x in _data]))/len(_data)
        _mean_list.append(_mean)
    #print(_mean_list)
    return _mean_list


def __zero_centered(_data: List[List[float]], _mean: List[float]) -> List[List[float]]:
    #make the data matrix be zero centered
    if not len(_data[0]) == len(_mean):
        print("dimensions of data and mean do not match")
        quit()
    _updated_data = copy.deepcopy(_data)
    for i in range(len(_data)):
        for j in range(len(_data[0])):
            _updated_data[i][j] -= _mean[j]
            #print(_mean[j])

    return _updated_data


def __get_covariance_matrix(_left: List[List[float]], _right: List[List[float]]) -> List[List[float]]:
    #get covariance matrix
    _temp = np.dot(_left,_right)

    for i in range(len(_temp)):
        for j in range(len(_temp[0])):
            _temp[i][j] /= len(_right)
    return _temp


def __get_eigns(_covar_matrix: List[List[float]]) -> List[float]:
    #get eignvectors
    _temp = np.linalg.svd(_covar_matrix)
    return _temp[0].tolist()


def __reduce_dimension(_utr: List[float], _data: List[List[float]]) -> List[List[float]]:
    #reduce dimension by one and return new coordinates
    _ans = []
    for i in _data:
        _xi = np.transpose(i)
        _new_xi = np.dot(_utr,_xi)
        _ans.append(np.transpose(_new_xi).tolist())
    return _ans


def __pca(k: int) -> List[List[float]]:
    _data = __read_file("pca-data.txt")
    _mean = __get_mean(_data)
    _data = __zero_centered(_data, _mean)
    _covar_matrix = __get_covariance_matrix(np.transpose(_data), _data)
    _eignvectors = __get_eigns(_covar_matrix)
    _u_tr = []
    for i in range(k):
        _temp = [x[i] for x in _eignvectors]
        _u_tr.append(_temp)
    _new_data = __reduce_dimension(_u_tr, _data)
    return _new_data, _data, _u_tr


if __name__ == "__main__":
    k = 2
    _new_data, _old_data, _eignvectors = __pca(k)
    __plot_pca(_new_data,_old_data, _eignvectors[:k])