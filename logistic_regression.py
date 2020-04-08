from typing import List
import random
import numpy as np
from perceptron_plot import __plot
import math


def __read_file(file_name: str) -> List:
    f1 = open(file_name, 'r')
    _list = []
    while True:
        _line = f1.readline()
        if not _line:
            break
        _line = _line.split(",")
        _line = [float(x) for x in _line]
        _list.append(_line[0:3]+[_line[-1]])
    f1.close()
    return _list


def __random_w(_dimension:int) -> List[float]:
    _list = []
    #_list.append(-0.001)
    for i in range(_dimension + 1):
        _list.append(random.random())
    return _list


def __sigmoid_func(_s: float) -> float:
    _temp = math.exp(_s)
    return _temp/(1 + _temp)


def __sigmoid_func_prime(_s: float) -> float:
    _temp = math.exp(_s)
    return ((1 + _temp) * _temp - pow(_temp, 2)) / pow(1+_temp, 2)


def __cal_ein(_w: List[float], _data: List[float]) -> float:
    _N = len(_data)
    _ans = 0
    for i in range(len(_data)):
        _xi = _data[i][0:4]
        _yi = _data[i][-1]
        _ans += math.log2(1/(__sigmoid_func(_yi * np.dot(_w, np.transpose(_xi)))))
    return _ans / _N


def __add_x_zero(_data:List) -> List:
    _new_data = []
    for row in _data:
        _temp = [1]+row
        _new_data.append(_temp)
    return _new_data


def __gradient_descent(_data: List[List[float]], _w: List[float]) -> List[float]:
    _N = len(_data)
    _ans = [0] * 4
    for i in range(_N):
        _xi = _data[i][0:4]
        _yi = _data[i][-1]
        _temp = 1/(1 + math.exp(_yi * np.dot(_w, np.transpose(_xi)))) * _yi
        for j in range(len(_ans)):
            _ans[j] += _temp * _xi[j]

    for i in range(len(_ans)):
        _ans[i] /= (0-_N)
    return _ans


def __update_w(_w: List[float], _alpha: float, _data: List[List[float]]) -> List[float]:
    _temp = __gradient_descent(_data, _w)
    for i in range(len(_w)):
        _w[i] -= _alpha * _temp[i]
    return _w


def __logic_regression():
    _data = __read_file("classification.txt")
    _new_data = __add_x_zero(_data)
    _w = __random_w(3)
    _last_ein = __cal_ein(_w, _new_data)
    for i in range(2000):
        _w = __update_w(_w, 0.0002, _new_data)
        _Ein = __cal_ein(_w, _new_data)
        _last_ein = _Ein

    _mis_class = 0
    for i in range(len(_new_data)):
        _score = __sigmoid_func(_new_data[i][-1] * (np.dot(_w, np.transpose(_new_data[i][0:4]))))
        if _score < 0.5:
            _mis_class += 1

    print("Ein:", _Ein, "  W:", _w[1:], "accuracy:", (1 - _mis_class/len(_new_data))*100, "%")
    #__plot(_w[1:], _data)


if __name__ == "__main__":
    __logic_regression()