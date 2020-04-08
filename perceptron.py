from typing import List
import random
import numpy as np
from perceptron_plot import __plot


def __read_file(file_name: str) -> List:
    f1 = open(file_name, 'r')
    _list = []
    while True:
        _line = f1.readline()
        if not _line:
            break
        _line = _line.split(",")
        _line = [float(x) for x in _line[:-1]]
        _list.append(_line)

    return _list


def __random_w(_dimensions: int) -> List:
    _list = []
    _list.append(-0.001)
    for i in range(_dimensions):
        _list.append(random.random())
    return _list


def __get_violated_constraints(_data:List[List[float]],_w:List[float]) -> bool:
    _list = []
    for i in range(len(_data)):
        _y = _data[i][-1]
        if _y == 1:
            if np.dot(_w, np.transpose(_data[i][:-1])) < 0:
                _list.append(i)
        elif _y == -1:
            if np.dot(_w, np.transpose(_data[i][:-1])) >= 0:
                _list.append(i)

    return _list


def __update_w(_w: List[float], _xi: List[float], _alpha: float, _yi: int) -> List[float]:
    for i in range(len(_w)):
        _w[i] = _w[i] + _xi[i]*_alpha*_yi

    return _w


def __add_x_zero(_data:List) -> List:
    _new_data = []
    for row in _data:
        _temp = [1]+row
        _new_data.append(_temp)
    return _new_data


def __perceptron():
    _data = __read_file("classification.txt")
    _new_data = __add_x_zero(_data)
    _w = __random_w(3)
    _alpha = 0
    i = 0
    while True:
        _violated_index = __get_violated_constraints(_new_data, _w)
        if len(_violated_index) == 0:
            break
        i += 1

        if len(_violated_index) >= 900:
            _alpha = 0.08
        elif len(_violated_index) >= 500:
            _alpha = 0.05
        elif len(_violated_index) >= 100:
            _alpha = 0.01
        elif len(_violated_index) >= 30:
            _alpha = 0.005
        else:
            _alpha = 0.0001
        _data_i = _new_data[_violated_index.pop()]
        y_i = _data_i[-1]
        _w = __update_w(_w, _data_i[:-1], _alpha, y_i)
    print("iteration:", i, "accuracy:",1-len(_violated_index)/len(_data))
    __plot(_w[1:], _data)


if __name__ == "__main__":
    __perceptron()