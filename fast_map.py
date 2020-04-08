import numpy as np
from typing import List
import random
import copy
import math
from fast_map_plot import __plot


def __read_file(file_name, _type) -> List[int]:
    #read file and return data
    try:
        file1 = open(file_name,'r')
    except:
        print(file_name,'does not exist\n')
        quit()
    _ans = []
    while True:
        _line = file1.readline()
        if not _line:
            break
        if _type.lower().strip() == 'data':
            _line = _line.split()
            _line = [int(x) for x in _line]
        else:
            _line = _line.strip()
        _ans.append(_line)
    file1.close()
    return _ans


def __generate_dis_dic(_data: List[List[int]]) -> dict:
    #put all distances corresponding to points in a dictionary for quick access
    _my_dic = {}
    for i in _data:
        if i[0] in _my_dic:
            _my_dic[i[0]][i[1]] = i[2]
        else:
            _my_dic[i[0]] = {i[1]:i[2]}

    return _my_dic


def __get_farest_pair(_dist_dic: dict, _coordinates: List[List[int]]) -> List[int]:
    #get two objects which have the greatest distance
    _temp = random.randint(1,10)
    _ans = []
    _last_max = 0
    while True:
        _max = 0
        _max_id = 0
        for i in range(1,11):
            if i == _temp:
                continue
            _dist = __cal_distance(_dist_dic, _coordinates, _temp,i)
            if _dist > _max:
                _max = _dist
                _max_id = i
        if _max == _last_max:
            _ans = [_max_id,_temp]
            _ans.sort()
            break
        _last_max = _max
        _temp = _max_id
    print(_ans)
    return _ans, _max


def __generate_coordinates(_base_pair:List[int],_dist_dic:dict, _base_dist:int, _coordinates: List[List[int]]) -> List[int]:
    #generate coordinates for single dimension
    _list = []
    a = _base_pair[0]
    b = _base_pair[1]
    for i in range(1,11):
        if i == _base_pair[0]:
            _list.append(0)
        elif i == _base_pair[1]:
            _list.append(_base_dist)
        else:
            _coordinate = pow(__cal_distance(_dist_dic, _coordinates, a, i), 2) + pow(_base_dist, 2) -\
                          pow(__cal_distance(_dist_dic, _coordinates, b, i), 2)
            _coordinate /= (2*_base_dist)
            _list.append(round(_coordinate,2))
    return _list


def __cal_distance(_dist_dic: dict,_coordinates: List[List[int]],object_a: int, object_b: int) -> float:
    #distance function
    _pair = [object_a, object_b]
    _pair.sort()
    _base = pow(_dist_dic[_pair[0]][_pair[1]],2)
    for i in range(len(_coordinates)):
        _base -= pow((_coordinates[i][_pair[0]-1] - _coordinates[i][_pair[1]-1]), 2)

    return math.sqrt(_base)

def __fast_map(k: int) -> List[List[float]]:
    _coordinates = []
    _data = __read_file("fastmap-data.txt", "data")
    _wordlist = __read_file("fastmap-wordlist.txt", "word")
    _wordlist.reverse()
    _wordlist.append('')
    _wordlist.reverse()
    _dist_dic = __generate_dis_dic(_data)

    for i in range(k):
        _pair, _max = __get_farest_pair(_dist_dic, _coordinates)
        _xi = __generate_coordinates(_pair, _dist_dic, _max, _coordinates)
        _coordinates.append(_xi)

    return np.transpose(_coordinates), _wordlist


if __name__ == "__main__":
    k = 2
    _coordinates, _headers = __fast_map(k)
    #print(_coordinates)
    __plot(_coordinates, _headers)