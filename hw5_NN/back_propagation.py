from typing import List
import math
import random
import numpy as np
from PIL import Image
import copy


def __read_file(file:str) -> List[str]:
    try:
        f1 = open(file,'r')
    except:
        print(file,"does not exist\n")
        quit()
    _content = []
    _label = []
    while True:
        _line = f1.readline().strip()
        if not _line:
            break
        if "down" in _line:
            _label.append(1)
        else:
            _label.append(0)
        _content.append(_line)
    f1.close()
    return _content, _label


def __imageToMatrix(filename):
    """
    convert the input image to a normalized matrix
    :param filename:
    :return: normalized matrix
    """
    img = Image.open(filename)
    width, height = img.size
    image = img.convert("L")
    matrix = image.getdata()
    matrix = np.array(matrix, dtype='float') / 255.0
    new_data = np.reshape(matrix, (height, width))
    return new_data


def __sigmoid_function(s):
    if isinstance(s, list):
        _ans=[]
        for i in s:
            _ans.append(__sigmoid_function(i))
        return _ans

    if s < 0:
        return 1 - 1 / (1 + math.exp(s))
    else:
        return 1 / (1 + math.exp(-s))


def __initialize_w(upper_bound:float, lower_bound:float, dimensions:List[int]) -> List:
    if upper_bound < lower_bound:
        upper_bound, lower_bound = lower_bound, upper_bound
    _w = []
    for i in range(len(dimensions)-1):
        _row = dimensions[i]
        _col = dimensions[i+1]
        _temp = [[random.uniform(lower_bound, upper_bound) for j in range(_col)] for k in range(_row)]
        _w.append(_temp)
    return _w


def __error_func(first: float, second: float) -> float:
    return pow(first - second, 2)


def __compute_x(input: List[float], d_layers: List[int], w: List[List[float]]) -> List[List[float]]:
    _x_this_img = copy.deepcopy(input)
    for L in range(len(d_layers) - 1):
        _x_last_layer = _x_this_img[-1]
        _w_ij = w[L]
        _x_this_layer = np.dot(np.transpose(_w_ij), np.transpose(_x_last_layer))
        _x_this_img.append(__sigmoid_function(_x_this_layer.tolist()))

    return _x_this_img


def __compute_six(_x_this_image:List[List[float]], _w:List[List[float]], d_layers: List[int],_y_this_img:int) -> List[float]:
    _transposed_w = copy.deepcopy(_w)
    _six_this_image = []
    for L in range(len(d_layers)-1, -1, -1):
        _xi_l = _x_this_image[L]
        if L == len(d_layers) - 1:   #base case
            _xil = _xi_l[0]
            _temp =  2 * (_xil - _y_this_img) * (1 - _xil) * _xil
            _six_this_image.append([_temp])
            continue
        _six_this_layer = np.dot( _transposed_w[L], np.transpose(_six_this_image[-1]))
        _six_this_layer = np.transpose(_six_this_layer).tolist()
        for i in range(len(_six_this_layer)):
            _six_this_layer[i] *= (1 - _xi_l[i]) * _xi_l[i]
        _six_this_image.append(_six_this_layer)
    _six_this_image.reverse()
    return _six_this_image


def __update_w(_w:List[List[float]], learn_rate:float, _x:List[List[float]], _six: List[List[float]]):
    _new_w = copy.deepcopy(_w)
    for i in range(len(_w)):
        a = np.array(_x[i])
        b = np.array(_six[i+1])
        _temp = np.dot(a[:, None], b[None, :])
        _temp = np.array(_temp) * learn_rate
        _new_w[i] = (np.array(_new_w[i])- _temp)
    return _new_w


def __get_input(_train_images: List[str], _label: List[int]) -> List:
    _input_list = []
    for i in range(len(_train_images)):
        image = _train_images[i]
        label = _label[i]
        matrix = __imageToMatrix(image)
        _input = matrix.flatten().tolist()
        _input_list.append([_input])
    return _input_list


def __err_monitor(a , b):
    _sum = 0
    for i in range(len(a)):
        _sum += (a[i] - b[i])**2
    return _sum


def __test(test_list, _d_layers, _label, _x_list, _w):
    _test_images, _test_label = __read_file(test_list)
    _input_list = __get_input(_test_images, _test_label)
    _run_x_result = []
    for input in _input_list:
        test_x = __compute_x(input, _d_layers, _w)
        _run_x_result.append(test_x[-1][0])
    try:
        _run_result = [int(x+0.5) for x in _run_x_result]
    except:
        _run_result = [x for x in _run_x_result]
    _correct_ones = 0
    for i in range(len(_test_label)):
        if _test_label[i] == _run_result[i]:
            _correct_ones+=1
    print('test:', "accuracy:",round(_correct_ones/len(_test_label) * 100, 2),"%")
    print("True",_test_label)
    print("Prediction",_run_result)
    #print(_run_x_result)
    return


def __back_propagation(train_list:str, test_list:str, hidden_layer:List[int], no_of_output:int, epochs:int, learn_rate):
    _train_images, _label = __read_file(train_list)
    _temp = (__imageToMatrix(_train_images[0]))
    _row = len(_temp)
    _col = len(_temp[0])
    _d_layers = [len(_temp)*len(_temp[0])]+hidden_layer+[no_of_output]
    _w = __initialize_w(0.01, -0.01, _d_layers)
    _input_list = __get_input(_train_images, _label)
    for i in range(epochs):
        _x_list=[]
        a, b = random.randint(0,len(_train_images)), random.randint(0,len(_train_images))
        upper, lower = max(a,b), min(a,b)
        #print(upper,lower)
        for img  in range(len(_train_images) - 1):
            if img >= lower and img <= upper:
                _x =  __compute_x(_input_list[img], _d_layers, _w)
                _x_list.append(_x[-1][0])
                _six = __compute_six(_x, _w, _d_layers, _label[img])
                _w = __update_w(_w,learn_rate,_x, _six)
            else:
                continue
        if i % 100 == 0:
            print("training:", i, "/", epochs)
            err = (__err_monitor(_x_list, _label))
            print("loss:",err,"\n")
    __test(test_list, _d_layers, _label, _x_list, _w)


if __name__ == "__main__":
    __back_propagation("downgesture_train.list", "downgesture_test.list", hidden_layer=[100], no_of_output=1, epochs=1000, learn_rate=0.1)
