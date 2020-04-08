import numpy as np
import random
import copy
import math
from liner_algerba_funcs import __scalar_product, __vector_product, __add_two_matrix
from visualization import __plot,__plot_gaussian

def __read_file() -> list:
    """
    read data file
    :return: the data matrix in a 2 dimension list
    """
    try:
        _file = open("clusters.txt")
    except:
        print("file does not exist\n")
        return
    _matrix = []
    while True:
        _line = _file.readline()
        if not _line:
            break
        _line = _line.split(",")
        for i in range(len(_line)):
            _line[i] = float(_line[i])
        _matrix.append(_line)

    _file.close()
    return _matrix, len(_matrix), len(_matrix[0])


def __initialize_ric(N:int, k:int) -> list:
    """
    Create a random normalized gamma matrix
    :param N: the number of data points
    :param k: the number of gaussians(clusters) which is 3 in this homework
    :return:  The created randomly gamma matrix in a 2 dimension list
    """
    _matrix=[[0 for i in range(k)] for j in range(N)]
    for i in range(len(_matrix)):
        _row = _matrix[i]
        _sum = 0
        for i in range(k):
            _row[i] = random.random()
            _sum += _row[i]
        for i in range(k):
            _row[i] = round(_row[i]/_sum,3)

    for i in range(len(_matrix)):
        _matrix[i][0] += (1.0 - sum(_matrix[i]))  # gurantee that the sum of each row equals to 1
    return _matrix

def __get_pi(ric_matrix: list) -> list:
    """
    calculate pi of gaussians
    :param ric_matrix: The gamma matrix
    :return: Pi for each gaussian which stores in a list
    """
    _N = len(ric_matrix)
    _K = len(ric_matrix[0])
    _pi_list=[]
    for c in range(_K):
        _temp = 0
        for i in range(_N):
            _temp += ric_matrix[i][c]
        _temp /= _N
        _pi_list.append(_temp)
    return _pi_list


def __get_mean(ric_matrix:list,data_matrix:list) -> list:
    """
    Calculate the mean vector
    :param ric_matrix: The gamma matrix
    :param data_matrix: The data matrix
    :return: The mean vectors which stored as a 2 dimension list
    """
    data_matrix = copy.deepcopy(data_matrix)
    _K = len(ric_matrix[0])
    _N = len(ric_matrix)
    _d = len(data_matrix[0])
    _mean=[]
    for c in range(_K):
        _sum_ric = 0
        _sum_ric_x = [0] * _d
        for i in range(_N):
            _sum_ric += ric_matrix[i][c]
            for d in range(_d):
                _sum_ric_x[d] += data_matrix[i][d] * ric_matrix[i][c]

        for d in range(_d):
            _sum_ric_x[d] /= _sum_ric

        _mean.append(_sum_ric_x)

    return _mean


def __get_sub_covariance_matrix(ric_matrix:list, data_matrix:list, mean_vector:list,c:int)->list:
    """
    Calculate the covariance matrix of a specific gaussian (cluster)
    :param ric_matrix: gamma matrix
    :param data_matrix: data matrix
    :param mean_vector: the mean vector
    :param c: the index of the specific cluster (gaussian)
    :return: The co-variance matrix for this specific cluster
    """
    _N = len(ric_matrix)
    _d = len(data_matrix[0])
    _covar_matrix = [[0 for i in range(_d)] for j in range(_d)]
    _sum_ric = 0
    for i in range(_N):
        _sum_ric += ric_matrix[i][c]
        _xi = data_matrix[i]
        _vector = [_xi[m]- mean_vector[m] for m in range(_d)]
        _covar_matrix = __add_two_matrix(_covar_matrix, __vector_product(ric_matrix[i][c],_vector,_vector))

    for i in range(_d):
        for j in range(_d):
            _covar_matrix[i][j] /= _sum_ric

    return _covar_matrix


def __get_covariance_list(ric_matrix,data_matrix,mean_matrix) -> list:
    """
    Calculate covariance matrix for each cluster
    :return: covariance matrix for each cluster stores in a multi-dimension list
    """
    if not len(ric_matrix) == len(data_matrix):
        print("dimensions does not match")
        return []
    _K = len(ric_matrix[0])
    data_matrix=copy.deepcopy(data_matrix)
    _covariance_list=[]

    for c in range(_K):
        _mean_vector = mean_matrix[c]
        _covariance_matrix = __get_sub_covariance_matrix(ric_matrix,data_matrix,_mean_vector,c)
        _covariance_list.append(_covariance_matrix)

    return _covariance_list


def __check_if_converge(last_log:float, new_log:float, tolerance) -> bool:
    if abs(new_log - last_log) <= tolerance:
        return True
    return False


def __cal_single_ric(pi_list:float, xi:list,mean_list:list, covar_matrix_list:list, _c:int) -> float:
    """
    calculate single ric value
    :param pi_c:
    :param xi:
    :param mean_vector:
    :param covar_matrix:
    :return: single ric value
    """
    _K = len(pi_list)
    _sum = 0
    for c in range(_K):
        _sum += pi_list[c]*__cal_normal_distribution(xi,mean_list[c],covar_matrix_list[c])

    _ans = pi_list[_c]*__cal_normal_distribution(xi,mean_list[_c],covar_matrix_list[_c])

    return _ans/_sum


def __normal_distribution_helper_function(xi:list, mean_vector:list, covar_matrix) -> float:
    """
    help function that deals with
    :param xi:
    :param mean_vector:
    :param covar_matrix:
    :return: a number
    """
    _d = len(xi)
    vector1 = [xi[i] - mean_vector[i] for i in range(_d)]
    _temp_vector = [0 for i in range(_d)]
    covar_matrix = copy.deepcopy(covar_matrix)
    covar_matrix = np.linalg.inv(covar_matrix)
    for d in range(_d):
        _temp_vector[d] = __scalar_product(vector1,[a[d] for a in covar_matrix])
    ############################################
    return __scalar_product(_temp_vector,vector1)


def __cal_normal_distribution(xi:list,mean_vector:list,covar_matrix:list) -> float:
    """
    calculate N(xxx)
    :param xi:
    :param mean_vector:
    :param covar_matrix:
    :return:probability, a number
    """
    _d = len(xi)
    _temp = __normal_distribution_helper_function(xi, mean_vector, covar_matrix)
    _ans = pow(math.pi*2,(0 - _d)/2) * pow(np.linalg.det(covar_matrix), -0.5) * math.exp(-0.5*_temp)

    return _ans


def __cal_ric_matrix(data_matrix, pi_list, mean_list, covariance) -> list:
    """
    re-calculate gamma matrix (ric)
    :param data_matrix:
    :param pi_list: a list which contains pi for each cluster
    :param mean_list: a list which contains mean vector for each cluster
    :param covariance: a list which contains covariance matrix for each cluster
    :return: The new calculated gamma matrix
    """
    _N = len(data_matrix)
    _K = len(pi_list)
    _new_ric_matrix=[[0 for i in range(_K)]for j in range(_N)]
    for i in range(_N):
        for c in range(_K):
            _new_ric_matrix[i][c] = __cal_single_ric(pi_list,data_matrix[i],mean_list,covariance,c)

    return _new_ric_matrix


def __get_log_like_hood(_data_matrix, _mean_list,_pi_list,_covar_matrix_list ) -> float:

    _outer_sum = 0
    for i in range(len(_data_matrix)):
        _xi = _data_matrix[i]
        _inner_sum=0
        for c in range(len(_pi_list)):
            _inner_sum += _pi_list[c] * __cal_normal_distribution(_xi,_mean_list[c],_covar_matrix_list[c])

        _outer_sum += math.log(_inner_sum, math.e)

    return _outer_sum


def __gmm(_K:int, _tol:float):
    """
    The main function which generates the gmm model
    :param _K: The number of gaussians(clusters) which is 3 for this homework
    :param _tol: The tolerance of convergence
    :return:
    """
    _data_matrix, _N, _d = __read_file()        # this matrix is n x d for easier calculate
    _ric_matrix = __initialize_ric(_N, _K)      # randomly choose
    _last_log = 0

    while True:
        _pi_list = __get_pi(_ric_matrix)
        _mean_matrix = __get_mean(_ric_matrix,_data_matrix)
        _covariance_matrix_list = __get_covariance_list(_ric_matrix,_data_matrix,_mean_matrix)
        _ric_matrix = __cal_ric_matrix(_data_matrix,_pi_list,_mean_matrix,_covariance_matrix_list)
        _new_log = __get_log_like_hood(_data_matrix, _mean_matrix,_pi_list,_covariance_matrix_list)

        if __check_if_converge(_last_log, _new_log,_tol):
            break
        else:
            _last_log = _new_log
        #print(_new_log)
    return _pi_list,_mean_matrix,_covariance_matrix_list,_ric_matrix, _data_matrix

if __name__ == "__main__":
    _pi_list,_mean_matrix,_covariance_matrix_list,_ric_matrix,_data_matrix = __gmm(3,math.exp(-5))
    __plot_gaussian(_mean_matrix,_covariance_matrix_list,_data_matrix,_pi_list)
    __plot(_data_matrix,_ric_matrix)