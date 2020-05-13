#main file of inf 552 final project
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from prepare_data_for_tree import __prepare_train_data_for_tree, __prepare_test_data_for_tree
from prepare_data_for_LR import __prepare_data_for_linear_regression, __prepare_test_data_for_LR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings


def read_data(_file):
    data = pd.read_csv(_file, sep=' ')
    return data


def __plot_distribution(_y,col):
    """
    plot distribution for the input dataframe column for data analysis purpose

    """
    _path = "screenshots/others/"+col + ".png"
    plt.hist(_y, bins=70, histtype='bar' , alpha=0.75)
    plt.title("distribution of " + col)
    plt.savefig(_path)
    plt.close()
    return


def __random_forest(_train_data_for_tree):
    print("training: random forest")
    _x = _train_data_for_tree
    _x = _x.dropna().replace("-", 0).reset_index(drop=True)
    _y = _x['price']
    _features = []
    for i in _train_data_for_tree.columns:
        if i == "price":
            continue
        _features.append(i)
    _x = _x[_features]
    model = RandomForestRegressor()
    scores = cross_val_score(model, X=_x, y=_y, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    model = RandomForestRegressor()
    model.fit(_x,_y)
    return np.mean(scores), model


def __neural_network(data):
    print("training: neural network")
    _features = []
    for i in data.columns:
        if i == 'price':
            continue
        _features.append(i)
    _y = data["price"]
    _x = data[_features]
    model = MLPRegressor(solver='lbfgs',max_iter=100)
    
    scores = cross_val_score(model, X=_x, y=_y, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    model = MLPRegressor(solver='lbfgs',max_iter=100)
    model.fit(_x,_y)
    return np.mean(scores), model
    

def __Decision_tree(data):
    print("training: decision tree")
    _features = []
    for i in data.columns:
        if i == 'price':
            continue
        _features.append(i)
    data = data.dropna().replace("-", 0).reset_index(drop=True)
    _x = data[_features]
    _y = data["price"]
    model = DecisionTreeRegressor()
    scores = cross_val_score(model, X=_x, y=_y, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    model = DecisionTreeRegressor()
    model.fit(_x,_y)
    return np.mean(scores),model
    

def __take_log(col):
    """

    take log for the input dataframe column
    """
    return np.log(col+1)


def __Linear_regression(data):
    print("training: linear regression")
    _features = []
    for i in data.columns:
        if i == 'price':
            continue
        _features.append(i)
    data = data.dropna().replace("-", 0).reset_index(drop=True)
    _y = data["price"]
    _x = data[_features]
    model = LinearRegression()
    scores = cross_val_score(model, X=_x, y=_y, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    model = LinearRegression()
    model.fit(_x,_y)
    return np.mean(scores), model


def __train_models():
    """
    train models and do the prediction
    return : dic that includes the prediction results

    """
    warnings.filterwarnings('ignore')
    _result_dic = {}
    _model_dic = {}
    _train_data = read_data("used_car_train_20200313.csv")  #train_data, dataframe
    _train_data["fuelType"].replace(np.nan, 0, inplace=True )
    #print(_train_data['fuelType'].isnull().sum())
    _train_data["price"] = __take_log(_train_data["price"])
    
    _new_features = __prepare_train_data_for_tree(_train_data)
    
    _test_data = read_data("used_car_testA_20200313.csv")
    _test_data_for_tree = __prepare_test_data_for_tree(_test_data,_new_features)
    _test_data_for_LR = __prepare_test_data_for_LR(_test_data_for_tree)

    _train_data_for_tree = pd.read_csv("train_data_for_tree.csv")
    __prepare_data_for_linear_regression(_train_data_for_tree)
    del _train_data
    _train_data_for_LR = pd.read_csv("train_data_for_LR.csv")

    #so far, all dataset been ready
    
    _result_dic['random_forest'],_model_dic['random_forest'] = __random_forest(_train_data_for_tree)
    _result_dic['decision_tree'] , _model_dic['decision_tree'] = __Decision_tree(_train_data_for_tree)
    _result_dic['linear_regression'],_model_dic['linear_regression'] = __Linear_regression(_train_data_for_LR)
    _result_dic['neural_network'],_model_dic['neural_network'] = __neural_network(_train_data_for_LR)
    _max = np.max(_train_data_for_tree['price'])
    _min = np.min(_train_data_for_tree['price'])
    _result_dic['neural_network'] *= (_max-_min)
    return _result_dic, _model_dic, (_max,_min), (_test_data_for_tree, _test_data_for_LR)


def __predict(_result_dic, _model_dic, _pair0, _pair1):
    _predict_dic = {}
    _test_data_for_tree = _pair1[0]
    _test_data_for_LR = _pair1[1]
    for key in _result_dic.keys():
        if key in ['decision_tree','random_forest']:
            _temp = _model_dic[key].predict(_test_data_for_tree)
            print(key,"scores:",_result_dic[key],"  predict result:","see predict_"+str(key)+".csv")
            _temp = list(_temp)
            for i in range(len(_temp)):
                _temp[i] = pow(np.e, _temp[i]) - 1
            _predict_dic[key] = _temp
            #print(_temp[0])

        else:
            _temp = _model_dic[key].predict(_test_data_for_LR)
            print(key,"scores:",_result_dic[key],"  predict result:","see predict_"+str(key)+".csv")
            
            _max = _pair0[0]
            _min = _pair0[1]
            for i in range(len(_temp)):
                _temp[i] = pow(np.e, (_temp[i]*(_max - _min) + _min)) - 1 
                _predict_dic[key] = _temp
            
    if os.path.exists("train_data_for_tree.csv"):
        os.remove("train_data_for_tree.csv")       #delete temp files
    if os.path.exists("train_data_for_LR.csv"):
        os.remove("train_data_for_LR.csv")
    return _predict_dic



def __run_me():
    _path = os.getcwd()
    for i in os.listdir(_path):
        if i.startswith("predict"):
            os.remove(i)
            
    _result_dic, _model_dic, _pair0, _pair1 =  __train_models()
    _predict_dic = __predict(_result_dic, _model_dic, _pair0, _pair1)
    del _predict_dic['linear_regression']
    del _result_dic['linear_regression']
    _sum = 0
    for key in _result_dic.keys():
        _sum += (1 - _result_dic[key])
       
    _temp = np.array(_predict_dic['decision_tree'])*(1 - _result_dic['decision_tree'])
    for key in ['random_forest','neural_network']:
        _temp += np.array(_predict_dic[key])*(1 - _result_dic[key])
    _temp /= _sum
    _temp = pd.DataFrame(data=_temp, columns = ['price'])
    _temp.to_csv("predict_result.csv")
    for i in os.listdir(_path):
        if i.endswith("~"):   #remove temp files
            os.remove(i)


if __name__ == "__main__":
    __run_me()
    
    
