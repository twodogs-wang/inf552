from math import log
import numpy
from plot import createPlot


def __read_data(data_file:str)->list:
    """
    :param data_file: txt file which stores training data
    :return: the training data set stores in list and labels in a list
    """
    _file1=open(data_file,"r")
    _lines=_file1.readlines()
    _ans=[]
    _labels=[]
    for _line in _lines:
        if not _line:
            continue
        elif "(" in _line:
            _labels=_line.split(",")
            _labels[0]=_labels[0][1:]
            _labels[-1]=_labels[-1][:-2]  #store labels
            continue
        else:
            _temp=_line.split()[1:]
            if len(_temp)==0:
                continue
            _list=[]
            for item in _temp:
                _str=""         #remove non-alpha symbols
                for i in item:
                    if i.isalpha():
                        _str+=i
                _list.append(_str)
            _ans.append(_list)
    for i in range(len(_labels)):
        _labels[i]=_labels[i].strip() #remove redundant white spaces

    return _ans,_labels


def __calclulate_entropy(_data_set:list)->float:
    """
    calculate entropy based on the last column of the input dataset
    :param _data_set:
    :return: the entropy
    """
    if not _data_set or len(_data_set)==0:
        return 0

    _num=len(_data_set)#total cases
    _temp_dic={}
    _entropy=0

    for row in _data_set:
        _temp=row[-1]
        if _temp in _temp_dic:  #stores unique values and frequencies of each(last column)
            _temp_dic[_temp]+=1
        else:
            _temp_dic[_temp]=1

    for key in list(_temp_dic.keys()):
        _probability = float(_temp_dic[key])/_num
        _entropy -= _probability * log(_probability,2)

    return _entropy

def __get_decisions(_data_set,_index):
    """
    This function checks decisions (e.g. yes , no ,high ,low) of input attribute index
    :param _data_set:
    :param _index: the index of target attribute
    :return: a dict which stores each unique decision and its frequency, for future calculation
    """
    if not _data_set or _index >= len(_data_set[0]):
        return {}

    _ans={}
    for row in _data_set:
        if row[_index] in _ans:
            _ans[row[_index]]+=1
        else:
            _ans[row[_index]]=1

    return _ans

def __split_data_set(_data_set,_index,_decision):
    """
    split dataset into sub-datasets based on input attribute index and its decision
    (e.g. VIP? NO)
    :param _data_set: upper level of dataset
    :param _index: the index of attribute
    :param _decision: decision
    :return: sub-dataset
    """
    _sub_data_set=[]
    for row in _data_set:
        if row[_index]==_decision:
            _sub_data_set.append(row[:_index]+row[_index+1:]) #creating sub-dataset
    return _sub_data_set

def __major_cases(_data_set):
    """
    this function calculates the majority classify of the last column of dataset,
    this function will be called when corner cases occured:
    1. no attributes anymore but last column still contain values
    2. all the rest attributes have the same decisions combinations but the last column labelsa are different
    :param _data_set:
    :return: the majority label
    """
    _index={}
    _freq=[]
    for row in _data_set:
        if row[-1] in _index:
            _freq[_index[row[-1]]][1]+=1
        else:
            _index.update({row[-1]: len(_freq)})
            _freq.append([row[-1],1])
    _freq=sorted(_freq,key= lambda x:x[1],reverse=True)

    return _freq[0][0]



def __get_best_attribute(_data_set):
    """
    get the best attribute which generates largest entropy gain
    :param _data_set:
    :return: the best attribute index
    """
    _Base_entropy = __calclulate_entropy(_data_set)
    _entropy_gain = 0
    _best_attribute_index = 0

    for i in range(len(_data_set[0])-1):  #loop over each attribute
        _decisions_dic = __get_decisions(_data_set, i)  #decisons for current attribute level
        _new_entropy = 0
        _num=len(_data_set)
        for _decision in _decisions_dic.keys():
            _sub_data_set = __split_data_set(_data_set, i, _decision)
            _temp_entropy = __calclulate_entropy(_sub_data_set)
            _new_entropy += _temp_entropy * (_decisions_dic[_decision] / _num)

        if _Base_entropy - _new_entropy > _entropy_gain:
            _entropy_gain = _Base_entropy - _new_entropy #find biggest entropy gain
            _best_attribute_index = i

    return _best_attribute_index, _entropy_gain == 0

def __check_if_same_attributes(_data_set)->bool:
    """
    this function checks if the corner case exists:
    all attributes values are same, if so, there is no need to do further analysis
    :param _data_set:
    :return: True if attributes are same
    """
    if len(_data_set[0])<2 or not _data_set:
        return False
    _temp=[]

    for row in _data_set:
        _temp.append(row[:-1])

    _temp=numpy.transpose(_temp) #transpose the "matrix" to check easier
    for row in _temp:
        if not len(set(row))==1:
            return False

    return True


def __generate_decision_tree(_data_set:list,_attributes:list)->dict:
    """
    main function that generates decision-tree in recursion way
    :param _data_set, _attributes:labels extracted from the data file
    :return the decsion tree
    """
    _my_tree={}
    _num=len(_data_set)
    if not _data_set:
        return _my_tree

    _final_label=[row[-1] for row in _data_set] #labbel of last column
    if _final_label.count(_final_label[0])==len(_final_label):#the last column contains multiple same values,terminate
        return _final_label[0]

    if len(_data_set[0])==1:     #only one possilbel final label,terminate
        return __major_cases(_data_set)

    _best_attribute_index, _zero_flag=__get_best_attribute(_data_set) #find best attribute with max entropy gain in this layer

    if _zero_flag:
        return __major_cases(_data_set)

    _best_attribute=_attributes[_best_attribute_index]
    _my_tree.update({_best_attribute:{}})   #create tree node for this layer
    _sub_tree=_my_tree[_best_attribute]
    _decisions_dic = __get_decisions(_data_set,_best_attribute_index) #get decisions(possilbel branches) for this node
    _attributes.remove(_best_attribute) #remove attribute from the original list


    for _decision in _decisions_dic.keys():

        _sub_data_set=__split_data_set(_data_set,_best_attribute_index,_decision)

        _sub_tree.update({_decision:__generate_decision_tree(_sub_data_set,_attributes[:])}) #update tree with recursion

    return _my_tree



def __predict(_my_tree, _labels, _test_data):
    _root_node = list(_my_tree.keys())[0]  # root node
    _temp_dict = _my_tree[_root_node]
    _index = _labels.index(_root_node)  #

    for key in _temp_dict.keys():  # loop over each branch
        if _test_data[_index] == key:
            if isinstance(_temp_dict[key],dict):  # not leaf, do resursion
                return __predict(_temp_dict[key],_labels, _test_data)
            else:  # return if leaf
                return _temp_dict[key]


if __name__ == "__main__":
    _data_set, _labels = __read_data("dt_data.txt")
    _best_labels=[]
    _my_tree=__generate_decision_tree(_data_set,_labels[:-1])
    print("The Decision Tree:")
    print(_my_tree)
    _test_data=['Moderate','Cheap','Loud','CityCenter','No','No']
    print("\nPrediction:",__predict(_my_tree,_labels,_test_data))
