from typing import List
from helper import Node
import os

def __read_grid()->List[List[int]]:
    """
    get coordinates of free cells
    """
    _file = "hmm-data.txt"
    f1 = open(_file, "r")
    _ans = []
    while True:
        _line = f1.readline()
        if _line.strip() == "Grid-World:":
            continue
        elif _line.strip() == "":
            continue
        elif _line.strip() == "Tower Locations:":
            break
        else:
            _temp = _line.strip().split(" ")
            _temp = [int(x) for x in _temp]
            _ans.append(_temp)
    return _ans


def __get_free_cells(grid: List[List[int]])->List[tuple]:
    """
    get free cells distances
    """
    _ans = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                continue
            _ans.append((i,j))
    return _ans


def __get_noisy_distances()->List[List[float]]:
    """
    get noist distances matrix
    """
    _file = "hmm-data.txt"
    f1 = open(_file, "r")
    _ans = []
    while "Noisy" not in f1.readline():
        continue
    found = False
    while True:
        _line = f1.readline()
        if _line.strip() == "":
            if found:
                break
            continue
        found = True
        _line = _line.split()
        _line = [float(x) for x in _line]
        _ans.append(_line)
    return _ans


def __get_tower_dists(free_cell:List[tuple], tower_loc: List[tuple])->List[List[float]]:
    _ans = []
    for _ in range(len(free_cell)):
        _point = free_cell[_]
        _temp = []
        for j in range(len(tower_loc)):
            _dist = pow((tower_loc[j][0] - _point[0])**2 + (tower_loc[j][1] - _point[1])**2, 0.5)
            _temp.append(_dist)
        _ans.append(_temp)
    return _ans


def __get_possible_cells_per_step(noisy_dists: List[tuple], free_cells: List[tuple], to_tower_dists: List[List[float]])->dict:
    """
    input:
    noisy_dists: noisy distances matrix
    free_cells: coordinates of free cells
    to_tower_dists: ditances to 4 towers of each free cells
    output:
    (coordinates)possible cells for each time step

    """
    _ans = {}
    for _ in range(len(noisy_dists)):#loop over each time step
        _ans[_] = []
        _four_noisy_dists = noisy_dists[_]
        for i in range(len(free_cells)):#loop over each free cell
            _four_dists = to_tower_dists[i]
            counter = 0
            for j in range(len(_four_noisy_dists)):  #loop over each tower
                true_dist = _four_dists[j]
                noisy_dist = _four_noisy_dists[j]
                if round(true_dist*0.7,1) <= noisy_dist <= round(true_dist*1.3,1):
                    counter += 1
            if counter == len(_four_noisy_dists):
                _ans[_].append(free_cells[i])
    return _ans


def __get_possible_steps_per_cell(_dict: dict,free_cells:List[tuple]):
    """
    input:
    _dict: dict which stores possible cells for each time step
    free_cells: coordinates for all free cells
    
    output:
    possible states for each cells in a dict, for future quick access
    """
    _ans = {}
    for cell in free_cells:
        _ans[cell] = []
        for step in list(_dict.keys()):
            if cell in _dict[step]:
                _ans[cell].append(step)

    for key in list(_ans.keys()):
        if _ans[key] == []:
            del _ans[key]
    return _ans
        

def __get_neighbours(free_cells: List[tuple],grid:List[List[int]])->dict:
    """
    get neighbours for each free cell
    """
    _ans = {}
    for i in range(len(free_cells)):
        point = free_cells[i]
        _ans[point] = []
        x = point[0]
        y = point[1]
        if x+1<len(grid) and grid[x+1][y]==1:
            _ans[point].append((x+1,y))
        if y+1<len(grid[0]) and grid[x][y+1]==1:
            _ans[point].append((x,y+1))
        if x-1>=0 and grid[x-1][y]==1:
            _ans[point].append((x-1,y))
        if y-1>=0 and grid[x][y-1]==1:
            _ans[point].append((x,y-1))
    return _ans


def __get_transit_prob_table(neighbour:dict, cell_to_steps:dict, step_to_cells:dict)->dict:
    """
    based on current data, calculate the probability that one cell moves to the specific next cell
    """
    _ans = {}
    for cell in cell_to_steps.keys():
        _total_for_this_cell = 0.0
        steps = cell_to_steps[cell]#possible steps for this cell
        _ans[cell] = {}
        neighbours = neighbour[cell]
        for step in steps:
            for next_cell in neighbours:
                if (step+1) in step_to_cells.keys() and (next_cell in step_to_cells[step+1]):
                    _ans[cell][next_cell]=_ans[cell].get(next_cell,0)+1
                    _total_for_this_cell += 1.0
                else:
                    continue
                
        for key in _ans[cell].keys():
            _ans[cell][key]/=_total_for_this_cell
    return _ans
            

def __find_most_likely_path(trans_prob_table:dict,step_to_cells:dict,neighbour:dict):
    _ans = {}
    for i in range(len(step_to_cells.keys())):
        _ans[i]={}
    for cell in step_to_cells[0]:
        num = len(step_to_cells[0])
        _ans[0][cell] = (Node(prob=1/num, c=cell))
    for i in range(1,len(step_to_cells.keys())):
        _pre = _ans[i-1]
        for cell in _pre.keys():
            for next_cell in step_to_cells[i]:
                if not next_cell in neighbour[cell]:
                    continue
                if next_cell not in _ans[i]:
                    _ans[i][next_cell] = Node(pre=_ans[i-1][cell],c=next_cell, prob=_ans[i-1][cell].prob * trans_prob_table[cell][next_cell])
                    continue
                if _ans[i-1][cell].prob * trans_prob_table[cell][next_cell]>_ans[i][next_cell].prob:
                    _ans[i][next_cell] = Node(pre = _ans[i-1][cell],c=next_cell, prob=_ans[i-1][cell].prob * trans_prob_table[cell][next_cell])
    return _ans


def __generate_path(_graph:dict)->List[tuple]:
    keys = list(_graph.keys())
    keys.sort()
    k = keys[-1]
    _path = []
    _temp = []
    for key in _graph[k].keys():
        _temp.append(_graph[k][key])
    _temp = sorted(_temp,key = lambda x:x.prob, reverse=True)
    tail = _temp[0]
    while tail:
        _path.append(tail.coordinates)
        tail = tail.pre
    _path.reverse()
    return _path
    

def __viterbi()->List[tuple]:
    """
    run me
    """
    grid = __read_grid()
    free_cells = __get_free_cells(grid)
    tower_locations = [(0, 0), (0, 9), (9, 0), (9, 9)]
    noisy_dists = __get_noisy_distances()
    _to_tower_dists = __get_tower_dists(free_cells, tower_locations)
    _step_to_cells = __get_possible_cells_per_step(noisy_dists, free_cells,_to_tower_dists)
    _cell_to_steps = __get_possible_steps_per_cell(_step_to_cells, free_cells)
    _neighbours = __get_neighbours(free_cells, grid)
    _trans_prob_table = __get_transit_prob_table(_neighbours, _cell_to_steps, _step_to_cells)
    _graph = __find_most_likely_path(_trans_prob_table, _step_to_cells, _neighbours)
    _ans = __generate_path(_graph)
    return _ans


if __name__ == "__main__":
    _ans = __viterbi()
    print(_ans)
    path = os.getcwd()
    for i in os.listdir(path):
        if i.endswith("~"):
            os.remove(i)
