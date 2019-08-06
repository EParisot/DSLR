import re
import os
import json
from math import isnan

def read_data(data_file, sep):
    data = []
    labels = []
    if os.path.exists(data_file):
        with open(data_file) as f:
            for i, line in enumerate(f):
                line = line.replace('\n', '')
                line_data = line.split(sep)
                # read labels
                if i == 0 and len(line_data) > 0:
                    for label in line_data:
                        labels.append(label)
                # read data
                elif len(line_data) > 0:
                    line_dict = {}
                    for j, feature in enumerate(line_data):
                        line_dict[labels[j]] = feature
                    data.append(line_dict)
    return data, labels

def read_model(model_file, classes):
    model = {}
    ranges = {}
    if os.path.exists(model_file):
        with open(model_file, "r") as f:
            check = f.read(2)
            f.seek(0)
            if len(check) != 0 and check[0] != "\n" and check != "{}":
                data = json.load(f)
                model = data["weights"]
                ranges = data["ranges"]
                if len(classes) == 0:
                    classes = list(model.keys())
            else:
                for _class in classes:
                    model[_class] = {}
    else:
        print("No model file")
        exit(0)
    return model, ranges, classes

def save_model(model, ranges, model_file):
    data = {}
    data["weights"] = model
    data["ranges"] = ranges
    if not os.path.exists(model_file):
        mode = "w+"
    else:
        mode = "w"
    with open(model_file, mode) as f:
        json.dump(data, f)

def get_numerics(data, get_hand=False):
    r = re.compile(r"-?\d+\.\d+")
    num_data = {}
    # double check num values
    if get_hand:
        num_data["Best Hand"] = []
    for key in data[0]:
        if r.match(data[0][key]):
            num_data[key] = []
    for key in data[-1]:
        if r.match(data[-1][key]):
            num_data[key] = []
    # build numeric array
    for elem in data:
        for key in elem:
            if get_hand and key == "Best Hand":
                if elem[key] == "Left":
                    num_data[key].append(0)
                else:
                    num_data[key].append(1)
            elif key in num_data:
                if r.match(elem[key]):
                    num_data[key].append(float(elem[key]))
                else:
                    num_data[key].append("NaN")
    return num_data

def clean(X):
    clean_X = {}
    for key in X:
        clean_X[key] = []
    for idx, _ in enumerate(X[next(iter(X))]):
        for key in X:
            if X[key][idx] == "NaN":
                clean_X[key].append(mean(X[key]))
            else:
                clean_X[key].append(X[key][idx])
    return clean_X

def hard_clean(X, Y):
    clean_X = {}
    clean_Y = []
    for key in X:
        clean_X[key] = []
    for idx, _ in enumerate(X[next(iter(X))]):
        nan = False
        for key in X:
            if X[key][idx] == "NaN":
                nan = True
                break
        if nan == False:
            clean_Y.append(Y[idx])
            for key in X:
                clean_X[key].append(X[key][idx])
    return clean_X, clean_Y

def get_classes(data, idx):
    classes = {} 
    for elem in data:
        classes[elem[idx]] = []
    for elem in data:
        classes[elem[idx]].append(data.index(elem))
    return classes

def classes_list(data, idx):
    classes = []
    for elem in data:
        if elem[idx] not in classes:
            classes.append(elem[idx])
    return classes

def get_Y(data, idx):
    Y = []
    for elem in data:
        Y.append(elem[idx])
    return Y

def mean(tab):
    total = 0
    tab_count = count(tab)
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            total += elem
    return total / tab_count

def count(tab):
    count = 0
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            count += 1
    return count

def std(tab):
    variance = 0
    tab_mean = mean(tab)
    tab_count = count(tab)
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            variance += (elem - tab_mean) ** 2
    return (variance / (tab_count - 1)) ** (1/2)

def _min(tab):
    _min = tab[0]
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            if elem < _min:
                _min = elem
    return _min

def _max(tab):
    _max = tab[0]
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            if elem > _max:
                _max = elem
    return _max

def q_1(tab):
    clean_tab = []
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            clean_tab.append(elem)
    tab_count = count(clean_tab)
    tab = sorted(clean_tab)
    rank = ((tab_count + 3) / 4) - 1
    if rank - int(rank) > 0:
        if rank - int(rank) == 0.25:
            res = ((3 * tab[int(rank)]) + (1 * tab[int(rank) + 1])) / 4
        elif rank - int(rank) == 0.5:
            res = ((2 * tab[int(rank)]) + (2 * tab[int(rank) + 1])) / 4
        elif rank - int(rank) == 0.75:
            res = ((1 * tab[int(rank)]) + (3 * tab[int(rank) + 1])) / 4
        return res
    else:
        return tab[int(rank)]

def q_2(tab):
    clean_tab = []
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            clean_tab.append(elem) 
    tab_count = count(clean_tab)
    tab = sorted(clean_tab)
    rank = ((tab_count + 1) / 2) - 1
    if rank - int(rank) > 0:
        res = (tab[int(rank)] + tab[int(rank) + 1]) / 2
        return res
    else:
        return tab[int(rank)]

def q_3(tab):
    clean_tab = []
    for elem in tab:
        if elem != "NaN" and not isnan(elem):
            clean_tab.append(elem)
    tab_count = count(clean_tab)
    tab = sorted(clean_tab)
    rank = ((3 * tab_count + 1) / 4) - 1
    if rank - int(rank) > 0:
        if rank - int(rank) == 0.25:
            res = ((3 * tab[int(rank)]) + (1 * tab[int(rank) + 1])) / 4
        elif rank - int(rank) == 0.5:
            res = ((2 * tab[int(rank)]) + (2 * tab[int(rank) + 1])) / 4
        elif rank - int(rank) == 0.75:
            res = ((1 * tab[int(rank)]) + (3 * tab[int(rank) + 1])) / 4
        return res
    else:
        return tab[int(rank)]
