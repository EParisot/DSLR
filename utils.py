import re
import os
import json

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
    return data

def read_model(model_file, classes):
    model = {}
    ranges = {}
    if os.path.exists(model_file):
        with open(model_file, "r") as f:
            check = f.read(2)
            f.seek(0)
            if len(check) != 0 and check[0] != "\n" and check != "{}":
                data = json.load(f)
                for curr_class in data["weights"].items():
                    for key, val in data["weights"][curr_class].items():
                        model[key] = val
                    for key, val in data["ranges"].items():
                        ranges[key] = val
            else:
                for _class in classes:
                    model[_class] = {}
    return model, ranges

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

def get_numerics(data, exclude_nan):
    r = re.compile(r"-?\d+\.\d+")
    num_data = {}
    # double check num values
    for key in data[0]:
        if r.match(data[0][key]):
            num_data[key] = []
    for key in data[-1]:
        if r.match(data[-1][key]):
            num_data[key] = []
    # build numeric array
    for elem in data:
        for key in elem:
            if key in num_data:
                if r.match(elem[key]):
                    num_data[key].append(float(elem[key]))
                elif exclude_nan == False:
                    num_data[key].append("NaN")
    return num_data

def get_classes(data, idx):
    classes = {} 
    for elem in data:
        classes[elem[idx]] = []
    for elem in data:
        classes[elem[idx]].append(data.index(elem))
    return classes

def get_Y(data, idx):
    Y = []
    for elem in data:
        Y.append(elem[idx])
    return Y

def mean(tab):
    total = 0
    for elem in tab:
        if elem != "NaN":
            total += elem
    return total / count(tab)

def count(tab):
    count = 0
    for elem in tab:
        if elem != "NaN":
            count += 1
    return count

def std(tab):
    variance = 0
    tab_mean = mean(tab)
    tab_count = count(tab)
    for elem in tab:
        if elem != "NaN":
            variance += (elem - tab_mean) ** 2
    return (variance / (tab_count - 1)) ** (1/2)

def _min(tab):
    _min = tab[0]
    for elem in tab:
        if elem != "NaN":
            if elem < _min:
                _min = elem
    return _min

def _max(tab):
    _max = tab[0]
    for elem in tab:
        if elem != "NaN":
            if elem > _max:
                _max = elem
    return _max

def q_1(tab):
    clean_tab = []
    for elem in tab:
        if elem != "NaN":
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
        if elem != "NaN":
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
        if elem != "NaN":
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