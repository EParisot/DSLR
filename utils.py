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
    return (variance / tab_count) ** (1/2)

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