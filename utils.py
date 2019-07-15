def mean(tab):
    total = 0
    for elem in tab:
        if elem != "NaN":
            total += elem
    return total / count(tab)

def count(tab):
    count = 0
    for _ in tab:
        count += 1
    return count

def std(tab):
    variance = 0
    tab_mean = mean(tab)
    for elem in tab:
        variance += (elem ** 2) - (tab_mean ** 2)
    std = ft_sqrt(variance/count(tab))
    return std

def _min(tab):
    _min = tab[0]
    for elem in tab:
        if elem < _min:
            _min = elem
    return _min

def _max(tab):
    _max = tab[0]
    for elem in tab:
        if elem > _max:
            _max = elem
    return _max

def q_1(tab):
    tab_count = count(tab)
    tab = sorted(tab)
    while tab_count % 4 != 0:
        tab_count += 1
    return tab[tab_count//4]

def q_2(tab):
    tab_count = count(tab)
    tab = sorted(tab)
    if tab_count % 2 != 0:
        tab_count += 1
    return tab[tab_count//2]

def q_3(tab):
    tab_count = count(tab)
    tab = sorted(tab)
    while 3 * tab_count % 4 != 0:
        tab_count += 1
    return tab[3*tab_count//4]

def ft_sqrt(x):
        if x==0 or x==1:
            return x
        else:
            start = 0
            end = x  
            while (start <= end):
                mid = (start + end) // 2
                if (mid * mid == x):
                    return mid
                elif (mid * mid < x):
                    start = mid + 1
                    res = mid
                else:
                    end = mid - 1
            return res