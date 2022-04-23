def one_hot(i, n):
    v = [0] * n
    if i >= 0:
        v[int(i)] = 1
    return v
