







def assign(X,means):
    '''Assign closest mean to the points'''
    Z = euclidean_distance(X,means)
    return Z.argmin(axis=0)


def euclidean_distance(X, Y):
    '''X, a matrix (n,p), Y a matrix (m,p), returns an (n,m) distance matrix'''
    x_square = (x * x).sum(axis=1, keepdims=True) * da.ones((1, y.shape[0]), chunks=(y.shape[0]))
    y_square = da.ones((x.shape[0], 1), chunks=(x.shape[0])) * (y * y).sum(axis=1, keepdims=True).transpose()
    return x_square - 2 * x.dot(y.transpose()) + y_square