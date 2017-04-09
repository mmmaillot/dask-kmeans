import dask.array as da
import numpy as np

def init(X, nclust):
    ''' Choose the values of the means for the first iteration'''
    return X[np.random.choice(range(X.shape[0]), size=nclust)]
    
def assign(X,M):
    '''Assign closest mean to the points'''
    return euclidean_distance(X,M).argmin(axis=1)

def update(X, Label):
    '''Update the means by using the labels computed by assign'''
    Y = X.to_dask_dataframe()
    return Y.groupby(Label.to_dask_dataframe()).mean().values


def euclidean_distance(x, y):
    '''X, a matrix (n,p), Y a matrix (m,p), returns an (n,m) distance matrix'''
    x_square = (x * x).sum(axis=1, keepdims=True) * da.ones((1, y.shape[0]), chunks=(y.shape[0]))
    y_square = da.ones((x.shape[0], 1), chunks=(x.shape[0])) * (y * y).sum(axis=1, keepdims=True).transpose()
    return x_square - 2 * x.dot(y.transpose()) + y_square

def main(x,k,max_iter):
    means = init(x,k)
    for i in range(max_iter):
        old_means = means
        labels = assign(x,means)
        means = update(x,labels).compute()
        # if means == old_means:
        #    break
    return (means,labels)
