import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spLinAlg
import pandas as pd
import pickle
import math


#Y is an n x m matrix (n businesses, m users) - sparse
#Lambda is n x k matrix
def initialize(n, k):
    Lam = np.random.randn(n, k)*1.0
    psi = np.random.randn(1, 1)*1.0
    #x   = np.random.randn(k, m) #canonical preferences of each user
    return Lam, psi

def E_step(Lam, psi, Yj, item_idx, k, n, m):
    Lam_j = np.zeros((n, k))
    Lam_j[item_idx, :] = Lam[item_idx, :] #rows corresponding to user j's rated biz
    try:
        assert (Lam_j.shape == (n, k))
        assert(Lam.shape == (n, k))
    except AssertionError:
        raise(AssertionError('Lam_j shape = {}. Lam shape = {}. n, k = {}, {}'.format(Lam_j.shape, Lam.shape, n, k)))
    M_j = np.linalg.inv(psi * np.eye(k, k)  + np.dot(Lam.T, Lam_j)) #Lam_j equivalent to D_j * Lam
    assert (M_j.shape == (k, k))
    # print(Yj.shape)
    x_j = sp.csc_matrix.dot(M_j.dot(Lam_j.T), Yj)
    try:
        assert( x_j.shape == (k, 1) )
    except AssertionError:
        raise(AssertionError('x_j shape = {}. k = {}'.format(x_j.shape, k)))
    

    return M_j, x_j.reshape(-1)




def M_step(Ycol, Lam, psi, k, n, m):
    accum_A = [np.zeros((k,k)) for i in range(n)] 
    accum_B = np.zeros((n,k)) #sp.csc_matrix((n,k))
    x = np.zeros((k, m))
    psi_p = 0.0
    for j in range(0, m):
        if (j % 10000 == 0):
            print('Evaluated user {}'.format(j))
        item_idx = Ycol.indices[Ycol.indptr[j]:Ycol.indptr[j + 1]]  # items user has rated
        assert (item_idx.shape[0] != 0)
        idx_vec = np.zeros(n)
        idx_vec[item_idx] = 1
        
        Mj, x[:, j] = E_step(Lam, psi, Ycol[:, j], item_idx, k, n, m)
        Aj = (np.outer(x[:, j], x[:, j]) + psi * Mj) / item_idx.shape[0]
        for biz in item_idx:
            accum_A[biz] += Aj
        Bj = Ycol[:, j].dot(x[:, j].reshape(1, -1)) / item_idx.shape[0] 
        accum_B += Bj
        psi_p += (Ycol[:, j].T.dot(Ycol[:, j])).todense() / item_idx.shape[0]

    # print(accum_A.__class__)
    Lam_p = np.zeros((n,k))
    for i, A in enumerate(accum_A):
        Lam_p[i,:] = np.linalg.inv(A).dot(accum_B[i,:].T).T

    # Lam_p = spLinAlg.inv(accum_A).dot(accum_B.reshape((-1, 1))).reshape( (n, k))

    # print("psi = {}".format(psi_p))
    psi_p -= np.trace(Lam_p.T.dot(accum_B))
    psi_p /= m #normalize
    # psi_p = psi_p.todense()

    return Lam_p, psi_p, x


def train(Ycol, Ytrain, k, iters, Ytest, test_row_ind, test_col_ind, PROPORTION):
    n, m = Ycol.shape

    print("n={}\nm={}\nk={}".format(n, m, k))

    Lam, psi = initialize(n, k)
    # print(Lam.shape)
    # print(psi.shape)

    print('Starting Iterations\n========================')
    lam_diff = np.zeros(iters)
    psi_diff = np.zeros(iters)
    for i in range(iters):
        new_lam, new_psi, x = M_step(Ytrain, Lam, psi, k, n, m)
        lam_diff[i], psi_diff[i] = np.linalg.norm(Lam-new_lam), np.linalg.norm(psi-new_psi)
        # Test results. For Train only test on the 
        train_err = test(Ytrain, x, new_lam)
        # Get the correct rows and columns of Lambda and x
        x_test, Lam_test = split_others(Ycol, x, new_lam, PROPORTION, test_row_ind, test_col_ind)
        test_err = test(Ytest, x_test, Lam_test)

        print('iter: {} \tlam_diff:  {:.4f}\tpsi_diff:  {:.4f}\ttrain_err:  {:.4f}\ttest_err:  {:.4f}'
            .format(i, lam_diff[i], psi_diff[i], train_err, test_err))
        Lam, psi = np.array(new_lam), np.array(new_psi)


    return lam_diff, psi_diff, x, Lam


def test(Ytr, X, Lam):
    """
    Determine the mean absolute error for the non-zero entries of the training data
    Ytr is (n, m) representing ratings by m users of n restaurants 
        -- to accomodate large data sets, Ytr is assumed to be sparse
    X is (k,m) representing k canonical preferences of m users
    Lam is (n,k) representing factor loading of k canonical preferences for n businesses
    """
    try:
        assert (Ytr.shape[0] == Lam.shape[0])
        assert (Ytr.shape[1] == X.shape[1])
        assert (Lam.shape[1] == X.shape[0])
    except AssertionError:
        print('Ytr shape = {}'.format(Ytr.shape))
        print('X shape = {}'.format(X.shape))
        print('Lam shape = {}'.format(Lam.shape))
        raise(AssertionError('shape mismatch'))

    #find non-zero ratings in training data
    row, col, data = sp.find(Ytr)
    absolute_err_sum = 0.0
    for i, ytr in enumerate(data):
        ypred = np.inner(Lam[row[i],:], X[:,col[i]])
        absolute_err_sum += abs(ypred - ytr)
    return (absolute_err_sum / len(data))


def split_Y(Ycol, proportion):
    """
    splits the dataset Ycol into training and testing datasets of the given proportion.
    Because this is EM we take the bottom right corner of Y as the testing set
    """
    n, m = Ycol.shape
    test_n, test_m = int(n*proportion), int(m*proportion)

    # first get Train
    row, col, data = sp.find(Ycol)
    save_inds = [(r, c, d) for (r, c, d) in zip(row, col, data) if r < n-test_n or c < m - test_m]
    save_inds = np.array(save_inds)
    train_row, train_col, train_data = save_inds[:, 0], save_inds[:, 1], save_inds[:, 2]
    Ytrain = sp.csc_matrix((train_data, (train_row, train_col)) )
    # Now get rid of columns or rows that are empty
    print('Ytrain before removing empty: {}'.format(Ytrain.shape))
    train_row_ind, train_col_ind = np.unique(Ytrain.nonzero()[0]), np.unique(Ytrain.nonzero()[1])
    Ytrain = Ytrain[train_row_ind,:]
    Ytrain = Ytrain[:, train_col_ind]
    print('Ytrain after removing empty: {}'.format(Ytrain.shape))

    # Now get test: 
    #have to only get rows that have actually been trained on first of all. 
    Ytest = Ycol[:, train_col_ind] #do column slicing first for increased efficiency
    Ytest = Ytest[train_row_ind, :]
    # now take the last 20% (which will no longer actually be 20%)
    Ytest = Ytest[n-test_n:, m-test_m:]
    # Again remove empty columns or rows
    print('Ytest before removing empty: {}'.format(Ytest.shape))
    test_row_ind, test_col_ind = np.unique(Ytest.nonzero()[0]), np.unique(Ytest.nonzero()[1])
    Ytest = Ytest[test_row_ind,:]
    Ytest = Ytest[:, test_col_ind]
    print('Ytest after removing empty: {}'.format(Ytest.shape))

    return Ytest, Ytrain, test_row_ind, test_col_ind, train_row_ind, train_col_ind



def split_others(Ycol, x, Lam, proportion, test_row_ind, test_col_ind):
    """
    split the resulting x and lambda into the just the piece for the test set using the same methodology as above
    """
    n, m = Ycol.shape
    test_n, test_m = int(n*proportion), int(m*proportion)

    # Perform the exact same iterations as the test set
    # x_test = x[:, train_col_ind]
    x_test = x[:, m-test_m:]
    x_test = x_test[:, test_col_ind]
    print('x_test shape: {}'.format(x_test.shape))

    # Lam_test = Lam[train_row_ind, :]
    Lam_test = Lam[n-test_n:, :]
    Lam_test = Lam_test[test_row_ind, :]
    print('Lam_test shape: {}'.format(Lam_test.shape))

    return x_test, Lam_test


