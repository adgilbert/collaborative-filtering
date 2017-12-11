import pickle
from canny_cf import *
# import matplotlib.pyplot as plt
import scipy.sparse as sp

# This function trains the dataset and saves the results

k = 15 # number of latent factors
FNAME = 'data/LasVegas_local.pck'
# FNAME2 should be the tourist dataset
FNAME2 = 'data/LasVegas_local.pck' # Only used if MODE = 'combined'
PROPORTION = .2 # Train to test percentage
ITERS = 25 # Number of times to run EM
SAVENAME='data/local15results.pck' # where to save the results of the simulation
MODE='regular' # mode is either 'combined' or 'regular'. 

# mode = 'regular' means that the script loads in a dataset, trains on most of it, and tests on a small portion
# mode  = 'combined' should be used when you want to load in local and tourist data. Then it trains on the local data (and some tourist data) 
# and tests on the remainder of the tourist data


if MODE == 'regular':
    with open(FNAME, 'rb') as pickle_file:
        Ycol = pickle.load(pickle_file)
    print('Ycol shape: {}'.format(Ycol.shape))

    # Split the dataset
    Ytest, Ytrain, test_row_ind, test_col_ind, _, _  = split_Y(Ycol, PROPORTION)
    Ytotal = Ycol
    original_shape = Ytotal.shape # not used for 'regular' case

elif MODE == 'combined':
    # Open the first file (local file)
    with open(FNAME, 'rb') as pickle_file:
        Ycol1 = pickle.load(pickle_file)
    print('Ycol shape: {}'.format(Ycol1.shape))

    # Open the second file (tourist file)
    with open(FNAME2, 'rb') as pickle_file:
        Ycol = pickle.load(pickle_file)
    print('Ycol shape: {}'.format(Ycol.shape))

    # Split dataset 2 (the tourist data) into testing and training
    Ytest, Ytrain, test_row_ind, test_col_ind, train_row_ind, train_col_ind  = split_Y(Ycol, PROPORTION)
    original_shape = Ytrain.shape # used to split the data again when evaluating 
    
    # Only take businesses that survived the splitting for Ycol2
    Ycol1 = Ycol1[train_row_ind, :]
    # Now combine our two training matrices (Ycol1 is enturely used for training)
    Ytrain = sp.hstack([Ytrain, Ycol1], format='csc')
    Ytotal = sp.hstack([Ycol, Ycol1], format='csc')
    
    print('Ytrain shape: {}'.format(Ytrain.shape))

    # Now go through again and make sure Ytrain has no empty rows (precautionary measure).
    print('Ytrain before removing empty: {}'.format(Ytrain.shape))
    train_row_ind, train_col_ind = np.unique(Ytrain.nonzero()[0]), np.unique(Ytrain.nonzero()[1])
    Ytrain = Ytrain[train_row_ind,:]
    Ytrain = Ytrain[:, train_col_ind]
    Ytotal = Ytotal[:, train_col_ind]
    Ytotal = Ytotal[train_row_ind, :] 
    print('below shouldn\'t change. If it does there may be a problem. See code comments for details')   
    print('Ytrain after removing empty: {}'.format(Ytrain.shape))
    print('Ytotal after removing empty: {}'.format(Ytotal.shape))
    # The above really shouldn't really change. If it does we don't know if users were removed from the test set or the training set and it 
    # might throw off the testing results as we would no longer be comparing the same users anymore. 


# Train 
lam_diff, psi_diff, train_err, test_err, x, Lam = train(Ytotal, Ytrain, k, ITERS, Ytest, PROPORTION, test_row_ind, test_col_ind, original_shape)

# Get the correct rows and columns of Lambda and x
x_test, Lam_test = split_others(Ytotal[:, :original_shape[1]], x, Lam, PROPORTION, test_row_ind, test_col_ind)


results = dict(
        x=x,
        Ytest=Ytest,
        Ytrain=Ytrain,
        Lam=Lam,
        x_test=x_test,
        Lam_test=Lam_test,
        lam_diff=lam_diff,
        psi_diff=psi_diff,
        train_err=train_err,
        test_err=test_err 
)
pickle.dump(results, open(SAVENAME, 'wb'))

