
# coding: utf-8

# In[111]:

import pickle
from canny_cf import *
import matplotlib.pyplot as plt


# In[ ]:

k = 10 # number of latent factors
FNAME = 'data/LasVegas_local.pck'
PROPORTION = .2 # Train to test percentage
ITERS = 20 # Number of times to run EM

with open(FNAME, 'rb') as pickle_file:
    Ycol = pickle.load(pickle_file)
print('Ycol shape: {}'.format(Ycol.shape))

# Split the dataset
Ytest, Ytrain, test_row_ind, test_col_ind  = split_Y(Ycol, PROPORTION)

# Train 
lam_diff, psi_diff, x, Lam = train(Ytrain, k, ITERS)

# Get the correct rows and columns of Lambda and x
x_test, Lam_test = split_others(Ycol, x, Lam, PROPORTION, test_row_ind, test_col_ind)


results = dict(
        x=x,
        Ytest=Ytest,
        Ytrain=Ytrain,
        Lam=Lam,
        x_test=x_test,
        Lam_test=Lam_test,
        lam_diff=lam_diff,
        psi_diff=psi_diff )
pickle.dump(results, open('data/training_results.pck', 'wb'))


# Test the result
res = test(Ytest, x_test, Lam_test)
tr_err = test(Ytrain, x, Lam)
print('\nResults\n===========')
print("Mean Absolute Error for training = {}".format(tr_err))
print("Mean Absolute Error for test = {}".format(res))
#print('Mean of test array: {}'.format(np.mean(abs(res))))
#print('Std of test array: {}'.format(np.std(abs(res))))

    


# In[158]:

plt.plot(np.arange(ITERS), lam_diff, label='Lambda')
plt.plot(np.arange(ITERS), psi_diff, label="Psi")
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('L2 norm of Difference')
plt.title('Difference between Successive Iterations')
plt.show()


# In[ ]:



