import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle
import math

k = 10 # number of latent factors
FNAME = 'data/LasVegas_local.pck'

with open(FNAME) as pickle_file:
	dfY = pickle.load(pickle_file)

# Y is a pandas dataframe array with d0 = users and d1 = businesses
# columns are axis 0 in df
# Ycol = sp.csc_matrix(dfY.v)
Ycol = dfY
n, m = Ycol.shape

#Y is an n x m matrix (n businesses, m users) - sparse
#Lambda is n x k matrix
def initialize():
	Lam = np.random.randn(n, k)*1.0
	psi = np.random.randn(1, 1)*1.0
	#x   = np.random.randn(k, m) #canonical preferences of each user
	return Lam, psi

def E_step(Lam, psi, Yj, item_idx):
	Lam_j = Lam[item_idx, :] #rows corresponding to user j's rated biz
	assert (Lam_j.shape == (n, k))
	M_j = psi * np.eye(k, k)  + np.dot(Lam.T, Lam_j)
	assert (M_j.shape == (k, k))
	x_j = M_j.dot(Lam_j.T).dot(Yj) #kx1
	assert( x_j.shape == (k, 1) )

	return M_j, x_j




def M_step(Ycol, Lam, psi):
	accum_A = sp.csr_matrix((k*n, k*n))
	accum_B = np.zeros((n,k))
	psi_p = 0
	for j in range(0, m):
		item_idx = Ycol.indices[Ycol.indptr[j]:Ycol.indptr[j + 1]]  # items user has rated
		assert (item_idx.shape[0] != 0)
		idx_vec = np.zeros(1, n);
		idx_vec[item_idx] = 1
		Dj = sp.diags(idx_vec)
		assert(Dj.shape == (n,n))

		Mj, xj = E_step(Lam, psi, Ycol[j], item_idx)
		Aj = (np.outer(xj, xj) + psi * Mj) / item_idx.shape[0]
		accum_A +=  sp.kron(Dj, Aj)
		Bj = np.outer(Ycol[j], xj) / item_idx.shape[0]
		accum_B += Bj
		psi_p += np.inner(Ycol[j], Ycol[j])

	Lam_p = np.reshape(sp.linalg.inv(accum_A).dot(np.reshape(accum_B, -1, 1)), n, k)

	psi_p -= np.trace(Lam_p.dot(accum_B))

	return Lam_p, psi_p








