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
	M_j = psi * np.eye(k, k)  + np.dot(Lam.T, Lam_j)
	assert (M_j.shape == (k, k))
	# print(Yj.shape)
	# x_j = M_j.dot(Lam_j.T).dot(Yj) #kx1
	temp = M_j.dot(Lam_j.T)
	# yjtemp = Yj.reshape(-1)
	# print(temp.shape)
	x_j = sp.csc_matrix.dot(temp, Yj)
	# print(temp2.shape)
	try:
		assert( x_j.shape == (k, 1) )
	except AssertionError:
		raise(AssertionError('x_j shape = {}. k = {}'.format(x_j.shape, k)))
	

	return M_j, x_j




def M_step(Ycol, Lam, psi, k, n, m):
	accum_A = sp.csc_matrix((k*n, k*n))
	accum_B = sp.csc_matrix((n,k))
	psi_p = 0.0
	for j in range(0, m):
		item_idx = Ycol.indices[Ycol.indptr[j]:Ycol.indptr[j + 1]]  # items user has rated
		assert (item_idx.shape[0] != 0)
		idx_vec = np.zeros(n)
		idx_vec[item_idx] = 1
		Dj = sp.diags(idx_vec)
		assert(Dj.shape == (n,n))

		Mj, xj = E_step(Lam, psi, Ycol[:, j], item_idx, k, n, m)
		Aj = (np.outer(xj, xj) + psi * Mj) / item_idx.shape[0]
		accum_A +=  sp.kron(Dj, Aj)
		Bj = Ycol[:, j].dot(xj.T) / item_idx.shape[0]
		accum_B += Bj
		psi_p += (Ycol[:, j].T.dot(Ycol[:, j])).todense()

	# print(accum_A.__class__)

	Lam_p = spLinAlg.inv(accum_A).dot(accum_B.reshape((-1, 1))).reshape( (n, k))

	# print("psi = {}".format(psi_p))
	psi_p -= np.trace(Lam_p.T.dot(accum_B))
	psi_p /= m #normalize
	# psi_p = psi_p.todense()

	return Lam_p, psi_p


# if __name__ == '__main__':
# 	k = 10 # number of latent factors
# 	FNAME = 'data/LasVegas_local.pck'

# 	with open(FNAME, 'rb') as pickle_file:
# 		dfY = pickle.load(pickle_file)

# 	# Y is a pandas dataframe array with d0 = users and d1 = businesses
# 	# columns are axis 0 in df
# 	# Ycol = sp.csc_matrix(dfY.v)
# 	Ycol = dfY
# 	n, m = Ycol.shape


# 	Lam, psi = initialize()

# 	iters = 0
# 	while True:
# 		new_lam, new_psi = M_step(Ycol, Lam, psi)
# 		lam_diff, psi_diff = np.linalg.norm(Lam-new_lam), np.linalg.norm(psi-new_psi)
# 		print('iter: {}\tlam_diff:{}\tpsi_diff{}'.format(iters, lam_diff, psi_diff))
# 		iters += 1
# 		if iters > 10:
# 			break





