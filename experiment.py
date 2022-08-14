import scipy
import numpy as np

import tensorly as tl
from tensorly.decomposition import parafac

from src.partitions import Tensorization, Sampler
from src.algorithms.parafac import Parafac
from src.algorithms.mrlr import Mrlr

data = scipy.io.loadmat("data/amino.mat")['X']
tensorizer = Tensorization(data)

partition = [[1], [0, 2]]
partition = [[0], [1], [2]]

X_2, X_2_recovery_map = tensorizer.tensor_unfold(partition)
data_recovered = tensorizer.tensor_refold(X_2, X_2_recovery_map)

assert np.array_equal(data_recovered, data)

sampler = Sampler(data.shape)
sampler.sample_entry()
sampler.sample_factor()

factors = parafac(tensor=data, rank=10, n_iter_max=100, init='random', tol=1e-12)
data_hat = tl.cp_to_tensor(factors)
error = np.linalg.norm(data - data_hat)
#print(error)

data_hat_custom = Parafac(data, 10).optimize(100, 1e-12)
print(np.linalg.norm(data_hat_custom - data_hat))

data_hat_mrlr = Mrlr(data, ranks=[10, 10], partitions=[[[0], [1], [2]], [[1], [0, 2]]]).optimize(100, 1e-8)
print(np.linalg.norm(data - data_hat_mrlr))

