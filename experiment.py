import scipy
import numpy as np

from src.partitions import Tensorization

data = scipy.io.loadmat("data/amino.mat")['X']
tensorizer = Tensorization(data)

partition = [[1], [0, 2]]

X_2, X_2_recovery_map = tensorizer.tensor_unfold(partition)
data_recovered = tensorizer.tensor_refold(X_2, X_2_recovery_map)

print(type(X_2))
print(type(data))
print(type(data_recovered))

assert np.array_equal(data_recovered, data)