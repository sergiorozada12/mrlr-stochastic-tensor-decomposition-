import itertools
import numpy as np

class Tensorization:
    def __init__(self, tensor):
        self.tensor = tensor

    def _reindex(self, partitions, index):
        new_index = [0]*len(partitions)
        for i in range(len(partitions)):
            partition = partitions[i]
            dimension_1 = partition[0]
            n_p_1 = index[dimension_1]

            if len(partition) == 1:
                new_index[i] = n_p_1
                continue

            result = n_p_1

            for j in range(1, len(partition)):
                dimension = partition[j]
                n_p_k = index[dimension]
                prod = np.prod([self.tensor.shape[partition[k]] for k in range(j)])
                result += n_p_k*prod
            new_index[i] = int(result)

        return tuple(new_index)

    def tensor_unfold(self, partitions):
        generators = [list(range(dimension_shape)) for dimension_shape in self.tensor.shape]
        indices = itertools.product(*generators)

        new_shape = [0]*len(partitions)
        for idx, partition in enumerate(partitions):
            new_shape[idx] = np.prod([self.tensor.shape[dimension] for dimension in partition])

        idx_recovery_map = {}
        new_tensor = np.zeros(new_shape)
        for idx in indices:
            new_idx = self._reindex(partitions, idx)
            new_tensor[new_idx] = self.tensor[idx]
            idx_recovery_map[new_idx] = idx
        return new_tensor, idx_recovery_map

    def tensor_refold(self, tensor, recovery_map):
        tensor_recovered = np.zeros(self.tensor.shape)
        for idx in recovery_map:
            tensor_recovered[recovery_map[idx]] = tensor[idx]
        return tensor_recovered
