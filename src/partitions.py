import itertools
import random
from typing import List, Tuple, Dict, Optional
import numpy as np

class Tensorization:
    def __init__(self, tensor: np.array) -> None:
        self.tensor = tensor

    def _reindex(
        self,
        partitions: List[List[int]],
        index: List[int]
    ) -> List[int]:

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

    def tensor_unfold(
        self,
        partitions: List[List[int]]
    ) -> Tuple[np.array, Dict[List[int], List[int]]]:

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

    def tensor_refold(
        self,
        tensor: np.array,
        recovery_map: Dict[List[int], List[int]]
    ) -> np.array:

        tensor_recovered = np.zeros(self.tensor.shape)
        for idx in recovery_map:
            tensor_recovered[recovery_map[idx]] = tensor[idx]
        return tensor_recovered


class Sampler:
    def __init__(self, shape: Tuple[int], n_lowres_comp: Optional[int]=None) -> None:
        self.shape = shape
        self.n_lowres_comp = n_lowres_comp

    def sample_factor(self) -> int:
        return random.randint(0, len(self.shape) - 1)

    def sample_entry(self) -> List[int]:
        return [random.randint(0, self.shape[i] - 1) for i in range(len(self.shape))]

    def sample_lowres_comp(self) -> int:
        if not self.n_lowres_comp:
            return None
        return random.randint(0, len(self.shape) - 1)
