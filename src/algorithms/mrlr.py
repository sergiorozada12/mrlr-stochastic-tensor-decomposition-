from typing import List

import numpy as np

from src.algorithms.parafac import Parafac
from src.partitions import Tensorization, Sampler


class Mrlr:
    def __init__(
        self,
        tensor: np.array,
        ranks: List[int],
        partitions: List[List[List[int]]]
    ) -> None:

        self.tensor = tensor
        self.ranks = ranks

        self.partitions = partitions

    def optimize(
        self,
        max_iter: int,
        tol: float
    ) -> np.array:

        tensor_approx = []
        residual = self.tensor
        for i in range(len(self.partitions)):
            partition = self.partitions[i]
            rank = self.ranks[i]
            tensorizer = Tensorization(residual)

            tensor_unf, tensor_remap = tensorizer.tensor_unfold(partition)
            tensor_unf_hat = Parafac(tensor=tensor_unf, rank=rank).optimize(max_iter=max_iter, tol=tol)
            residual_hat = tensorizer.tensor_refold(tensor_unf_hat, tensor_remap)

            tensor_approx.append(residual_hat)

            residual = residual - residual_hat
        return sum(tensor_approx)


class MrlrStochastic:
    def __init__(
        self,
        tensor: np.array,
        ranks: List[int],
        partitions: List[List[List[int]]]
    ) -> None:

        self.tensor = tensor
        self.ranks = ranks

        self.partitions = partitions

        self.tensors = []
        for partition in partitions:
            tensorizer = Tensorization(self._get_init_factor(*tensor.shape))
            tensor_unf, _ = tensorizer.tensor_unfold(partition)
            self.tensors.append(tensor_unf)

        self.sampler = Sampler(tensor.shape, len(partitions))

    def _get_init_factor(self, nrows: int, ncols: int) -> np.array:
        return np.random.rand(nrows, ncols)

    def optimize(
        self,
        max_iter: int,
        tol: float
    ) -> np.array:

        for _ in range(max_iter):
            idx_lowres_comp = self.sampler.sample_lowres_comp()
            idx_raw = self.sampler.sample_entry()

            indices = []
            for partition in self.partitions:
                indices.append(self._get_mapped_idx(idx_raw, partition))

            val = self.tensor[idx_raw]

            val_hat = 1
            for tensor, idx in zip(self.tensors, indices):
                val_hat *= tensor[idx]

            val = self.tensor[tuple(indices)]
            val_hat = self._khatri_vector(factors, indices)
            error = val - val_hat

            directions = -self._get_directions(factors, indices)
            for mode in range(len(factors)):
                gradient = alpha*error*directions[mode, :]
                factors[mode][indices[mode]] -= gradient

        tensor_approx = []
        residual = self.tensor
        for i in range(len(self.partitions)):
            partition = self.partitions[i]
            rank = self.ranks[i]
            tensorizer = Tensorization(residual)

            tensor_unf, tensor_remap = tensorizer.tensor_unfold(partition)
            tensor_unf_hat = Parafac(tensor=tensor_unf, rank=rank).optimize(max_iter=max_iter, tol=tol)
            residual_hat = tensorizer.tensor_refold(tensor_unf_hat, tensor_remap)

            tensor_approx.append(residual_hat)

            residual = residual - residual_hat
        return sum(tensor_approx)
