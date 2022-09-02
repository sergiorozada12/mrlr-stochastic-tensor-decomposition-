from re import T
from typing import List, Tuple

import numpy as np
import tensorly as tl

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

        self.tensorizer = Tensorization(tensor)
        self.tensors, self.tensor_remaps = [], []
        for partition in partitions:
            tensor_unf, tensor_remap = self.tensorizer.tensor_unfold(partition)

            self.tensors.append(tensor_unf)
            self.tensor_remaps.append(tensor_remap)

        self.sampler = Sampler(self.tensor.shape, len(partitions))

    def _get_init_factor(self, nrows: int, ncols: int) -> np.array:
        return np.random.rand(nrows, ncols)

    def _get_init_factors(self, tensor: np.array, rank: int) -> List[np.array]:
        factors = []
        for dims in tensor.shape:
            factors.append(self._get_init_factor(dims, rank))
        return factors

    def _khatri_vector(self, factors: List[np.array], indices: List[int], rank: int) -> Tuple[np.array, List[np.array]]:
        vecs = []
        kr = np.ones(rank)
        for mode, factor in enumerate(factors):
            row = factor[indices[mode], :]

            vecs.append(row)
            kr *= row
        return kr, vecs

    def _get_hat_val(self, factors: List[List[np.array]], indices: List[List[int]]) -> float:
        res = 0
        for i in range(len(factors)):
            prod, _ = self._khatri_vector(factors[i], indices[i], self.ranks[i])
            res += sum(prod)
        return res

    def _get_mapped_indices(self, idx: List[int]) -> List[List[int]]:
        res = []
        for i in range(len(self.partitions)):
            partition = self.partitions[i]

            idx_remap = self.tensorizer._reindex(partition, idx)
            res.append(idx_remap)
        return res

    def _rebuild_tensor(self, factors_components: List[List[np.array]]) -> np.array:
        tensor_approx = []
        for i in range(len(factors_components)):
            factors = factors_components[i]
            tensor_remap = self.tensor_remaps[i]

            weights = np.ones(self.ranks[i])
            tensor_hat = tl.cp_to_tensor((weights, factors))
            tensor_refold = self.tensorizer.tensor_refold(tensor_hat, tensor_remap)

            tensor_approx.append(tensor_refold)
        return sum(tensor_approx)

    def optimize(
        self,
        max_iter: int,
        alpha: float
    ) -> np.array:

        factors_components = []
        for rank, tensor in zip(self.ranks, self.tensors):
            factors = self._get_init_factors(tensor, rank)
            factors_components.append(factors)

        for _ in range(max_iter):
            idx_lowres_comp = self.sampler.sample_lowres_comp()
            idx_raw = self.sampler.sample_entry()

            indices_components = self._get_mapped_indices(idx_raw)

            tensor = self.tensors[idx_lowres_comp]
            factors = factors_components[idx_lowres_comp]
            indices = indices_components[idx_lowres_comp]
            rank = self.ranks[idx_lowres_comp]

            prod, vecs = self._khatri_vector(factors, indices, rank)

            val = tensor[tuple(indices)]
            val_hat = self._get_hat_val(factors_components, indices_components)
            error = val - val_hat

            for mode in range(len(factors)):
                direction = -prod/vecs[mode]
                gradient = alpha*error*direction
                factors[mode][indices[mode]] -= gradient

        return self._rebuild_tensor(factors_components)
