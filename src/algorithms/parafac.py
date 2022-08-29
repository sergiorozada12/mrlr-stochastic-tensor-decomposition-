from typing import List, Tuple

import numpy as np
import tensorly as tl

from tensorly.base import unfold
from tensorly.tenalg import khatri_rao

from src.partitions import Sampler


class Parafac:
    def __init__(self, tensor: np.array, rank: int) -> None:
        self.tensor = tensor
        self.rank = rank

    def _get_init_factor(self, nrows: int, ncols: int) -> np.array:
        return np.random.rand(nrows, ncols)

    def optimize(
        self,
        max_iter: int,
        tol: float
    ) -> List[np.array]:

        factors = []
        for dims in self.tensor.shape:
            factors.append(self._get_init_factor(dims, self.rank))

        for _ in range(max_iter):
            for mode in range(len(factors)):
                mat = unfold(self.tensor, mode).T
                kr = khatri_rao(factors, skip_matrix=mode)
                mttkrp = kr.T @ mat
                factors[mode] = (np.linalg.inv(kr.T @ kr) @ mttkrp).T

            weights = np.ones(self.rank)
            tensor_hat = tl.cp_to_tensor((weights, factors))
            error = np.linalg.norm(self.tensor - tensor_hat)

            if error < tol:
                return tensor_hat
        return tensor_hat


class ParafacStochastic:
    def __init__(self, tensor: np.array, rank: int) -> None:
        self.tensor = tensor
        self.rank = rank
        self.sampler = Sampler(tensor.shape)

    def _get_init_factor(self, nrows: int, ncols: int) -> np.array:
        return np.random.rand(nrows, ncols)

    def _khatri_vector(self, factors: List[np.array], indices: List[int]) -> Tuple[np.array, List[np.array]]:
        vecs = []
        kr = np.ones(self.rank)
        for mode, factor in enumerate(factors):
            row = factor[indices[mode], :]

            vecs.append(row)
            kr *= row
        return kr, vecs

    def optimize(
        self,
        max_iter: int,
        alpha: float
    ) -> List[np.array]:

        factors = []
        for dims in self.tensor.shape:
            factors.append(self._get_init_factor(dims, self.rank))

        for _ in range(max_iter):
            indices = self.sampler.sample_entry()
            prod, vecs = self._khatri_vector(factors, indices)

            val = self.tensor[tuple(indices)]
            val_hat = sum(prod)
            error = val - val_hat

            for mode in range(len(factors)):
                direction = -prod/vecs[mode]
                gradient = alpha*error*direction
                factors[mode][indices[mode]] -= gradient

        weights = np.ones(self.rank)
        tensor_hat = tl.cp_to_tensor((weights, factors))
        return tensor_hat
