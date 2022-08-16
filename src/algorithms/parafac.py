from typing import List

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

    def _khatri_vector(self, factors: List[np.array], indices: List[int]) -> float:
        kr = np.ones(self.rank)
        for mode, factor in enumerate(factors):
            kr *= factor[indices[mode], :]
        return sum(kr)

    def _get_direction(self, factors: List[np.array], indices: List[int], dim: int) -> np.array:
        direction = np.ones(self.rank)
        for mode, factor in enumerate(factors):
            if dim == mode:
                continue
            direction *= factor[indices[mode], :]
        return direction

    def optimize(
        self,
        max_iter: int,
        tol: float,
        alpha: float
    ) -> List[np.array]:

        factors = []
        for dims in self.tensor.shape:
            factors.append(self._get_init_factor(dims, self.rank))

        for iteration in range(max_iter):
            indices = self.sampler.sample_entry()

            val = self.tensor[tuple(indices)]
            val_hat = self._khatri_vector(factors, indices)
            error = val - val_hat

            gradient = [np.zeros(factor.shape) for factor in factors]
            for mode in range(len(factors)):
                direction = -self._get_direction(factors, indices, mode)
                gradient[mode][indices[mode], :] = alpha*error*direction
            factors = [factors[i] - gradient[i] for i in range(len(factors))]

            if iteration % 1_000 == 0:
                weights = np.ones(self.rank)
                tensor_hat = tl.cp_to_tensor((weights, factors))
                error = np.linalg.norm(self.tensor - tensor_hat)

                if error < tol:
                    return tensor_hat
        return tensor_hat
