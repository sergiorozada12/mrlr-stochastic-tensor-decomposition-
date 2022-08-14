from audioop import reverse
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
                print(error)
                return tensor_hat
        print(error)
        return tensor_hat
