from typing import List

import numpy as np

from src.algorithms.parafac import Parafac
from src.partitions import Tensorization


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
