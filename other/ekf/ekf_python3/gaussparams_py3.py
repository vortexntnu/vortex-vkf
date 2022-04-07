from dataclasses import dataclass
import numpy as np
from numpy import linalg as nla, ndarray

from utils.debugtools import isPSD
from config import DEBUG


@dataclass(frozen=True)
class MultiVarGaussian:
    """A class for using Gaussians"""
    mean: ndarray  # shape=(n,)
    cov: ndarray  # shape=(n, n)

    def __post_init__(self):
        # This is only for
        if __debug__:
            if DEBUG:
                if not self.mean.shape * 2 == self.cov.shape:
                    raise ValueError(f"mean has shape {self.mean.shape} "
                                     "while cov has shape {self.cov.shape}")
                if not np.all(np.isfinite(self.mean)):
                    raise ValueError(f"Non finite mean = {self.mean}")
                if not np.all(np.isfinite(self.cov)):
                    raise ValueError(f"Non finite cov = {self.cov}")
                if not isPSD(self.cov):
                    raise ValueError(f"Not PSD cov = {self.cov}.")
        else:
            pass

    @property
    def ndim(self) -> int:
        return self.mean.shape[0]

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """Calculate the mahalanobis distance between self and x.

        This is also known as the quadratic form of the Gaussian.
        See (3.2) in the book.
        """
        # this method could be vectorized for efficient calls
        error = x - self.mean
        mahalanobis_distance = error.T @ nla.solve(self.cov, error)
        return mahalanobis_distance

    def pdf(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def logpdf(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def gate(self, x: np.ndarray, prob: float) -> bool:
        raise NotImplementedError

    def __iter__(self):  # in order to use tuple unpacking
        return iter((self.mean, self.cov))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, MultiVarGaussian):
            return False
        else:
            return (np.allclose(self.mean, o.mean) and
                    np.allclose(self.cov, o.cov))