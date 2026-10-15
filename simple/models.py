from typing import NamedTuple
from gmpy2 import mpz


class VDFSolution(NamedTuple):
    y: mpz
    checkpoints: dict[int, mpz]


class VDFProof(NamedTuple):
    mu_values: list[mpz]
