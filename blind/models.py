from dataclasses import dataclass
from gmpy2 import mpz

from simple.models import VDFProof


@dataclass(frozen=True, slots=True)
class BlindSetup:
    g: mpz
    h: mpz
    T: int
    blinders: dict[int, mpz]


@dataclass(frozen=True, slots=True)
class BlindState:
    r: mpz
    r_T: mpz
    T: int
    blinders: dict[int, mpz]


@dataclass(frozen=True, slots=True)
class BlindedSolution:
    y_tilde: mpz
    checkpoints: dict[int, mpz]


@dataclass(frozen=True, slots=True)
class BlindResult:
    y: mpz
    proof: VDFProof


@dataclass(frozen=True, slots=True)
class HybridBlindSetup:
    g: mpz
    T: int
    s: int
    blinders: dict[int, mpz]