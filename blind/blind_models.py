from dataclasses import dataclass
from typing import NamedTuple

from gmpy2 import mpz


@dataclass(frozen=True, slots=True)
class BasicPublicParams:
    a: mpz
    a_checkpoints: dict[int, mpz]


@dataclass(frozen=True, slots=True)
class OptimizedPublicParams:
    u: mpz
    v: mpz
    u_steps: dict[int, mpz]


@dataclass(frozen=True, slots=True)
class HybridPublicParams:
    s: int
    g: mpz
    h: mpz
    g_steps: dict[tuple[int, int], mpz]


class UnblindingState(NamedTuple):
    r: mpz


class BasicServerResponse(NamedTuple):
    y_blind: mpz
    checkpoints_blind: dict[int, mpz]


class OptimizedServerStepResponse(NamedTuple):
    pi_blind: mpz


class HybridServerStepResponse(NamedTuple):
    y_blind: mpz
    checkpoints_blind: dict[int, mpz]
