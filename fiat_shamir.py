import hashlib
from dataclasses import dataclass, field, InitVar
from typing import Protocol

from gmpy2 import mpz


class ShakeHash(Protocol):
    def update(self, data: bytes) -> None: ...
    def digest(self, length: int) -> bytes: ...
    def copy(self) -> 'ShakeHash': ...


@dataclass(frozen=True, slots=True)
class FiatShamirTranscript:
    N: InitVar[mpz]
    x0: InitVar[mpz]
    y0: InitVar[mpz]
    T0: InitVar[int]
    challenge_bits: int = 128

    mod_bytes: int = field(init=False)
    challenge_bytes: int = field(init=False)
    challenge_mask: int = field(init=False)
    _base_hash: ShakeHash = field(init=False)

    def __post_init__(self, N: mpz, x0: mpz, y0: mpz, T0: int):
        object.__setattr__(self, 'mod_bytes', (N.bit_length() + 7) // 8)
        object.__setattr__(self, 'challenge_bytes', (self.challenge_bits + 7) // 8)
        object.__setattr__(self, 'challenge_mask', (1 << self.challenge_bits) - 1)

        base_hash = hashlib.shake_256()
        base_hash.update(b"Simple_VDF")
        base_hash.update(self._to_bytes(N))
        base_hash.update(self._to_bytes(x0))
        base_hash.update(self._to_bytes(y0))
        base_hash.update(T0.to_bytes(8, 'big'))

        object.__setattr__(self, '_base_hash', base_hash)

    def _to_bytes(self, val: mpz) -> bytes:
        return int(val).to_bytes(self.mod_bytes, 'big')

    def get_challenge(self, x_i: mpz, T_i: int, y_i: mpz, mu_i: mpz) -> mpz:
        h = self._base_hash.copy()

        h.update(self._to_bytes(x_i))
        h.update(T_i.to_bytes(8, 'big'))
        h.update(self._to_bytes(y_i))
        h.update(self._to_bytes(mu_i))

        raw_bytes = h.digest(self.challenge_bytes)
        r = int.from_bytes(raw_bytes, 'big') & self.challenge_mask

        return mpz(r)
