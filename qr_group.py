from dataclasses import dataclass, field
from gmpy2 import mpz, powmod, invert


@dataclass(frozen=True, slots=True)
class QRGroup:
    N: mpz
    half_N: mpz = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'N', mpz(self.N))
        object.__setattr__(self, 'half_N', self.N // 2)

    def abs(self, x: mpz) -> mpz:
        x = x % self.N
        if x > self.half_N:
            return self.N - x
        return x

    def mul(self, a: mpz, b: mpz) -> mpz:
        return self.abs(a * b)

    def sqr(self, a: mpz) -> mpz:
        return self.abs(powmod(a, 2, self.N))

    def pow(self, base: mpz, exp: mpz) -> mpz:
        return self.abs(powmod(base, exp, self.N))

    def inv(self, a: mpz) -> mpz:
        return self.abs(invert(a, self.N))
