import secrets
from gmpy2 import mpz, is_prime, next_prime

from qr_group import QRGroup


def generate_base(base_bits: int) -> mpz:
    rand_val = secrets.randbits(base_bits)
    return mpz(rand_val) | 1 | (1 << (base_bits - 1))


def generate_safe_prime(bits: int) -> mpz:
    base_bits = bits - 1
    p_prime = generate_base(base_bits)

    while True:
        p_prime = next_prime(p_prime)
        p = 2 * p_prime + 1

        if p.bit_length() > bits:
            p_prime = generate_base(base_bits)
            continue

        if is_prime(p, 25):
            return p


def generate_N(bits: int = 2048) -> mpz:
    prime_bits = bits // 2

    p = generate_safe_prime(prime_bits)
    q = generate_safe_prime(prime_bits)

    while p == q:
        q = generate_safe_prime(prime_bits)

    return p * q


def generate_x(group: QRGroup) -> mpz:
    while True:
        z = mpz(secrets.randbelow(int(group.N - 2)) + 2)
        x = group.sqr(z)

        if x > 1:
            return x
