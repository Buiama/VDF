import os
import secrets
import multiprocessing
from multiprocessing.synchronize import Event as SynchronizeEvent
from concurrent.futures import ProcessPoolExecutor, as_completed
from gmpy2 import mpz, is_prime, next_prime

from simple.vdf import VDFGroup


def generate_base(base_bits: int) -> mpz:
    rand_val = secrets.randbits(base_bits)
    return mpz(rand_val) | 1 | (1 << (base_bits - 1))


def _safe_prime_worker(bits: int, stop_event: SynchronizeEvent) -> mpz | None:
    base_bits = bits - 1
    p_prime = generate_base(base_bits)

    while not stop_event.is_set():
        p_prime = next_prime(p_prime)
        p = 2 * p_prime + 1

        if p.bit_length() > bits:
            p_prime = generate_base(base_bits)
            continue

        if is_prime(p, 25):
            return p

    return None


def generate_safe_prime(bits: int, max_workers: int | None = None) -> mpz:
    workers = max_workers or (os.cpu_count() or 4)
    with multiprocessing.Manager() as manager:
        stop_event = manager.Event()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_safe_prime_worker, bits, stop_event) for _ in range(workers)]

            done_prime = None
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    done_prime = result
                    stop_event.set()
                    break

            if done_prime is not None:
                return done_prime
            raise RuntimeError("Failed to generate safe prime")


def generate_N(bits: int = 2048) -> mpz:
    prime_bits = bits // 2

    p = generate_safe_prime(prime_bits)
    q = generate_safe_prime(prime_bits)

    while p == q:
        q = generate_safe_prime(prime_bits)

    return p * q


def generate_x(group: VDFGroup) -> mpz:
    while True:
        z = mpz(secrets.randbelow(int(group.N - 2)) + 2)
        x = group.sqr(z)

        if x > 1:
            return x
