import math
import time

from qr_group import QRGroup
from utils import generate_N, generate_x
from vdf import SimpleVDF


def run_benchmarks():
    print("Generating 2048-bit RSA modulus N")
    t0 = time.perf_counter()
    N = generate_N(bits=2048)
    group = QRGroup(N)
    x = generate_x(group)
    vdf = SimpleVDF(group, challenge_bits=128)
    print(f"Setup completed in {time.perf_counter() - t0:.2f} seconds.\n")

    t_values = [2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 24, 2 ** 26, 2 ** 28]  # , 2**30

    print(f"{'T':<12} | {'Solve (s)':<12} | {'Prove (s)':<12} | {'Verify (s)':<12} | {'Proof Len'}")
    print("-" * 70)

    for T in t_values:
        delta = max(1, math.isqrt(T))

        t_start = time.perf_counter()
        y, checkpoints = vdf.solve(x, T, delta=delta)
        t_solve = time.perf_counter() - t_start

        t_start = time.perf_counter()
        proof = vdf.prove(x, y, T, checkpoints, delta=delta)
        t_prove = time.perf_counter() - t_start

        t_start = time.perf_counter()
        is_valid = vdf.verify(x, y, T, proof)
        t_verify = time.perf_counter() - t_start

        assert is_valid, f"Error: Proof verification failed for T={T}!"

        print(f"{T:<12} | {t_solve:<12.4f} | {t_prove:<12.4f} | {t_verify:<12.4f} | {len(proof.mu_values)}")


if __name__ == "__main__":
    run_benchmarks()
