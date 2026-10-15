import pickle
import time
from pathlib import Path

from helpers.qr_group import QRGroup
from simple.vdf import SimpleVDF

CACHE_FILE = Path("../vdf_precomputed_data_lite26.pkl")


def run_benchmarks():
    if not CACHE_FILE.exists():
        print(f"Error: Cache file '{CACHE_FILE}' not found!")
        return

    print(f"Loading precomputed VDF data from {CACHE_FILE}")
    t0 = time.perf_counter()
    with open(CACHE_FILE, "rb") as f:
        dataset = pickle.load(f)

    N = dataset["N"]
    group = QRGroup(N)
    x = dataset["x"]
    vdf = SimpleVDF(group, challenge_bits=128)
    print(f"Setup completed in {time.perf_counter() - t0:.2f} seconds.\n")

    # t_values = [2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 24, 2 ** 26, 2 ** 28, 2**30]  #

    print(f"{'T':<12} | {'Solve (s)':<12} | {'Prove (s)':<12} | {'Verify (s)':<12} | {'Proof Len'}")
    print("-" * 70)

    for T, data in dataset["benchmarks"].items():
        delta = data["delta"]
        y = data["y"]
        checkpoints = data["checkpoints"]
        t_solve = data["t_solve"]

        t_start = time.perf_counter()
        proof = vdf.prove(x, y, T, checkpoints, delta)
        t_prove = time.perf_counter() - t_start

        t_start = time.perf_counter()
        is_valid = vdf.verify(x, y, T, proof, delta)
        t_verify = time.perf_counter() - t_start

        assert is_valid, f"Error: Proof verification failed for T={T}!"

        print(f"{T:<12} | {t_solve:<12.4f} | {t_prove:<12.4f} | {t_verify:<12.4f} | {len(proof.mu_values)}")


if __name__ == "__main__":
    run_benchmarks()
