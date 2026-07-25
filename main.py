import math
import time

from blind.blind_vdf_basic import (
    BasicBlindVDFSetup,
    BasicBlindVDFClient,
    BasicBlindVDFServer,
)
from blind.blind_vdf_hybrid import (
    HybridBlindVDFSetup,
    HybridBlindVDFServer,
    HybridBlindVDFSession,
)
from blind.blind_vdf_optimized import (
    OptimizedBlindVDFSetup,
    OptimizedBlindVDFServer,
    OptimizedBlindVDFSession,
)

from qr_group import QRGroup
from utils import generate_N, generate_x
from vdf import SimpleVDF


def run_simple_vdf(vdf, x, T):
    delta = max(1, math.isqrt(T))

    t0 = time.perf_counter()
    y, checkpoints = vdf.solve(x, T, delta=delta)
    t_server = time.perf_counter() - t0

    t0 = time.perf_counter()
    proof = vdf.prove(x, y, T, checkpoints, delta=delta)
    t_client = time.perf_counter() - t0

    t0 = time.perf_counter()
    is_valid = vdf.verify(x, y, T, proof)
    t_verify = time.perf_counter() - t0

    assert is_valid, "SimpleVDF verification failed!"
    return t_server, t_client, t_verify, len(proof.mu_values)


def run_basic_blind_vdf(group, vdf, x, T):
    setup_runner = BasicBlindVDFSetup(group, vdf)
    server_runner = BasicBlindVDFServer(vdf)

    t0 = time.perf_counter()
    params = setup_runner.setup(T)
    t_setup = time.perf_counter() - t0

    client_runner = BasicBlindVDFClient(group, params, vdf)
    secret_r = generate_x(group)

    t0 = time.perf_counter()
    x_blind, state = client_runner.blind(x, secret_r)
    t_client_blind = time.perf_counter() - t0

    t0 = time.perf_counter()
    resp = server_runner.solve_blind(x_blind, T)
    t_server = time.perf_counter() - t0

    t0 = time.perf_counter()
    y, proof = client_runner.unblind_and_prove(x, T, state, resp)
    t_client_unblind_prove = time.perf_counter() - t0

    t_client = t_client_blind + t_client_unblind_prove

    t0 = time.perf_counter()
    is_valid = vdf.verify(x, y, T, proof)
    t_verify = time.perf_counter() - t0

    assert is_valid, "BasicVDF verification failed!"
    return t_setup, t_server, t_client, t_verify, len(proof.mu_values)


def run_optimized_blind_vdf(group, vdf, x, T):
    t_bits = int(math.log2(T))
    setup_runner = OptimizedBlindVDFSetup(group, vdf)
    server_runner = OptimizedBlindVDFServer(group, vdf)

    t0 = time.perf_counter()
    params = setup_runner.setup(T)
    t_setup = time.perf_counter() - t0

    session = OptimizedBlindVDFSession(group, params, vdf)
    r_list = [generate_x(group) for _ in range(t_bits)]

    t0 = time.perf_counter()
    y, proof = session.run_interactive_protocol(x, T, r_list, server_runner)
    t_protocol = time.perf_counter() - t0

    t0 = time.perf_counter()
    is_valid = vdf.verify(x, y, T, proof)
    t_verify = time.perf_counter() - t0

    assert is_valid, "OptimizedVDF verification failed!"
    return t_setup, t_protocol, t_verify, len(proof.mu_values)


def run_hybrid_blind_vdf(group, vdf, x, T, s):
    setup_runner = HybridBlindVDFSetup(group, vdf)
    server_runner = HybridBlindVDFServer(vdf)

    t0 = time.perf_counter()
    params = setup_runner.setup(T, s)
    t_setup = time.perf_counter() - t0

    session = HybridBlindVDFSession(group, params, vdf)
    r_list = [generate_x(group) for _ in range(s)]

    t0 = time.perf_counter()
    y, proof = session.run_hybrid_protocol(x, T, r_list, server_runner)
    t_protocol = time.perf_counter() - t0

    t0 = time.perf_counter()
    is_valid = vdf.verify(x, y, T, proof)
    t_verify = time.perf_counter() - t0

    assert is_valid, "HybridVDF verification failed!"
    return t_setup, t_protocol, t_verify, len(proof.mu_values)


def run_benchmarks():
    print("Generating 2048-bit RSA modulus N...")
    t0 = time.perf_counter()
    N = generate_N(bits=2048)
    group = QRGroup(N)
    x = generate_x(group)
    vdf = SimpleVDF(group, challenge_bits=128)
    print(f"Setup completed in {time.perf_counter() - t0:.2f} seconds.\n")

    t_values = [2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 24, 2 ** 26]

    for T in t_values:
        t_bits = int(math.log2(T))
        divisors = [i for i in range(2, t_bits) if t_bits % i == 0]

        if divisors:
            s_param = divisors[len(divisors) // 2]
        else:
            s_param = 0

        print(f"\nT = {T} (2^{t_bits}) <<<")

        s_t_serv, s_t_cli, s_t_ver, s_mus = run_simple_vdf(vdf, x, T)
        b_t_set, b_t_serv, b_t_cli, b_t_ver, b_mus = run_basic_blind_vdf(group, vdf, x, T)
        o_t_set, o_t_proto, o_t_ver, o_mus = run_optimized_blind_vdf(group, vdf, x, T)

        if t_bits % s_param == 0:
            h_t_set, h_t_proto, h_t_ver, h_mus = run_hybrid_blind_vdf(group, vdf, x, T, s_param)
        else:
            h_t_set = h_t_proto = h_t_ver = 0.0
            h_mus = 0

        header = f"{'Protocol':<20} | {'Setup (s)':<10} | {'Eval/Server':<11} | {'Client/Prove':<12} | {'Verify (s)':<10} | {'Mus'}"
        divider = "-" * len(header)

        print(divider)
        print(header)
        print(divider)
        print(f"{'Simple VDF':<20} | {'N/A':<10} | {s_t_serv:<11.4f} | {s_t_cli:<12.4f} | {s_t_ver:<10.4f} | {s_mus}")
        print(f"{'Basic Blind VDF':<20} | {b_t_set:<10.4f} | {b_t_serv:<11.4f} | {b_t_cli:<12.4f} | {b_t_ver:<10.4f} | {b_mus}")
        print(f"{'Optimized Blind VDF':<20} | {o_t_set:<10.4f} | {o_t_proto:<11.4f} | {'Interactive':<12} | {o_t_ver:<10.4f} | {o_mus}")
        if t_bits % s_param == 0:
            print(f"{'Hybrid Blind VDF':<20} | {h_t_set:<10.4f} | {h_t_proto:<11.4f} | {'Interactive':<12} | {h_t_ver:<10.4f} | {h_mus}")
        else:
            print(f"{'Hybrid Blind VDF':<20} | {'N/A':<10} | {'Skipped':<11} | {'Skipped':<12} | {'Skipped':<10} | N/A")
        print(divider)


if __name__ == "__main__":
    run_benchmarks()
