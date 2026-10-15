import math
import json
import time
from dataclasses import dataclass
from gmpy2 import mpz

from helpers.executors import ProcessParallelExecutor, SequentialExecutor
from helpers.qr_group import QRGroup
from helpers.utils import generate_N, generate_x
from simple.vdf import SimpleVDF
from blind.engine import BlinderEngine
from blind.basic import BasicBlindVDF
from blind.optimized import OptimizedBlindVDF
from blind.interactive import InteractiveBlindVDF
from blind.hybrid import HybridBlindVDF


@dataclass(frozen=True, slots=True)
class VDFBenchmarkResult:
    name: str
    mode: str
    blind_time: float
    solve_time: float
    prove_time: float
    unblind_time: float
    verify_time: float
    is_valid: bool
    y: mpz
    rounds: int

    @property
    def total_time(self) -> float:
        return (
                self.blind_time
                + self.solve_time
                + self.prove_time
                + self.unblind_time
                + self.verify_time
        )

    @property
    def total_without_solve_time(self) -> float:
        return self.total_time - self.solve_time


def benchmark_simple(vdf: SimpleVDF, mode: str, x: mpz, T: int, delta: int) -> VDFBenchmarkResult:
    t_start = time.perf_counter()
    sol = vdf.solve(x, T, delta)
    t_solve = time.perf_counter() - t_start

    t_start = time.perf_counter()
    proof = vdf.prove(x, sol.y, T, sol.checkpoints, delta)
    t_prove = time.perf_counter() - t_start

    t_start = time.perf_counter()
    is_valid = vdf.verify(x, sol.y, T, proof, delta)
    t_verify = time.perf_counter() - t_start

    return VDFBenchmarkResult(
        name="Simple",
        mode=mode,
        blind_time=0.0,
        solve_time=t_solve,
        prove_time=t_prove,
        unblind_time=0.0,
        verify_time=t_verify,
        is_valid=is_valid,
        y=sol.y,
        rounds=0
    )


def benchmark_basic_blind(blind_vdf: BasicBlindVDF, mode: str, x: mpz, T: int, delta: int) -> VDFBenchmarkResult:
    rtt_count = 0
    orig_solve = blind_vdf.solve

    def tracked_solve(*args, **kwargs):
        nonlocal rtt_count
        rtt_count += 1
        return orig_solve(*args, **kwargs)

    try:
        blind_vdf.solve = tracked_solve

        t_start = time.perf_counter()
        x_tilde, r, s = blind_vdf.blind(x)
        t_blind = time.perf_counter() - t_start

        t_start = time.perf_counter()
        sol = blind_vdf.solve(x_tilde, T, delta)
        t_solve = time.perf_counter() - t_start

        t_start = time.perf_counter()
        _ = blind_vdf.prove(x_tilde, sol.y_tilde, sol.checkpoints, T, delta)
        t_prove = time.perf_counter() - t_start

        t_start = time.perf_counter()
        unblind_res = blind_vdf.unblind(x, sol, r, s, delta)
        t_unblind = time.perf_counter() - t_start

        t_start = time.perf_counter()
        is_valid = blind_vdf.verify(x, unblind_res.y, T, unblind_res.proof, delta)
        t_verify = time.perf_counter() - t_start
    finally:
        blind_vdf.solve = orig_solve

    return VDFBenchmarkResult(
        name="BasicBlind",
        mode=mode,
        blind_time=t_blind,
        solve_time=t_solve,
        prove_time=t_prove,
        unblind_time=t_unblind,
        verify_time=t_verify,
        is_valid=is_valid,
        y=unblind_res.y,
        rounds=rtt_count
    )


def benchmark_optimized_blind(blind_vdf: OptimizedBlindVDF, mode: str, x: mpz, T: int, delta: int) -> VDFBenchmarkResult:
    rtt_count = 0

    orig_solve = blind_vdf.solve

    def tracked_solve(*args, **kwargs):
        nonlocal rtt_count
        rtt_count += 1
        return orig_solve(*args, **kwargs)

    blind_vdf.solve = tracked_solve

    try:
        t_start = time.perf_counter()
        x_tilde, r, s = blind_vdf.blind(x)
        t_blind = time.perf_counter() - t_start

        t_start = time.perf_counter()
        sol = blind_vdf.solve(x_tilde, T, delta)
        t_solve = time.perf_counter() - t_start

        t_start = time.perf_counter()
        _ = blind_vdf.prove(x_tilde, sol.y_tilde, sol.checkpoints, T, delta)
        t_prove = time.perf_counter() - t_start

        t_start = time.perf_counter()
        unblind_res = blind_vdf.unblind(x, sol, r, delta)
        t_unblind = time.perf_counter() - t_start

        t_start = time.perf_counter()
        is_valid = blind_vdf.verify(x, unblind_res.y, T, unblind_res.proof, delta)
        t_verify = time.perf_counter() - t_start
    finally:
        blind_vdf.solve = orig_solve

    return VDFBenchmarkResult(
        name="OptimizedBlind",
        mode=mode,
        blind_time=t_blind,
        solve_time=t_solve,
        prove_time=t_prove,
        unblind_time=t_unblind,
        verify_time=t_verify,
        is_valid=is_valid,
        y=unblind_res.y,
        rounds=rtt_count
    )


def benchmark_interactive_blind(blind_vdf: InteractiveBlindVDF, mode: str, x: mpz, T: int, delta: int) -> VDFBenchmarkResult:
    total_blind_time = 0.0
    total_solve_time = 0.0
    total_unblind_time = 0.0
    rounds_count = 0

    orig_blind = blind_vdf.blind
    orig_unblind = blind_vdf.unblind_element
    orig_solve_init = blind_vdf.solve_initial
    orig_solve_step = blind_vdf.solve_step

    def timed_blind(*args, **kwargs):
        nonlocal total_blind_time
        t0 = time.perf_counter()
        res = orig_blind(*args, **kwargs)
        total_blind_time += time.perf_counter() - t0
        return res

    def timed_unblind(*args, **kwargs):
        nonlocal total_unblind_time
        t0 = time.perf_counter()
        res = orig_unblind(*args, **kwargs)
        total_unblind_time += time.perf_counter() - t0
        return res

    def timed_solve_init(*args, **kwargs):
        nonlocal total_solve_time, rounds_count
        rounds_count += 1
        t0 = time.perf_counter()
        res = orig_solve_init(*args, **kwargs)
        total_solve_time += time.perf_counter() - t0
        return res

    def timed_solve_step(*args, **kwargs):
        nonlocal total_solve_time, rounds_count
        rounds_count += 1
        t0 = time.perf_counter()
        res = orig_solve_step(*args, **kwargs)
        total_solve_time += time.perf_counter() - t0
        return res

    blind_vdf.blind = timed_blind
    blind_vdf.unblind_element = timed_unblind
    blind_vdf.solve_initial = timed_solve_init
    blind_vdf.solve_step = timed_solve_step

    try:
        t_start = time.perf_counter()
        res = blind_vdf.prove_interactive(x=x, T=T, delta=delta)
        t_interactive_total = time.perf_counter() - t_start
    finally:
        blind_vdf.blind = orig_blind
        blind_vdf.unblind_element = orig_unblind
        blind_vdf.solve_initial = orig_solve_init
        blind_vdf.solve_step = orig_solve_step

    t_prove_overhead = max(0.0, t_interactive_total - total_solve_time - total_blind_time - total_unblind_time)

    t_start = time.perf_counter()
    is_valid = blind_vdf.verify(x, res.y, T, res.proof, delta)
    t_verify = time.perf_counter() - t_start

    return VDFBenchmarkResult(
        name="InteractiveBlind",
        mode=mode,
        blind_time=total_blind_time,
        solve_time=total_solve_time,
        prove_time=t_prove_overhead,
        unblind_time=total_unblind_time,
        verify_time=t_verify,
        is_valid=is_valid,
        y=res.y,
        rounds=rounds_count
    )


def benchmark_hybrid_blind(
        blind_vdf: HybridBlindVDF, mode: str, x: mpz, T: int, delta: int, s_rounds: int | None = None
) -> VDFBenchmarkResult:
    total_blind_time = 0.0
    total_solve_time = 0.0
    total_unblind_time = 0.0
    rounds_count = 0

    orig_blind = blind_vdf.blind
    orig_solve_round = blind_vdf.solve_round
    orig_unblind = blind_vdf.unblind_checkpoints_for_step

    def timed_blind(*args, **kwargs):
        nonlocal total_blind_time
        t0 = time.perf_counter()
        res = orig_blind(*args, **kwargs)
        total_blind_time += time.perf_counter() - t0
        return res

    def timed_solve_round(*args, **kwargs):
        nonlocal total_solve_time, rounds_count
        rounds_count += 1
        t0 = time.perf_counter()
        res = orig_solve_round(*args, **kwargs)
        total_solve_time += time.perf_counter() - t0
        return res

    def timed_unblind(*args, **kwargs):
        nonlocal total_unblind_time
        t0 = time.perf_counter()
        res = orig_unblind(*args, **kwargs)
        total_unblind_time += time.perf_counter() - t0
        return res

    blind_vdf.blind = timed_blind
    blind_vdf.solve_round = timed_solve_round
    blind_vdf.unblind_checkpoints_for_step = timed_unblind

    try:
        t_start = time.perf_counter()
        res = blind_vdf.prove_hybrid(x=x, T=T, delta=delta, s_rounds=s_rounds)
        t_hybrid_total = time.perf_counter() - t_start
    finally:
        blind_vdf.blind = orig_blind
        blind_vdf.solve_round = orig_solve_round
        blind_vdf.unblind_checkpoints_for_step = orig_unblind

    t_prove_overhead = max(0.0, t_hybrid_total - total_solve_time - total_blind_time - total_unblind_time)

    t_start = time.perf_counter()
    is_valid = blind_vdf.verify(x, res.y, T, res.proof, delta)
    t_verify = time.perf_counter() - t_start

    return VDFBenchmarkResult(
        name="HybridBlind(4)" if s_rounds == 4 else "HybridBlind(Auto)",
        mode=mode,
        blind_time=total_blind_time,
        solve_time=total_solve_time,
        prove_time=t_prove_overhead,
        unblind_time=total_unblind_time,
        verify_time=t_verify,
        is_valid=is_valid,
        y=res.y,
        rounds=rounds_count
    )


def print_result_row(T_str: str, res: VDFBenchmarkResult) -> None:
    blind_str = f"{res.blind_time:<10.4f}" if res.blind_time > 0 else f"{'-':<10}"
    unblind_str = f"{res.unblind_time:<10.4f}" if res.unblind_time > 0 else f"{'-':<10}"
    prove_str = f"{res.prove_time:<10.4f}" if res.prove_time > 0 else f"{'-':<10}"

    print(
        f"{T_str:<8} | {res.name:<17}| {res.mode:<4} | {res.rounds:<6} | {blind_str} | {res.solve_time:<10.4f} | "
        f"{prove_str} | {unblind_str}  | {res.verify_time:<10.4f} | "
        f"{res.total_time:<10.4f} | {res.total_without_solve_time:<12.4f} | {res.is_valid}"
    )


def format_t(T: int) -> str:
    k = round(math.log2(T))
    diff = T - (1 << k)
    if diff == 0:
        return f"2^{k}"
    sign = "+" if diff > 0 else "-"
    return f"2^{k} {sign} {abs(diff)}"


def benchmark_result_to_dict(res: VDFBenchmarkResult) -> dict:
    return {
        "name": res.name,
        "mode": res.mode,
        "rounds": res.rounds,
        "blind_time": res.blind_time,
        "solve_time": res.solve_time,
        "prove_time": res.prove_time,
        "unblind_time": res.unblind_time,
        "verify_time": res.verify_time,
        "total_time": res.total_time,
        "total_without_solve_time": res.total_without_solve_time,
        "is_valid": res.is_valid,
        "y": str(res.y),
    }


def run_blind_benchmarks(output_json_path: str = "vdf_benchmarks.json") -> None:
    print("Generating 2048-bit RSA modulus N...")
    t0 = time.perf_counter()

    N = generate_N(bits=2048)
    group = QRGroup(N)
    x = generate_x(group)
    g = generate_x(group)

    bench_domain_sep = b"BENCHMARK_VDF"
    t_group_setup = time.perf_counter() - t0

    print(f"Group setup completed in {t_group_setup:.2f} seconds.\n")

    t_values = [
        2 ** 17 - 1, 2 ** 17, 2 ** 17 + 1,
        2 ** 18 - 1, 2 ** 18, 2 ** 18 + 1,
        2 ** 20 - 1, 2 ** 20, 2 ** 20 + 1,
        # 2 ** 24 - 1, 2 ** 24, 2 ** 24 + 1,
        # 2 ** 27 - 1, 2 ** 27, 2 ** 27 + 1,
        # 2 ** 29 - 1, 2 ** 29, 2 ** 29 + 1,
        # 2**27,
        # 2**29
    ]

    header = (
        f"{'T':<8} | {'Impl':<17}| {'Mode':<4} | {'Rounds':<6} | {'Blind (s)':<10} | {'Solve (s)':<10} | "
        f"{'Prove (s)':<10} | {'Unblind (s)':<10} | {'Verify (s)':<10} | {'Total (s)':<10} | {'W/o Solve(s)':<12} | "
        f"{'Valid'}"
    )

    benchmark_data = {
        "setup": {
            "group_setup_time_s": t_group_setup,
            "N_bits": 2048,
        },
        "benchmarks": []
    }

    with ProcessParallelExecutor(N=group.N, max_workers=8, threshold=12) as parallel_executor:
        seq_executor = SequentialExecutor()

        simple_vdf_seq = SimpleVDF(group=group, challenge_bits=128, domain_separator=bench_domain_sep,
                                   executor=seq_executor)
        simple_vdf_par = SimpleVDF(group=group, challenge_bits=128, domain_separator=bench_domain_sep,
                                   executor=parallel_executor)

        blinder_engine_seq = BlinderEngine(vdf=simple_vdf_seq)

        print(header)
        print("-" * len(header))

        for T in t_values:
            delta = max(1, math.isqrt(T))

            t_setup_start = time.perf_counter()
            setup = blinder_engine_seq.create_setup(g, T, delta)
            t_setup = time.perf_counter() - t_setup_start

            basic_seq = BasicBlindVDF(vdf=simple_vdf_seq, setup=setup, domain_separator=bench_domain_sep)
            # opt_seq = OptimizedBlindVDF(vdf=simple_vdf_seq, setup=setup, domain_separator=bench_domain_sep)
            interactive_seq = InteractiveBlindVDF(vdf=simple_vdf_seq, setup=setup, domain_separator=bench_domain_sep)
            hybrid_seq = HybridBlindVDF(vdf=simple_vdf_seq, setup=setup, domain_separator=bench_domain_sep)

            basic_par = BasicBlindVDF(vdf=simple_vdf_par, setup=setup, domain_separator=bench_domain_sep)
            # opt_par = OptimizedBlindVDF(vdf=simple_vdf_par, setup=setup, domain_separator=bench_domain_sep)
            interactive_par = InteractiveBlindVDF(vdf=simple_vdf_par, setup=setup, domain_separator=bench_domain_sep)
            hybrid_par = HybridBlindVDF(vdf=simple_vdf_par, setup=setup, domain_separator=bench_domain_sep)

            res_simple_seq = benchmark_simple(simple_vdf_seq, "Seq", x, T, delta)
            res_basic_seq = benchmark_basic_blind(basic_seq, "Seq", x, T, delta)
            # res_opt_seq = benchmark_optimized_blind(opt_seq, "Seq", x, T, delta)
            res_interactive_seq = benchmark_interactive_blind(interactive_seq, "Seq", x, T, delta)
            res_hybrid_auto_seq = benchmark_hybrid_blind(hybrid_seq, "Seq", x, T, delta)

            res_simple_par = benchmark_simple(simple_vdf_par, "Par", x, T, delta)
            res_basic_par = benchmark_basic_blind(basic_par, "Par", x, T, delta)
            # res_opt_par = benchmark_optimized_blind(opt_par, "Par", x, T, delta)
            res_interactive_par = benchmark_interactive_blind(interactive_par, "Par", x, T, delta)
            res_hybrid_auto_par = benchmark_hybrid_blind(hybrid_par, "Par", x, T, delta)

            results_list = [
                res_simple_seq, res_basic_seq, res_interactive_seq, res_hybrid_auto_seq,
                res_simple_par, res_basic_par, res_interactive_par, res_hybrid_auto_par
            ]
            first_y = results_list[0].y

            assert all(r.y == first_y for r in results_list), f"Output mismatch between modes for T={T}!"
            assert all(r.is_valid for r in results_list), f"Proof verification failed for T={T}!"

            t_str = format_t(T)
            print(f"[Setup T={T} ({t_str}) completed in {t_setup:.4f}s]")

            for seq_res, par_res in [
                (res_simple_seq, res_simple_par),
                (res_basic_seq, res_basic_par),
                (res_interactive_seq, res_interactive_par),
                (res_hybrid_auto_seq, res_hybrid_auto_par)
            ]:
                print_result_row(str(T), seq_res)
                print_result_row("", par_res)

            print("-" * len(header))

            t_entry = {
                "T": T,
                "T_formatted": t_str,
                "setup_time_s": t_setup,
                "results": [benchmark_result_to_dict(r) for r in results_list]
            }
            benchmark_data["benchmarks"].append(t_entry)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    run_blind_benchmarks()
