import math
from dataclasses import replace
from typing import Any

from gmpy2 import mpz

from blind.engine import BlinderEngine
from blind.models import BlindSetup, BlindResult
from helpers.fiat_shamir import FiatShamirTranscript
from helpers.executors import get_worker_group
from simple.models import VDFProof
from simple.vdf import SimpleVDF, split_T, get_intermediate_value


def estimate_optimal_s_rounds(T: int, target_checkpoints_per_round: int = 32) -> int:
    total_bits = max(1, T.bit_length() - 1)
    target_depth = max(1, target_checkpoints_per_round.bit_length() - 1)
    return max(1, math.ceil(total_bits / target_depth))


def _unblind_step_batch_worker(
        chunk_steps: list[int],
        extra_args: tuple[dict[int, mpz], dict[int, mpz], mpz, int]
) -> list[tuple[int, mpz]]:
    checkpoints, blinders, s, delta = extra_args
    group = get_worker_group()

    pow_mod = group.pow
    inv_mod = group.inv
    mul_mod = group.mul

    res = []
    for step in chunk_steps:
        c_tilde = checkpoints[step]
        b_step = get_intermediate_value(blinders, step, delta, group)
        r_term = pow_mod(b_step, s)
        res.append((step, mul_mod(c_tilde, inv_mod(r_term))))
    return res


class HybridBlindVDF:
    def __init__(self, vdf: SimpleVDF, setup: BlindSetup, domain_separator: bytes = b"HybridBlind_VDF"):
        self.vdf = replace(vdf, domain_separator=domain_separator)
        self.setup = setup
        self.blinder = BlinderEngine(self.vdf)

    def blind(self, x: mpz) -> tuple[mpz, mpz, mpz]:
        return self.blinder.blind(self.setup, x)

    def unblind_checkpoints_for_step(self, checkpoints: dict[int, mpz], s: mpz, delta: int) -> dict[int, mpz]:
        steps = list(checkpoints.keys())
        extra_args = (checkpoints, self.setup.blinders, s, delta)

        batch_results = self.vdf.executor.run_batch(steps, _unblind_step_batch_worker, extra_args)
        return dict(batch_results)

    def solve_round(self, x_tilde: mpz, T: int, step_delta: int) -> dict[int, mpz]:
        sol = self.vdf.solve(x_tilde, T, step_delta)
        return sol.checkpoints

    def prove_hybrid(self, x: mpz, T: int, delta: int, s_rounds: int | None = None) -> BlindResult:
        if s_rounds is None:
            s_rounds = estimate_optimal_s_rounds(T, target_checkpoints_per_round=32)

        total_bits = max(1, T.bit_length() - 1)
        depth_per_round = max(1, math.ceil(total_bits / max(1, s_rounds)))

        curr_x = x
        curr_T = T
        final_y: mpz = mpz(0)
        proof_mus: list[mpz] = []

        fiat_shamir: FiatShamirTranscript | None = None

        while curr_T > 1:
            curr_bits = max(1, curr_T.bit_length() - 1)
            depth_round = min(depth_per_round, curr_bits)

            raw_step_delta = curr_T >> depth_round
            step_delta = max(delta, (raw_step_delta // delta) * delta)

            x_tilde, _, s_k = self.blind(curr_x)

            round_checkpoints_tilde = self.solve_round(x_tilde, curr_T, step_delta)

            round_checkpoints = self.unblind_checkpoints_for_step(round_checkpoints_tilde, s_k, delta)

            curr_y = round_checkpoints[curr_T]

            if fiat_shamir is None:
                final_y = curr_y
                fiat_shamir = FiatShamirTranscript(
                    self.vdf.group.N, x, final_y, T, self.vdf.domain_separator, self.vdf.challenge_bits
                )

            halving_step = 0
            while halving_step < depth_round and curr_T > 1:
                T_left, T_right = split_T(curr_T, delta)

                mu = self.vdf.get_intermediate_value(round_checkpoints, T_left, step_delta)
                proof_mus.append(mu)

                r_challenge = fiat_shamir.get_challenge(curr_x, curr_T, curr_y, mu)

                round_checkpoints = self.vdf.reduce_checkpoints(
                    round_checkpoints, T_left, T_right, r_challenge, step_delta
                )
                curr_x, curr_y = self.vdf.compute_next_state(curr_x, curr_y, mu, r_challenge, T_left, T_right)

                curr_T = T_right
                halving_step += 1

        return BlindResult(y=final_y, proof=VDFProof(mu_values=proof_mus))

    def verify(self, x: mpz, y: mpz, T: int, proof: VDFProof, delta: int) -> bool:
        return self.vdf.verify(x, y, T, proof, delta)
