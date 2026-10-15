from gmpy2 import mpz

from typing import Protocol
from bisect import bisect_right
from dataclasses import dataclass, field
from helpers.fiat_shamir import FiatShamirTranscript
from simple.models import VDFSolution, VDFProof
from helpers.executors import Executor, SequentialExecutor, get_worker_group


class VDFGroup(Protocol):
    N: mpz

    def abs(self, a: mpz) -> mpz: ...
    def sqr(self, a: mpz) -> mpz: ...
    def raw_sqr(self, a: mpz) -> mpz: ...
    def mul(self, a: mpz, b: mpz) -> mpz: ...
    def pow(self, base: mpz, exp: mpz) -> mpz: ...
    def inv(self, a: mpz) -> mpz: ...


def split_T(T: int, delta: int) -> tuple[int, int]:
    if T >= 2 * delta:
        T_left = (T // (2 * delta)) * delta
    else:
        T_left = T // 2

    return T_left, T - T_left


def get_intermediate_value(checkpoints: dict[int, mpz], target_i: int, delta: int, group: VDFGroup) -> mpz:
    if target_i in checkpoints:
        return checkpoints[target_i]

    base_i = (target_i // delta) * delta

    if base_i not in checkpoints:
        sorted_keys = sorted(checkpoints.keys())
        idx = bisect_right(sorted_keys, target_i) - 1
        base_i = sorted_keys[idx]

    current = checkpoints[base_i]

    sqr = group.raw_sqr
    for _ in range(target_i - base_i):
        current = sqr(current)

    return group.abs(current)


def compute_next_checkpoint(checkpoints: dict[int, mpz], idx: int, T_left: int, r: mpz, delta: int, group: VDFGroup
                            ) -> mpz:
    val_idx = get_intermediate_value(checkpoints, idx, delta, group)
    val_mid_idx = get_intermediate_value(checkpoints, T_left + idx, delta, group)
    term = group.pow(val_idx, r)

    return group.mul(term, val_mid_idx)


def _reduce_batch_worker(
    chunk_indices: list[int],
    extra_args: tuple[dict[int, mpz], int, mpz, int]
) -> list[tuple[int, mpz]]:
    checkpoints, T_left, r, delta = extra_args
    group = get_worker_group()

    res = []
    for idx in chunk_indices:
        res_val = compute_next_checkpoint(checkpoints, idx, T_left, r, delta, group)
        res.append((idx, res_val))
    return res


@dataclass(frozen=True, slots=True)
class SimpleVDF:
    group: VDFGroup
    domain_separator: bytes = b"Simple_VDF"
    challenge_bits: int = 128
    executor: Executor = field(default_factory=SequentialExecutor)

    def solve(self, x: mpz, T: int, delta: int) -> VDFSolution:
        checkpoints = {0: x}
        curr = x

        sqr = self.group.raw_sqr
        normalize = self.group.abs
        for block_start in range(0, T, delta):
            block_end = min(block_start + delta, T)

            for _ in range(block_start, block_end):
                curr = sqr(curr)

            curr = normalize(curr)
            checkpoints[block_end] = curr

        return VDFSolution(y=curr, checkpoints=checkpoints)

    def evaluate_squarings(self, x: mpz, steps: int) -> mpz:
        sqr = self.group.raw_sqr
        for _ in range(steps):
            x = sqr(x)
        return self.group.abs(x)

    def get_intermediate_value(self, checkpoints: dict[int, mpz], target_i: int, delta: int) -> mpz:
        return get_intermediate_value(checkpoints, target_i, delta, self.group)

    def compute_next_state(self, x: mpz, y: mpz, mu: mpz, r: mpz, T_left: int, T_right: int) -> tuple[mpz, mpz]:
        x_pow_r = self.group.pow(x, r)
        exp_y = r << (T_right - T_left)
        mu_pow_for_y = self.group.pow(mu, exp_y)

        next_x = self.group.mul(x_pow_r, mu)
        next_y = self.group.mul(mu_pow_for_y, y)

        return next_x, next_y

    def _compute_next_checkpoint(self, checkpoints: dict[int, mpz], idx: int, T_left: int, r: mpz, delta: int) -> mpz:
        return compute_next_checkpoint(checkpoints, idx, T_left, r, delta, self.group)

    def reduce_checkpoints(
            self, checkpoints: dict[int, mpz], T_left: int, T_right: int, r: mpz, delta: int
    ) -> dict[int, mpz]:
        indices = list(range(0, T_right + 1, delta))
        if T_right % delta != 0 and T_right not in indices:
            indices.append(T_right)

        extra_args = (checkpoints, T_left, r, delta)

        batch_results = self.executor.run_batch(indices, _reduce_batch_worker, extra_args)

        return dict(batch_results)

    def prove(self, x: mpz, y: mpz, T: int, checkpoints: dict[int, mpz], delta: int) -> VDFProof:
        proof = []
        current_checkpoints = checkpoints
        fiat_shamir = FiatShamirTranscript(self.group.N, x, y, T, self.domain_separator, self.challenge_bits)

        while T > 1:
            T_left, T_right = split_T(T, delta)

            mu = self.get_intermediate_value(current_checkpoints, T_left, delta)
            proof.append(mu)

            r = fiat_shamir.get_challenge(x, T, y, mu)
            current_checkpoints = self.reduce_checkpoints(current_checkpoints, T_left, T_right, r, delta)
            x, y = self.compute_next_state(x, y, mu, r, T_left, T_right)

            T = T_right

        return VDFProof(mu_values=proof)

    def verify(self, x: mpz, y: mpz, T: int, proof: VDFProof, delta: int) -> bool:
        proof_iter = iter(proof.mu_values)
        fiat_shamir = FiatShamirTranscript(self.group.N, x, y, T, self.domain_separator, self.challenge_bits)

        while T > 1:
            T_left, T_right = split_T(T, delta)

            try:
                mu = next(proof_iter)
            except StopIteration:
                return False

            r = fiat_shamir.get_challenge(x, T, y, mu)
            x, y = self.compute_next_state(x, y, mu, r, T_left, T_right)

            T = T_right

        expected_y = self.group.sqr(x)

        return expected_y == y
