from gmpy2 import mpz

from typing import Protocol
from dataclasses import dataclass
from fiat_shamir import FiatShamirTranscript
from models import VDFSolution, VDFProof


class VDFGroup(Protocol):
    N: mpz

    def sqr(self, a: mpz) -> mpz: ...
    def mul(self, a: mpz, b: mpz) -> mpz: ...
    def pow(self, base: mpz, exp: mpz) -> mpz: ...


@dataclass(frozen=True, slots=True)
class SimpleVDF:
    group: VDFGroup
    challenge_bits: int = 128

    def solve(self, x: mpz, T: int, delta: int = 1) -> VDFSolution:
        checkpoints = {0: x}
        curr = x

        sqr = self.group.sqr
        for i in range(1, T + 1):
            curr = sqr(curr)
            if i % delta == 0:
                checkpoints[i] = curr

        checkpoints[T] = curr
        return VDFSolution(y=curr, checkpoints=checkpoints)

    def get_intermediate_value(self, checkpoints: dict[int, mpz], target_i: int, delta: int) -> mpz:
        if target_i in checkpoints:
            return checkpoints[target_i]

        base_i = (target_i // delta) * delta
        current = checkpoints[base_i]

        sqr = self.group.sqr
        for _ in range(target_i - base_i):
            current = sqr(current)

        return current

    def compute_next_state(self, x: mpz, y: mpz, mu: mpz, r: mpz, T_left: int, T_right: int) -> tuple[mpz, mpz]:
        x_pow_r = self.group.pow(x, r)
        exp_y = r << (T_right - T_left)
        mu_pow_for_y = self.group.pow(mu, exp_y)

        next_x = self.group.mul(x_pow_r, mu)
        next_y = self.group.mul(mu_pow_for_y, y)

        return next_x, next_y

    def prove(self, x: mpz, y: mpz, T: int, checkpoints: dict[int, mpz], delta: int = 1) -> VDFProof:
        proof = []
        current_checkpoints = checkpoints
        fiat_shamir = FiatShamirTranscript(self.group.N, x, y, T, self.challenge_bits)

        while T > 1:
            T_left = T // 2
            T_right = T - T_left

            mu = self.get_intermediate_value(current_checkpoints, T_left, delta)
            proof.append(mu)

            r = fiat_shamir.get_challenge(x, T, y, mu)

            next_checkpoints = {}
            for i in range(0, T_right + 1, delta):
                val_i = self.get_intermediate_value(current_checkpoints, i, delta)
                val_mid_i = self.get_intermediate_value(current_checkpoints, T_left + i, delta)

                term = self.group.pow(val_i, r)
                next_checkpoints[i] = self.group.mul(term, val_mid_i)

            current_checkpoints = next_checkpoints

            x, y = self.compute_next_state(x, y, mu, r, T_left, T_right)

            T = T_right

        return VDFProof(mu_values=proof)

    def verify(self, x: mpz, y: mpz, T: int, proof: VDFProof) -> bool:
        proof_iter = iter(proof.mu_values)
        fiat_shamir = FiatShamirTranscript(self.group.N, x, y, T, self.challenge_bits)

        while T > 1:
            T_left = T // 2
            T_right = T - T_left

            try:
                mu = next(proof_iter)
            except StopIteration:
                return False

            r = fiat_shamir.get_challenge(x, T, y, mu)
            x, y = self.compute_next_state(x, y, mu, r, T_left, T_right)

            T = T_right

        expected_y = self.group.sqr(x)

        return expected_y == y
