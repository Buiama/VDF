import math
from dataclasses import dataclass
from gmpy2 import mpz

from qr_group import QRGroup
from vdf import SimpleVDF
from fiat_shamir import FiatShamirTranscript
from models import VDFProof
from blind.blind_models import OptimizedPublicParams
from utils import generate_x


@dataclass(frozen=True, slots=True)
class OptimizedBlindVDFSetup:
    group: QRGroup
    simple_vdf: SimpleVDF

    def setup(self, T: int) -> OptimizedPublicParams:
        u = generate_x(self.group)
        sol = self.simple_vdf.solve(u, T, delta=1)

        u_steps = {}
        curr_T = T
        step_idx = 1

        while curr_T > 1:
            T_left = curr_T // 2
            u_steps[step_idx] = sol.checkpoints[T_left]
            curr_T = curr_T - T_left
            step_idx += 1

        v = sol.checkpoints[T]
        return OptimizedPublicParams(u=u, v=v, u_steps=u_steps)


@dataclass(frozen=True, slots=True)
class OptimizedBlindVDFServer:
    group: QRGroup
    simple_vdf: SimpleVDF

    def eval_initial(self, x_blind: mpz, T: int) -> tuple[mpz, mpz]:
        sol = self.simple_vdf.solve(x_blind, T, delta=T // 2)
        y_blind = sol.y
        pi_blind_1 = sol.checkpoints[T // 2]
        return y_blind, pi_blind_1

    def eval_step(self, x_blind_j: mpz, T_current: int) -> mpz:
        sol = self.simple_vdf.solve(x_blind_j, T_current // 2, delta=T_current // 2)
        return sol.y


@dataclass(frozen=True, slots=True)
class OptimizedBlindVDFSession:
    group: QRGroup
    params: OptimizedPublicParams
    simple_vdf: SimpleVDF
    challenge_bits: int = 128

    def run_interactive_protocol(
            self,
            x: mpz,
            T: int,
            client_random_r_list: list[mpz],
            server_runner: OptimizedBlindVDFServer
    ) -> tuple[mpz, VDFProof]:
        t_steps = int(math.log2(T))
        assert len(client_random_r_list) == t_steps

        r1 = client_random_r_list[0]
        x_blind_1 = self.group.mul(x, self.group.pow(self.params.u, r1))

        y_blind, pi_blind_1 = server_runner.eval_initial(x_blind_1, T)

        v_pow_r1 = self.group.pow(self.params.v, r1)
        y = self.group.mul(y_blind, self.group.inv(v_pow_r1))

        u1_pow_r1 = self.group.pow(self.params.u_steps[1], r1)
        pi_1 = self.group.mul(pi_blind_1, self.group.inv(u1_pow_r1))

        proof_mus = [pi_1]

        fiat_shamir = FiatShamirTranscript(self.group.N, x, y, T, self.challenge_bits)

        curr_x = x
        curr_y = y
        curr_T = T
        curr_pi = pi_1

        for j in range(1, t_steps):
            T_left = curr_T // 2
            T_right = curr_T - T_left

            r_challenge = fiat_shamir.get_challenge(curr_x, curr_T, curr_y, curr_pi)

            curr_x, curr_y = self.simple_vdf.compute_next_state(curr_x, curr_y, curr_pi, r_challenge, T_left, T_right)
            curr_T = T_right

            rj = client_random_r_list[j]
            x_blind_j = self.group.mul(curr_x, self.group.pow(self.params.u, rj))

            pi_blind_j = server_runner.eval_step(x_blind_j, curr_T)

            uj_pow_rj = self.group.pow(self.params.u_steps[j + 1], rj)
            curr_pi = self.group.mul(pi_blind_j, self.group.inv(uj_pow_rj))

            proof_mus.append(curr_pi)

        proof = VDFProof(mu_values=proof_mus)

        if not self.simple_vdf.verify(x, y, T, proof):
            raise ValueError("Blind VDF output verification failed!")

        return y, proof
