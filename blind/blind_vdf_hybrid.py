import math
from dataclasses import dataclass
from gmpy2 import mpz

from qr_group import QRGroup
from vdf import SimpleVDF
from fiat_shamir import FiatShamirTranscript
from models import VDFProof
from utils import generate_x
from blind.blind_models import HybridPublicParams, HybridServerStepResponse


@dataclass(frozen=True, slots=True)
class HybridBlindVDFSetup:
    group: QRGroup
    simple_vdf: SimpleVDF

    def setup(self, T: int, s: int) -> HybridPublicParams:
        g = generate_x(self.group)

        t_bits = int(math.log2(T))
        step_bits = t_bits // s
        chunk_size = 1 << step_bits

        sol = self.simple_vdf.solve(g, T, delta=1)

        g_steps = {}
        for i in range(1, s + 1):
            sub_T = 1 << ((s - i + 1) * step_bits)
            step_size = sub_T // chunk_size
            for j in range(1, chunk_size + 1):
                step = j * step_size
                g_steps[(i, j)] = sol.checkpoints[step]
        h = sol.y

        return HybridPublicParams(s=s, g=g, h=h, g_steps=g_steps)


@dataclass(frozen=True, slots=True)
class HybridBlindVDFServer:
    simple_vdf: SimpleVDF

    def eval_hybrid_step(self, x_blind: mpz, sub_T: int, chunk_size: int) -> HybridServerStepResponse:
        delta = max(1, sub_T // chunk_size)
        sol = self.simple_vdf.solve(x_blind, sub_T, delta=delta)
        return HybridServerStepResponse(y_blind=sol.y, checkpoints_blind=sol.checkpoints)


@dataclass(frozen=True, slots=True)
class HybridBlindVDFSession:
    group: QRGroup
    params: HybridPublicParams
    simple_vdf: SimpleVDF
    challenge_bits: int = 128

    def run_hybrid_protocol(
            self,
            x: mpz,
            T: int,
            r_list: list[mpz],
            server: HybridBlindVDFServer
    ) -> tuple[mpz, VDFProof]:
        s = self.params.s
        assert len(r_list) == s

        t_bits = int(math.log2(T))
        step_bits = t_bits // s
        chunk_size = 1 << step_bits

        all_proof_mus: list[mpz] = []

        curr_x = x
        curr_T = T
        fiat_shamir = None
        final_y = None
        curr_y = None

        for round_idx in range(1, s + 1):
            r_i = r_list[round_idx - 1]

            x_blind = self.group.mul(curr_x, self.group.pow(self.params.g, r_i))

            sub_T = curr_T
            resp = server.eval_hybrid_step(x_blind, sub_T, chunk_size)

            if round_idx == 1:
                h_pow_r1 = self.group.pow(self.params.h, r_i)
                final_y = self.group.mul(resp.y_blind, self.group.inv(h_pow_r1))
                curr_y = final_y
                fiat_shamir = FiatShamirTranscript(self.group.N, x, final_y, T, self.challenge_bits)

            step_size = sub_T // chunk_size
            unblinded_checkpoints: dict[int, mpz] = {0: curr_x}

            for j in range(1, chunk_size + 1):
                step = j * step_size
                val_blind = resp.checkpoints_blind[step]
                g_val = self.params.g_steps[(round_idx, j)]
                g_pow_r = self.group.pow(g_val, r_i)
                unblinded_checkpoints[step] = self.group.mul(val_blind, self.group.inv(g_pow_r))

            current_checkpoints = unblinded_checkpoints
            for _ in range(step_bits):
                T_left = curr_T // 2
                T_right = curr_T - T_left

                mu = self.simple_vdf.get_intermediate_value(current_checkpoints, T_left, step_size)
                all_proof_mus.append(mu)

                r_challenge = fiat_shamir.get_challenge(curr_x, curr_T, curr_y, mu)

                next_checkpoints = {}
                for chk_i in range(0, T_right + 1, step_size):
                    val_i = self.simple_vdf.get_intermediate_value(current_checkpoints, chk_i, step_size)
                    val_mid_i = self.simple_vdf.get_intermediate_value(current_checkpoints, T_left + chk_i, step_size)

                    term = self.group.pow(val_i, r_challenge)
                    next_checkpoints[chk_i] = self.group.mul(term, val_mid_i)

                current_checkpoints = next_checkpoints

                curr_x, curr_y = self.simple_vdf.compute_next_state(curr_x, curr_y, mu, r_challenge, T_left, T_right)
                curr_T = T_right

        assert final_y is not None
        proof = VDFProof(mu_values=all_proof_mus)
        if not self.simple_vdf.verify(x, final_y, T, proof):
            raise ValueError("Hybrid Blind VDF proof validation failed!")
        return final_y, proof
