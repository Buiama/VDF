from gmpy2 import mpz
from dataclasses import replace

from blind.engine import BlinderEngine
from blind.models import BlindSetup, BlindResult
from helpers.fiat_shamir import FiatShamirTranscript
from simple.models import VDFProof
from simple.vdf import SimpleVDF, split_T


class InteractiveBlindVDF:
    def __init__(self, vdf: SimpleVDF, setup: BlindSetup, domain_separator: bytes = b"InteractiveBlind_VDF"):
        self.vdf = replace(vdf, domain_separator=domain_separator)
        self.setup = setup
        self.blinder = BlinderEngine(self.vdf)

    def blind(self, x: mpz) -> tuple[mpz, mpz, mpz]:
        return self.blinder.blind(self.setup, x)

    def unblind_element(self, val_tilde: mpz, s: mpz, step: int, delta: int) -> mpz:
        b_k = self.vdf.get_intermediate_value(self.setup.blinders, step, delta)
        r_term = self.vdf.group.pow(b_k, s)
        return self.blinder.unblind_output(val_tilde, r_term)

    def solve_initial(self, x_tilde: mpz, T: int, delta: int) -> tuple[mpz, mpz]:
        sol = self.vdf.solve(x_tilde, T, delta)
        T_left, _ = split_T(T, delta)
        mu_tilde = self.vdf.get_intermediate_value(sol.checkpoints, T_left, delta)
        return sol.y, mu_tilde

    def solve_step(self, x_tilde: mpz, T_left: int) -> mpz:
        return self.vdf.evaluate_squarings(x_tilde, T_left)

    def prove_interactive(self, x: mpz, T: int, delta: int) -> BlindResult:
        T_left, T_right = split_T(T, delta)

        x_tilde_0, _, s_0 = self.blind(x)
        y_tilde_0, mu_tilde_0 = self.solve_initial(x_tilde_0, T, delta)

        y = self.unblind_element(y_tilde_0, s_0, T, delta)
        mu_0 = self.unblind_element(mu_tilde_0, s_0, T_left, delta)

        fiat_shamir = FiatShamirTranscript(self.vdf.group.N, x, y, T, self.vdf.domain_separator, self.vdf.challenge_bits)

        proof_mus: list[mpz] = [mu_0]
        curr_x = x
        curr_y = y

        while T > 1:
            T_l, T_r = split_T(T, delta)

            r_challenge = fiat_shamir.get_challenge(curr_x, T, curr_y, mu_0)

            curr_x, curr_y = self.vdf.compute_next_state(curr_x, curr_y, mu_0, r_challenge, T_l, T_r)
            T = T_r

            if T <= 1:
                break

            T_next_l, _ = split_T(T, delta)
            x_tilde_k, _, s_k = self.blind(curr_x)

            mu_tilde_k = self.solve_step(x_tilde_k, T_next_l)

            mu_0 = self.unblind_element(mu_tilde_k, s_k, T_next_l, delta)
            proof_mus.append(mu_0)

        proof = VDFProof(mu_values=proof_mus)

        return BlindResult(y=y, proof=proof)

    def verify(self, x: mpz, y: mpz, T: int, proof: VDFProof, delta: int) -> bool:
        return self.vdf.verify(x, y, T, proof, delta)
