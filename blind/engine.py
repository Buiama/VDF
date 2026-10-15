import secrets
from gmpy2 import mpz

from simple.vdf import SimpleVDF, get_intermediate_value
from helpers.executors import get_worker_group
from blind.models import BlindSetup, BlindState


def _unblind_batch_worker(
    chunk_items: list[int],
    extra_args: tuple[dict[int, mpz], dict[int, mpz]]
) -> list[tuple[int, mpz]]:
    checkpoints, round_blinders = extra_args
    group = get_worker_group()

    inv_mod = group.inv
    mul_mod = group.mul

    res = []
    for step in chunk_items:
        c_tilde = checkpoints[step]
        b_k = round_blinders[step]
        res.append((step, mul_mod(c_tilde, inv_mod(b_k))))
    return res


def _compute_unblind_state_batch_worker(
    chunk_steps: list[int],
    extra_args: tuple[dict[int, mpz], mpz]
) -> list[tuple[int, mpz]]:
    setup_blinders, s = extra_args
    group = get_worker_group()

    pow_mod = group.pow

    res = []
    for step in chunk_steps:
        b_k = setup_blinders[step]
        res.append((step, pow_mod(b_k, s)))
    return res


def _compute_r_mid_batch_worker(
    chunk_steps: list[int],
    extra_args: tuple[dict[int, mpz], dict[int, mpz], int]
) -> list[mpz]:
    setup_blinders, r_mid_terms, delta = extra_args
    group = get_worker_group()

    pow_mod = group.pow

    res = []
    for off in chunk_steps:
        exp = r_mid_terms[off]
        b_off = get_intermediate_value(setup_blinders, off, delta, group)
        term = pow_mod(b_off, exp)
        res.append(term)
    return res


class BlinderEngine:
    def __init__(self, vdf: SimpleVDF):
        self.vdf = vdf

    def create_setup(self, g: mpz, T: int, delta: int) -> BlindSetup:
        sol = self.vdf.solve(g, T, delta)
        return BlindSetup(g=g, h=sol.y, T=T, blinders=sol.checkpoints)

    def compute_r(self, setup: BlindSetup, s: mpz) -> mpz:
        return self.vdf.group.pow(setup.g, s)

    def blind(self, setup: BlindSetup, x: mpz) -> tuple[mpz, mpz, mpz]:
        s_bits = self.vdf.group.N.bit_length() + self.vdf.challenge_bits

        s = mpz(secrets.randbits(s_bits))
        r = self.compute_r(setup, s)
        x_tilde = self.blind_input(x, r)
        return x_tilde, r, s

    def compute_r_mid(self, setup_blinders: dict[int, mpz], r_mid_terms: dict[int, mpz], s: mpz, delta: int) -> mpz:
        steps = list(r_mid_terms.keys())
        extra_args = (setup_blinders, r_mid_terms, delta)

        terms = self.vdf.executor.run_batch(steps, _compute_r_mid_batch_worker, extra_args)

        mul_mod = self.vdf.group.mul
        base_prod = mpz(1)
        for term in terms:
            base_prod = mul_mod(base_prod, term)

        return self.vdf.group.pow(base_prod, s)

    def compute_unblind_state(self, setup: BlindSetup, s: mpz, r: mpz) -> BlindState:
        pow_mod = self.vdf.group.pow
        r_T = pow_mod(setup.h, s)

        steps = list(setup.blinders.keys())
        extra_args = (setup.blinders, s)

        batch_results = self.vdf.executor.run_batch(steps, _compute_unblind_state_batch_worker, extra_args)
        blinders = dict(batch_results)

        return BlindState(r=r, r_T=r_T, T=setup.T, blinders=blinders)

    def compute_unblind_state_optimized(self, r: mpz, T: int, delta: int) -> BlindState:
        sol = self.vdf.solve(r, T, delta)
        return BlindState(r=r, r_T=sol.y, T=T, blinders=sol.checkpoints)

    def blind_input(self, x: mpz, r: mpz) -> mpz:
        return self.vdf.group.mul(x, r)

    def unblind_output(self, y_tilde: mpz, r_T: mpz) -> mpz:
        inv_r_T = self.vdf.group.inv(r_T)
        return self.vdf.group.mul(y_tilde, inv_r_T)

    def unblind_checkpoints(self, checkpoints: dict[int, mpz], round_blinders: dict[int, mpz]) -> dict[int, mpz]:
        steps = list(checkpoints.keys())
        extra_args = (checkpoints, round_blinders)

        batch_results = self.vdf.executor.run_batch(steps, _unblind_batch_worker, extra_args)
        return dict(batch_results)
