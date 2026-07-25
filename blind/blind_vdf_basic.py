import math
from dataclasses import dataclass
from gmpy2 import mpz

from qr_group import QRGroup
from vdf import SimpleVDF
from utils import generate_x
from models import VDFProof
from blind.blind_models import BasicPublicParams, UnblindingState, BasicServerResponse


@dataclass(frozen=True, slots=True)
class BasicBlindVDFSetup:
    group: QRGroup
    simple_vdf: SimpleVDF

    def setup(self, T: int) -> BasicPublicParams:
        delta = math.isqrt(T)
        a = generate_x(self.group)
        sol = self.simple_vdf.solve(a, T, delta=delta)

        return BasicPublicParams(a=a, a_checkpoints=sol.checkpoints)


@dataclass(frozen=True, slots=True)
class BasicBlindVDFClient:
    group: QRGroup
    params: BasicPublicParams
    simple_vdf: SimpleVDF

    def blind(self, x: mpz, secret_r: mpz) -> tuple[mpz, UnblindingState]:
        a_pow_r = self.group.pow(self.params.a, secret_r)
        x_blind = self.group.mul(x, a_pow_r)
        return x_blind, UnblindingState(r=secret_r)

    def unblind_and_prove(
            self,
            x: mpz,
            T: int,
            state: UnblindingState,
            response: BasicServerResponse
    ) -> tuple[mpz, VDFProof]:
        delta = math.isqrt(T)
        r = state.r

        a_T_pow_r = self.group.pow(self.params.a_checkpoints[T], r)
        inv_a_T_pow_r = self.group.inv(a_T_pow_r)
        y = self.group.mul(response.y_blind, inv_a_T_pow_r)

        unblinded_checkpoints: dict[int, mpz] = {0: x, T: y}
        for step, val_blind in response.checkpoints_blind.items():
            a_step_pow_r = self.group.pow(self.params.a_checkpoints[step], r)
            inv_a_step = self.group.inv(a_step_pow_r)
            unblinded_checkpoints[step] = self.group.mul(val_blind, inv_a_step)

        proof = self.simple_vdf.prove(x, y, T, unblinded_checkpoints, delta=delta)
        if not self.simple_vdf.verify(x, y, T, proof):
            raise ValueError("Blind VDF output verification failed!")
        return y, proof


@dataclass(frozen=True, slots=True)
class BasicBlindVDFServer:
    simple_vdf: SimpleVDF

    def solve_blind(self, x_blind: mpz, T: int) -> BasicServerResponse:
        delta = math.isqrt(T)
        sol = self.simple_vdf.solve(x_blind, T, delta=delta)
        return BasicServerResponse(y_blind=sol.y, checkpoints_blind=sol.checkpoints)
