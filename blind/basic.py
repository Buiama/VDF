from gmpy2 import mpz
from dataclasses import replace

from blind.engine import BlinderEngine
from blind.models import BlindSetup, BlindedSolution, BlindResult
from simple.models import VDFProof
from simple.vdf import SimpleVDF


class BasicBlindVDF:
    def __init__(self, vdf: SimpleVDF, setup: BlindSetup, domain_separator: bytes = b"BasicBlind_VDF"):
        self.vdf = replace(vdf, domain_separator=domain_separator)
        self.setup = setup
        self.blinder = BlinderEngine(self.vdf)

    def blind(self, x: mpz) -> tuple[mpz, mpz, mpz]:
        return self.blinder.blind(self.setup, x)

    def solve(self, x_tilde: mpz, T: int, delta: int) -> BlindedSolution:
        sol = self.vdf.solve(x_tilde, T, delta)
        return BlindedSolution(y_tilde=sol.y, checkpoints=sol.checkpoints)

    def prove(self, x_tilde: mpz, y_tilde: mpz, checkpoints: dict[int, mpz], T: int, delta: int) -> VDFProof:
        return self.vdf.prove(x_tilde, y_tilde, T, checkpoints, delta)

    def unblind(self, x: mpz, solution: BlindedSolution, r: mpz, s: mpz, delta: int) -> BlindResult:
        state = self.blinder.compute_unblind_state(self.setup, s, r)
        y = self.blinder.unblind_output(solution.y_tilde, state.r_T)
        unblinded_checkpoints = self.blinder.unblind_checkpoints(solution.checkpoints, state.blinders)
        proof = self.vdf.prove(x, y, state.T, unblinded_checkpoints, delta)
        return BlindResult(y=y, proof=proof)

    def verify(self, x: mpz, y: mpz, T: int, proof: VDFProof, delta: int) -> bool:
        return self.vdf.verify(x, y, T, proof, delta)
