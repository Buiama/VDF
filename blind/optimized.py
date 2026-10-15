from gmpy2 import mpz

from blind.engine import BlinderEngine
from blind.models import BlindSetup, BlindedSolution, BlindResult
from simple.models import VDFProof
from simple.vdf import SimpleVDF


class OptimizedBlindVDF:
    def __init__(self, vdf: SimpleVDF, setup: BlindSetup, domain_separator: bytes = b"OptimizedBlind_VDF"):
        self.vdf = vdf
        self.setup = setup
        self.blinder = BlinderEngine(vdf)
        self.domain_separator = domain_separator

    def blind(self, x: mpz) -> tuple[mpz, mpz, mpz]:
        return self.blinder.blind(self.setup, x)

    def solve(self, x_tilde: mpz, T: int, delta: int) -> BlindedSolution:
        sol = self.vdf.solve(x_tilde, T, delta)
        return BlindedSolution(y_tilde=sol.y, checkpoints=sol.checkpoints)

    def prove(self, x_tilde: mpz, y_tilde: mpz, checkpoints: dict[int, mpz], T: int, delta: int) -> VDFProof:
        return self.vdf.prove(x_tilde, y_tilde, T, checkpoints, delta)

    def unblind_checkpoints(self, checkpoints: dict[int, mpz], blinders: dict[int, mpz]) -> dict[int, mpz]:
        inv = self.vdf.group.inv
        mul = self.vdf.group.mul

        unblinded: dict[int, mpz] = {}
        for step, c_tilde in checkpoints.items():
            b_k = blinders[step]
            unblinded[step] = mul(c_tilde, inv(b_k))

        return unblinded

    def unblind(self, x: mpz, solution: BlindedSolution, r: mpz, delta: int) -> BlindResult:
        state = self.blinder.compute_unblind_state_optimized(r, self.setup.T, delta)
        y = self.blinder.unblind_output(solution.y_tilde, state.r_T)
        unblinded_checkpoints = self.unblind_checkpoints(solution.checkpoints, state.blinders)
        proof = self.vdf.prove(x, y, state.T, unblinded_checkpoints, delta)
        return BlindResult(y=y, proof=proof)

    def verify(self, x: mpz, y: mpz, T: int, proof: VDFProof, delta: int) -> bool:
        return self.vdf.verify(x, y, T, proof, delta)
