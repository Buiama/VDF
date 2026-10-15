import os
import copyreg
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, TypeVar, Any, Protocol
from gmpy2 import from_binary, to_binary, mpz
from helpers.qr_group import QRGroup

T = TypeVar("T")
R = TypeVar("R")


def _reduce_mpz(val: mpz):
    return from_binary, (to_binary(val),)


def get_worker_group() -> QRGroup:
    assert GLOBAL_GROUP is not None
    return GLOBAL_GROUP


GLOBAL_GROUP: QRGroup | None = None
copyreg.pickle(mpz, _reduce_mpz)


def init_worker_process(N_bytes: bytes) -> None:
    global GLOBAL_GROUP
    # get_context().allow_release_gil = True
    copyreg.pickle(mpz, _reduce_mpz)
    N = from_binary(N_bytes)
    GLOBAL_GROUP = QRGroup(N)


def chunkify(items: list[T], num_chunks: int) -> list[list[T]]:
    if not items or num_chunks <= 0:
        return []
    num_chunks = min(len(items), num_chunks)
    k, m = divmod(len(items), num_chunks)
    return [
        items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        for i in range(num_chunks)
    ]


class Executor(Protocol):
    def run_batch(
            self,
            items: list[T],
            worker_func: Callable[[list[T], Any], list[R]],
            extra_args: Any = (),
    ) -> list[R]:
        ...


class SequentialExecutor:
    @staticmethod
    def run_batch(
            items: list[T],
            worker_func: Callable[[list[T], Any], list[R]],
            extra_args: Any = (),
    ) -> list[R]:
        if not items:
            return []
        return worker_func(items, extra_args)


class ProcessParallelExecutor:
    def __init__(self, N: mpz, max_workers: int | None = None, threshold: int = 12):
        self.max_workers = max_workers or (os.cpu_count() or 8)
        self.threshold = threshold
        self.N_bytes = to_binary(N)
        self._pool: ProcessPoolExecutor | None = None

    def start(self) -> None:
        global GLOBAL_GROUP
        if GLOBAL_GROUP is None:
            init_worker_process(self.N_bytes)

        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=init_worker_process,
                initargs=(self.N_bytes,),
            )

    def run_batch(
            self,
            items: list[T],
            worker_func: Callable[[list[T], Any], list[R]],
            extra_args: Any = (),
    ) -> list[R]:
        if not items:
            return []

        if len(items) <= self.threshold or self.max_workers == 1:
            return worker_func(items, extra_args)

        if self._pool is None:
            self.start()

        chunks = chunkify(items, self.max_workers)

        futures = [self._pool.submit(worker_func, chunk, extra_args) for chunk in chunks]

        results: list[R] = []
        for future in futures:
            results.extend(future.result())

        return results

    def shutdown(self, wait: bool = True) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None

    def __enter__(self) -> "ProcessParallelExecutor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)
