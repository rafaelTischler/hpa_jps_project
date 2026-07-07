"""Decorator reutilizável para cronometragem de funções."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed_ms(func: Callable[P, R]) -> Callable[P, tuple[R, float]]:
    """Cronometra a execução de ``func`` e retorna ``(resultado, tempo_ms)``.

    O tempo é medido com ``time.perf_counter()`` para alta resolução,
    convertido para milissegundos.

    Example:
        >>> @timed_ms
        ... def work(): return 42
        >>> value, elapsed = work()
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result, elapsed_ms

    return wrapper
