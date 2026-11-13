import itertools


class Counter:

    def __init__(self, start: int = 0):
        self._counter = itertools.count(start)

    def __call__(self) -> int:
        return next(self._counter)
